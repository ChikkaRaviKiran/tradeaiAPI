"""AWS Lightsail proxy manager for multi-account IP isolation.

Ported verbatim from OptionSelling (c:\\OptionSelling\\backend\\app\\infra\\proxy_manager.py).
Only INSTANCE_PREFIX / app-tag differ so instances don't collide when both
apps share an AWS account.

Each broker account requires its own outbound IP:
  * AngelOne SmartAPI whitelists IPs per api_key.
  * Kite Connect whitelists a single IP per app in the developer console
    (Zerodha rejects orders / OAuth when traffic comes from a non-whitelisted
    IP, which collides if multiple Kite accounts share the main server IP).

This module auto-provisions a Lightsail Nano instance ($5/mo) running
microsocks SOCKS5 proxy, restricted by firewall to the main server IP only.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from app.core.config import settings

logger = logging.getLogger(__name__)

# microsocks setup script — runs on first boot
_USER_DATA_TEMPLATE = """#!/bin/bash
set -e
apt-get update -y
apt-get install -y gcc make git
cd /opt
git clone https://github.com/rofl0r/microsocks.git
cd microsocks
make
# Start microsocks with auth on port 1080
# -u user -P password -p port
./microsocks -p 1080 -u {proxy_user} -P {proxy_pass} &
# Persist across reboots
cat > /etc/systemd/system/microsocks.service <<EOF
[Unit]
Description=microsocks SOCKS5 proxy
After=network.target
[Service]
ExecStart=/opt/microsocks/microsocks -p 1080 -u {proxy_user} -P {proxy_pass}
Restart=always
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable microsocks
systemctl start microsocks
"""

PROXY_PORT = 1080
INSTANCE_PREFIX = "tradeai-proxy"


class ProxyManager:
    """Manages Lightsail proxy instances for broker accounts."""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not settings.aws_access_key_id or not settings.aws_secret_access_key:
                raise RuntimeError("AWS credentials not configured in .env")
            self._client = boto3.client(
                "lightsail",
                region_name=settings.aws_region,
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
            )
        return self._client

    def _instance_name(self, account_id: int) -> str:
        return f"{INSTANCE_PREFIX}-{account_id}"

    def create_proxy(self, account_id: int, proxy_user: str = "proxy",
                     proxy_pass: str = "proxy") -> dict:
        """Create a Lightsail proxy instance for an account.

        Returns dict with keys: instance_name, public_ip, proxy_url, status
        """
        name = self._instance_name(account_id)

        # Check if already exists
        try:
            resp = self.client.get_instance(instanceName=name)
            inst = resp["instance"]
            ip = inst.get("publicIpAddress", "")
            state = inst["state"]["name"]
            proxy_url = f"socks5://{proxy_user}:{proxy_pass}@{ip}:{PROXY_PORT}" if ip else ""
            logger.info("Proxy instance %s already exists (state=%s, ip=%s)", name, state, ip)
            return {
                "instance_name": name,
                "public_ip": ip,
                "proxy_url": proxy_url,
                "status": state,
                "already_existed": True,
            }
        except ClientError as e:
            if "NotFoundException" not in str(e):
                raise

        user_data = _USER_DATA_TEMPLATE.format(
            proxy_user=proxy_user,
            proxy_pass=proxy_pass,
        )

        logger.info("Creating Lightsail proxy instance %s ...", name)
        self.client.create_instances(
            instanceNames=[name],
            availabilityZone=f"{settings.aws_region}a",
            blueprintId="ubuntu_22_04",
            bundleId="nano_3_1",  # $5/mo (includes public IPv4)
            userData=user_data,
        )

        # Tag the instance (separate call — TagResource permission may be missing)
        try:
            self.client.tag_resource(
                resourceName=name,
                tags=[
                    {"key": "app", "value": "tradeai"},
                    {"key": "account_id", "value": str(account_id)},
                ],
            )
        except ClientError:
            logger.warning("Could not tag instance %s (TagResource permission missing — non-critical)", name)

        # Wait for instance to be running and get IP
        ip = self._wait_for_ip(name, timeout=180)

        # Open port 1080 only from our main server
        self._configure_firewall(name)

        proxy_url = f"socks5://{proxy_user}:{proxy_pass}@{ip}:{PROXY_PORT}" if ip else ""

        logger.info("Proxy instance %s ready — ip=%s", name, ip)
        return {
            "instance_name": name,
            "public_ip": ip,
            "proxy_url": proxy_url,
            "status": "running",
            "already_existed": False,
        }

    def _wait_for_ip(self, name: str, timeout: int = 180) -> str:
        """Poll Lightsail until instance has a public IP."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = self.client.get_instance(instanceName=name)
                ip = resp["instance"].get("publicIpAddress", "")
                state = resp["instance"]["state"]["name"]
                if ip and state == "running":
                    return ip
            except ClientError:
                pass
            time.sleep(5)
        logger.warning("Timed out waiting for IP on %s", name)
        return ""

    def _configure_firewall(self, name: str):
        """Lock down the instance firewall to only allow our main server."""
        try:
            # Close default SSH and add SOCKS5 rule
            self.client.put_instance_public_ports(
                instanceName=name,
                portInfos=[
                    {
                        "fromPort": 22,
                        "toPort": 22,
                        "protocol": "tcp",
                        "cidrs": [f"{settings.main_server_ip}/32"],
                    },
                    {
                        "fromPort": PROXY_PORT,
                        "toPort": PROXY_PORT,
                        "protocol": "tcp",
                        "cidrs": [f"{settings.main_server_ip}/32"],
                    },
                ],
            )
            logger.info("Firewall configured for %s — only %s allowed", name, settings.main_server_ip)
        except Exception:
            logger.exception("Failed to configure firewall for %s", name)

    def delete_proxy(self, account_id: int):
        """Delete the Lightsail proxy instance for an account."""
        name = self._instance_name(account_id)
        try:
            self.client.delete_instance(instanceName=name, forceDeleteAddOns=True)
            logger.info("Deleted proxy instance %s", name)
        except ClientError as e:
            if "NotFoundException" in str(e):
                logger.info("Proxy instance %s not found (already deleted?)", name)
            else:
                raise

    def get_status(self, account_id: int) -> dict:
        """Get status of a proxy instance."""
        name = self._instance_name(account_id)
        try:
            resp = self.client.get_instance(instanceName=name)
            inst = resp["instance"]
            return {
                "instance_name": name,
                "public_ip": inst.get("publicIpAddress", ""),
                "state": inst["state"]["name"],
                "blueprint": inst.get("blueprintName", ""),
                "bundle": inst.get("bundleId", ""),
                "created_at": str(inst.get("createdAt", "")),
            }
        except ClientError as e:
            if "NotFoundException" in str(e):
                return {"instance_name": name, "state": "not_found"}
            raise

    def list_proxies(self) -> list[dict]:
        """List all proxy instances managed by this module."""
        try:
            resp = self.client.get_instances()
            proxies = []
            for inst in resp.get("instances", []):
                if inst["name"].startswith(INSTANCE_PREFIX):
                    proxies.append({
                        "instance_name": inst["name"],
                        "public_ip": inst.get("publicIpAddress", ""),
                        "state": inst["state"]["name"],
                        "created_at": str(inst.get("createdAt", "")),
                    })
            return proxies
        except Exception:
            logger.exception("Failed to list proxy instances")
            return []


# Singleton
proxy_manager = ProxyManager()
