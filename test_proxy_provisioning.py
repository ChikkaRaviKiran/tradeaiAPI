import unittest
from unittest.mock import MagicMock, patch

import requests
from botocore.exceptions import ClientError

from app.data.dhan_client import DhanClient
from app.execution.dhan_broker import DhanBroker
from app.infra.proxy_manager import ProxyManager


def not_found(operation):
    return ClientError({"Error": {"Code": "NotFoundException", "Message": "missing"}}, operation)


class DhanProxyTests(unittest.TestCase):
    def test_order_transport_uses_configured_proxy(self):
        proxy = "socks5h://user:pass@127.0.0.1:1080"
        client = DhanClient("test", "test-token", proxy)
        session = client._dhan.dhan_http.session
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"orderId":"test"}'
        with patch.object(session, "send", return_value=response) as send:
            result = client._dhan.place_order(
                security_id="1", exchange_segment="NSE_FNO", transaction_type="BUY",
                quantity=1, order_type="MARKET", product_type="INTRADAY", price=0,
            )
        self.assertEqual(result["status"], "success")
        self.assertEqual(send.call_args.kwargs["proxies"]["https"], proxy)
        self.assertFalse(session.trust_env)

    def test_missing_sdk_transport_fails_closed(self):
        client = DhanClient.__new__(DhanClient)
        client._dhan = object()
        with self.assertRaisesRegex(RuntimeError, "refusing direct"):
            client._install_proxy_on_sdk_session("socks5h://localhost:1080")

    @patch("app.execution.dhan_broker.DhanClient")
    @patch("app.db.broker_credentials.get_dhan_credentials")
    def test_legacy_broker_refreshes_when_proxy_changes(self, credentials, client):
        credentials.return_value = {
            "client_id": "test", "access_token": "token",
            "proxy_url": "socks5h://old:1080",
        }
        broker = DhanBroker()
        self.assertEqual(client.call_args.kwargs["proxy_url"], "socks5h://old:1080")
        credentials.return_value["proxy_url"] = "socks5h://new:1080"
        broker._ensure_fresh_client()
        self.assertEqual(client.call_count, 2)
        self.assertEqual(client.call_args.kwargs["proxy_url"], "socks5h://new:1080")


class ProxyManagerTests(unittest.TestCase):
    def setUp(self):
        self.manager = ProxyManager()
        self.client = MagicMock()
        self.manager._client = self.client
        self.client.get_instance.side_effect = not_found("GetInstance")
        self.client.get_static_ip.side_effect = [
            not_found("GetStaticIp"), {"staticIp": {"ipAddress": "192.0.2.4"}}
        ]
        paginator = self.client.get_paginator.return_value
        paginator.paginate.return_value = [{
            "bundles": [
                {"bundleId": "ipv6", "isActive": True, "instanceType": "nano",
                 "publicIpv4AddressCount": 0, "supportedPlatforms": ["LINUX_UNIX"], "price": 3.5},
                {"bundleId": "nano_3_2", "isActive": True, "instanceType": "nano",
                 "publicIpv4AddressCount": 1, "supportedPlatforms": ["LINUX_UNIX"], "price": 5.0},
            ]
        }]

    @patch("app.infra.proxy_manager.settings")
    def test_provisions_ipv4_static_and_checks_readiness(self, settings):
        settings.aws_region = "ap-southeast-2"
        settings.main_server_ip = "192.0.2.1"
        with patch.object(self.manager, "_wait_for_ip", return_value="192.0.2.3"), \
             patch.object(self.manager, "_wait_until_ready") as ready:
            result = self.manager.create_proxy(5, "proxy", "strong-secret")
        self.assertEqual(self.client.create_instances.call_args.kwargs["bundleId"], "nano_3_2")
        self.client.attach_static_ip.assert_called_once_with(
            staticIpName="tradeai-proxy-5-ip", instanceName="tradeai-proxy-5"
        )
        ready.assert_called_once_with(result["proxy_url"], "192.0.2.4")
        self.assertEqual(result["public_ip"], "192.0.2.4")

    @patch("app.infra.proxy_manager.settings")
    def test_firewall_failure_does_not_report_success(self, settings):
        settings.aws_region = "ap-southeast-2"
        settings.main_server_ip = "192.0.2.1"
        self.client.put_instance_public_ports.side_effect = RuntimeError("firewall failed")
        with patch.object(self.manager, "_wait_for_ip", return_value="192.0.2.3"):
            with self.assertRaisesRegex(RuntimeError, "firewall failed"):
                self.manager.create_proxy(5, "proxy", "strong-secret")

    @patch("app.infra.proxy_manager.settings")
    def test_invalid_server_ip_prevents_instance_creation(self, settings):
        settings.main_server_ip = ""
        with self.assertRaises(ValueError):
            self.manager.create_proxy(5, "proxy", "strong-secret")
        self.client.create_instances.assert_not_called()


if __name__ == "__main__":
    unittest.main()
