#!/bin/bash
ssh -i /tmp/jump.pem -o StrictHostKeyChecking=no ubuntu@13.233.86.23 'bash -s' << 'REMOTE'
echo "net.ipv6.conf.all.disable_ipv6 = 1"     | sudo tee /etc/sysctl.d/99-disable-ipv6.conf
echo "net.ipv6.conf.default.disable_ipv6 = 1" | sudo tee -a /etc/sysctl.d/99-disable-ipv6.conf
echo "net.ipv6.conf.lo.disable_ipv6 = 1"      | sudo tee -a /etc/sysctl.d/99-disable-ipv6.conf
sudo sysctl -p /etc/sysctl.d/99-disable-ipv6.conf
echo "--- microsocks restart ---"
sudo systemctl restart microsocks
sleep 2
sudo systemctl is-active microsocks
echo "--- direct egress (should be IPv4) ---"
curl -s --max-time 5 https://api64.ipify.org && echo
REMOTE
