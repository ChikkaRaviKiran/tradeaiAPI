#!/bin/bash
echo "--- TCP probe ---"
nc -zv 13.233.86.23 1080 2>&1 || true

echo
echo "--- SOCKS5 probe (greeting 05 01 00) ---"
printf '\x05\x01\x00' | nc -w 3 13.233.86.23 1080 | xxd | head -3 || true

echo
echo "--- HTTP CONNECT probe ---"
printf 'CONNECT api.kite.trade:443 HTTP/1.1\r\nHost: api.kite.trade:443\r\n\r\n' | nc -w 3 13.233.86.23 1080 | head -3 || true
