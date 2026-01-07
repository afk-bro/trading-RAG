#!/usr/bin/env python3
import socket

hostname = "db.sypwqiozmpjhtthkjmvq.supabase.co"
try:
    ip = socket.gethostbyname(hostname)
    print(f"Resolved {hostname} to {ip}")
except Exception as e:
    print(f"Failed to resolve {hostname}: {e}")
