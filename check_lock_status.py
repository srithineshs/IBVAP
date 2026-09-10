import urllib.request
import re

url = "http://10.245.106.240/doc/page/login.asp"
try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=3) as resp:
        content = resp.read().decode('utf-8', errors='ignore')
        print("Camera HTTP 200 OK - Reachable!")
        print("Length:", len(content))
except Exception as e:
    print(f"Camera HTTP Error: {e}")

# Check RTSP port
import socket
s = socket.socket()
s.settimeout(2.0)
res = s.connect_ex(('10.245.106.240', 554))
print(f"RTSP Port 554 open: {res == 0}")
s.close()
