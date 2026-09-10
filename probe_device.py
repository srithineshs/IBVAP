import socket
import urllib.request
import cv2

def probe_http(ip, port):
    url = f"http://{ip}:{port}/"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=2) as resp:
            content = resp.read()[:500]
            headers = dict(resp.headers)
            return True, headers, content
    except Exception as e:
        return False, str(e), None

def scan_ports(ip, port_list):
    open_ports = []
    for p in port_list:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.4)
                if s.connect_ex((ip, p)) == 0:
                    open_ports.append(p)
        except:
            pass
    return open_ports

targets = ["10.245.106.111", "10.245.106.154"]
common_ports = [80, 81, 88, 554, 10554, 8554, 8000, 8080, 8081, 8082, 8888, 8899, 37777, 34567, 4747, 5000, 5554, 6667, 7070, 9000, 1935]

for target in targets:
    print(f"\nScanning {target}...")
    opened = scan_ports(target, common_ports)
    print(f"Open ports on {target}: {opened}")
    for p in opened:
        if p in [80, 81, 88, 8000, 8080, 8081, 8888, 4747, 5000]:
            ok, h, c = probe_http(target, p)
            print(f"  HTTP probe {target}:{p} -> OK={ok}")
            if ok:
                print(f"  Headers: {h}")
                print(f"  Snippet: {c[:200]}")
