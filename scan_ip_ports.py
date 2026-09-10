import socket
import concurrent.futures
import cv2

targets = ["10.245.106.111", "10.245.106.147", "10.245.106.154", "10.245.106.210", "10.245.106.240"]

common_ports = [
    80, 81, 82, 88, 554, 10554, 8554, 5554, 7070, 8000, 8080, 8081, 8082, 8090, 8888, 8899,
    37777, 34567, 4747, 5000, 6667, 1935, 9000, 9999, 10000, 2020, 8008, 8443, 443, 23, 22,
    5544, 3000, 8889, 7447, 10001, 8088, 8181
]

def scan_port(ip, port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            if s.connect_ex((ip, port)) == 0:
                return (ip, port)
    except:
        pass
    return None

print(f"Scanning target IPs {targets} on common camera ports...")
open_ports = {}
with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    futures = [executor.submit(scan_port, ip, p) for ip in targets for p in common_ports]
    for f in concurrent.futures.as_completed(futures):
        res = f.result()
        if res:
            ip, port = res
            open_ports.setdefault(ip, []).append(port)
            print(f"[+] Found {ip}:{port}")

print("\n--- Summary of Open Ports ---")
for ip, ports in open_ports.items():
    print(f"{ip}: {sorted(ports)}")

# Also test full range 1-10000 for 10.245.106.210, 10.245.106.240, 10.245.106.147
for target in ["10.245.106.210", "10.245.106.240", "10.245.106.147"]:
    print(f"\nScanning ports 1-10000 on {target}...")
    found = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        futs = [executor.submit(scan_port, target, p) for p in range(1, 10000)]
        for f in concurrent.futures.as_completed(futs):
            res = f.result()
            if res:
                found.append(res[1])
                print(f"  [+] {target}:{res[1]}")
    print(f"{target} open ports (1-10000): {sorted(found)}")
