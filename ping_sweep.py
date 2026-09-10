import subprocess
import concurrent.futures
import re

def ping(ip):
    try:
        # Use Windows ping: -n 1 -w 300
        res = subprocess.run(["ping", "-n", "1", "-w", "300", ip], capture_output=True, text=True)
        if "TTL=" in res.stdout or "Reply from" in res.stdout:
            return ip
    except:
        pass
    return None

def main():
    ips = [f"10.245.106.{i}" for i in range(1, 255)]
    print(f"Pinging {len(ips)} IPs on 10.245.106.0/24...")
    active_ips = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        results = executor.map(ping, ips)
        for r in results:
            if r:
                active_ips.append(r)
                print(f"[+] Active IP: {r}")
                
    print(f"\nAll active IPs ({len(active_ips)}): {active_ips}")
    
    # Read ARP table after ping sweep
    arp_res = subprocess.run(["C:\\Windows\\System32\\arp.exe", "-a"], capture_output=True, text=True)
    print("\n--- Current ARP Table ---")
    print(arp_res.stdout)

if __name__ == "__main__":
    main()
