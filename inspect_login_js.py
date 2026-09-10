import urllib.request

base = "http://10.245.106.240"
files = [
    "/script/lib/seajs/config/sea-config.js",
    "/script/login.js",
    "/script/common.js"
]

for f in files:
    try:
        url = base + f
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=2) as resp:
            content = resp.read().decode('utf-8', errors='ignore')
            print(f"=== {f} (len {len(content)}) ===")
            print(content[:600])
    except Exception as e:
        print(f"Error {f}: {e}")
