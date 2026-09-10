import urllib.request
import re

url = "http://10.245.106.240/doc/page/login.asp"
try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=3) as resp:
        content = resp.read().decode('utf-8', errors='ignore')
        print("Login ASP Content Length:", len(content))
        print("Login ASP Snippet:\n", content[:1500])
        
        # Look for script tags
        scripts = re.findall(r'src=[\"\'](.*?)[\"\']', content)
        print("\nScripts:", scripts)
except Exception as e:
    print(f"Error: {e}")
