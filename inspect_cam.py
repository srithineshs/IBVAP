import urllib.request
import re
import cv2

url = "http://10.245.106.111/"
try:
    resp = urllib.request.urlopen(url, timeout=3)
    content = resp.read().decode('utf-8', errors='ignore')
    with open("scratch/camera_webpage.html", "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved webpage ({len(content)} bytes)")
    
    titles = re.findall(r'<title>(.*?)</title>', content, re.I)
    print("Page Title:", titles)
    
    # Check for stream endpoints or video tags
    streams = re.findall(r'(\/[a-zA-Z0-9_\-\.\/]+(?:mjpg|mjpeg|video|stream|live|cgi|sdp|flv|h264|media))', content, re.I)
    print("Found stream paths in HTML:", set(streams))
except Exception as e:
    print(f"Error fetching {url}: {e}")
