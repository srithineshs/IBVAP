import sys
import socket
import concurrent.futures
import cv2

sys.stdout.reconfigure(line_buffering=True)

# Common camera / video streaming ports
CAMERA_PORTS = [
    554,    # RTSP Standard
    8554,   # RTSP Alternative
    10554,  # RTSP Alternative
    80,     # HTTP / Web / MJPEG / ONVIF
    8080,   # HTTP Alternative / IP Webcam / DroidCam
    4747,   # DroidCam
    8081,   # Motion / MJPEG
    8000,   # Hikvision / DVR / HTTP
    37777,  # Dahua DVR/NVR
    34567,  # XM / Xiongmai IP Camera
    8899,   # ONVIF WS-Discovery / Service
    5000,   # Synology / Python Flask stream
    8090,   # IPCamera stream
    1935,   # RTMP
    8888    # DVR Web
]

def scan_target(ip, port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.25)
        res = s.connect_ex((ip, port))
        s.close()
        if res == 0:
            return (ip, port)
    except:
        pass
    return None

def test_rtsp_stream(url):
    try:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                return True, frame.shape
    except Exception as e:
        pass
    return False, None

def main():
    subnet = "10.245.106."
    print(f"[*] Scanning subnet {subnet}1 - {subnet}254 for camera ports...")
    
    tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
        for host in range(1, 255):
            ip = f"{subnet}{host}"
            for p in CAMERA_PORTS:
                tasks.append(executor.submit(scan_target, ip, p))
                
        open_services = []
        for f in concurrent.futures.as_completed(tasks):
            r = f.result()
            if r:
                open_services.append(r)
                print(f"[+] Discovered open port: {r[0]}:{r[1]}")

    print(f"\n[+] Total open ports discovered: {len(open_services)}")
    
    # Common RTSP / HTTP path templates for IP Cameras
    path_templates = [
        "",
        "/live/ch0",
        "/h264Preview_01_main",
        "/h264Preview_01_sub",
        "/stream1",
        "/stream2",
        "/onvif1",
        "/live",
        "/video",
        "/mjpeg",
        "/cam/realmonitor?channel=1&subtype=0", # Dahua
        "/Streaming/Channels/101",              # Hikvision
        "/Streaming/Channels/102",
        "/11",
        "/12",
        "/ch0_0.264",
        "/mjpegfeed?640x480"
    ]
    
    for ip, port in open_services:
        print(f"\n[*] Testing video feeds on {ip}:{port}...")
        for path in path_templates:
            # RTSP
            if port in [554, 8554, 10554]:
                url = f"rtsp://{ip}:{port}{path}"
                ok, shape = test_rtsp_stream(url)
                if ok:
                    print(f"[*** SUCCESS ***] Connected to RTSP stream: {url} -> Resolution: {shape}")
            # HTTP
            if port in [80, 8080, 8081, 4747, 8000, 5000]:
                url = f"http://{ip}:{port}{path}"
                ok, shape = test_rtsp_stream(url)
                if ok:
                    print(f"[*** SUCCESS ***] Connected to HTTP video stream: {url} -> Resolution: {shape}")

if __name__ == "__main__":
    main()
