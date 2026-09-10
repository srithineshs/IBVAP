import socket
import concurrent.futures
import cv2

def check_port(ip, port, timeout=0.5):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            if s.connect_ex((ip, port)) == 0:
                return (ip, port)
    except:
        pass
    return None

def test_rtsp_or_http(url):
    try:
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                return True, frame.shape
    except:
        pass
    return False, None

def main():
    base_prefix = "10.245.106."
    ports = [554, 80, 8080, 8000, 8554, 4747, 8081, 5000, 1935]
    
    print(f"Scanning {base_prefix}1-254 for open camera ports: {ports}...")
    
    open_targets = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        futures = []
        for i in range(1, 255):
            ip = f"{base_prefix}{i}"
            for p in ports:
                futures.append(executor.submit(check_port, ip, p))
                
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res:
                open_targets.append(res)
                print(f"[+] Open port found: {res[0]}:{res[1]}")
                
    print(f"\nDiscovered {len(open_targets)} open targets.")
    
    # Test common video streams
    for ip, port in open_targets:
        test_urls = [
            f"rtsp://{ip}:{port}/h264Preview_01_main",
            f"rtsp://{ip}:{port}/live/ch0",
            f"rtsp://{ip}:{port}/stream1",
            f"rtsp://{ip}:{port}/1",
            f"rtsp://{ip}:{port}/",
            f"http://{ip}:{port}/video",
            f"http://{ip}:{port}/mjpeg",
            f"http://{ip}:{port}/video.mjpg",
            f"http://{ip}:{port}/live",
            f"http://{ip}:{port}/stream",
            f"http://{ip}:{port}/feed"
        ]
        if port == 4747: # DroidCam
            test_urls.insert(0, f"http://{ip}:4747/video")
            test_urls.insert(0, f"http://{ip}:4747/mjpegfeed?640x480")
        if port == 8080: # IP Webcam Android
            test_urls.insert(0, f"http://{ip}:8080/video")
            test_urls.insert(0, f"http://{ip}:8080/shot.jpg")

        for url in test_urls:
            ok, shape = test_rtsp_or_http(url)
            if ok:
                print(f"[!!!] SUCCESS! Found active camera stream: {url} (Resolution: {shape})")

if __name__ == "__main__":
    main()
