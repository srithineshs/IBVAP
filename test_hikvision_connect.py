import cv2
import os
import time

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Tested variants for Hikvision RTSP
rtsp_candidates = [
    # 1. Main stream (101) & Sub stream (102) with encoded @
    "rtsp://admin:admin%40123@10.245.106.240:554/Streaming/Channels/101",
    "rtsp://admin:admin%40123@10.245.106.240:554/Streaming/Channels/102",
    "rtsp://admin:admin%40123@10.245.106.240:554/Streaming/Channels/1",
    "rtsp://admin:admin%40123@10.245.106.240:554/Streaming/Channels/2",
    "rtsp://admin:admin%40123@10.245.106.240:554/h264/ch1/main/av_stream",
    "rtsp://admin:admin%40123@10.245.106.240:554/h264/ch1/sub/av_stream",
    "rtsp://admin:admin%40123@10.245.106.240:554/ISAPI/Streaming/channels/101",
    "rtsp://admin:admin%40123@10.245.106.240:554/ISAPI/Streaming/channels/102",
    "rtsp://admin:admin%40123@10.245.106.240:554/ch1/main/av_stream",
    "rtsp://admin:admin%40123@10.245.106.240:554/live/ch0",
    "rtsp://admin:admin%40123@10.245.106.240:554/",
]

print("Testing RTSP streams on Hikvision 10.245.106.240...")
found_working_url = None

for url in rtsp_candidates:
    print(f"\n[?] Trying: {url}")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2500)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2500)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            print(f"[*** SUCCESS! ***] Connected to camera stream!")
            print(f"URL: {url}")
            print(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")
            cv2.imwrite("scratch/camera_active_frame.jpg", frame)
            found_working_url = url
            break
        else:
            print("[-] Stream opened but no frame received.")
    else:
        print("[-] Stream failed to open.")

if found_working_url:
    print(f"\n>>> READY TO RUN PIPELINE WITH: {found_working_url} <<<")
else:
    print("\nCould not establish RTSP session. Testing socket directly...")
