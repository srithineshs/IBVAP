import cv2
import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

urls = [
    "rtsp://admin:admin@123@10.245.106.240:554/Streaming/Channels/102",
    "rtsp://admin:admin%40123@10.245.106.240:554/Streaming/Channels/102",
    "rtsp://admin:admin@123@10.245.106.240:554/Streaming/Channels/101",
    "rtsp://admin:admin%40123@10.245.106.240:554/Streaming/Channels/101",
    "rtsp://admin:admin@123@10.245.106.240:554/h264Preview_01_main",
    "rtsp://admin:admin%40123@10.245.106.240:554/h264Preview_01_main",
    "rtsp://admin:admin@123@10.245.106.240:554/live/ch0",
    "rtsp://admin:admin%40123@10.245.106.240:554/live/ch0"
]

print("Testing RTSP connections with password...")
for u in urls:
    print(f"\nTrying: {u}")
    cap = cv2.VideoCapture(u, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            print(f"[*** SUCCESS ***] Connected to: {u}")
            print(f"Frame resolution: {frame.shape[1]}x{frame.shape[0]}")
            cv2.imwrite("scratch/camera_live_snapshot.jpg", frame)
            break
        else:
            print("[!] Opened but failed to read frame.")
    else:
        print("[!] Failed to open stream.")
