import subprocess

urls = [
    "rtsp://admin:admin@123@10.245.106.240:554/Streaming/Channels/101",
    "rtsp://admin:admin%40123@10.245.106.240:554/Streaming/Channels/101",
    "rtsp://admin:admin@123@10.245.106.240:554/Streaming/Channels/102",
    "rtsp://admin:admin%40123@10.245.106.240:554/Streaming/Channels/102",
    "rtsp://admin:admin@123@10.245.106.240:554/ISAPI/Streaming/channels/101",
    "rtsp://admin:admin%40123@10.245.106.240:554/ISAPI/Streaming/channels/102",
    "rtsp://admin:admin@123@10.245.106.240:554/h264/ch1/main/av_stream",
    "rtsp://admin:admin@123@10.245.106.240:554/h264/ch1/sub/av_stream",
    "rtsp://admin:admin%40123@10.245.106.240:554/h264/ch1/main/av_stream",
    "rtsp://admin:admin%40123@10.245.106.240:554/h264/ch1/sub/av_stream"
]

# We can run python OpenCV or check ffmpeg command
for u in urls:
    print(f"\n======================================")
    print(f"Testing URL: {u}")
    import cv2
    import os
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(u, cv2.CAP_FFMPEG)
    if cap.isOpened():
        ret, frame = cap.read()
        print(f"cap.isOpened() = True, ret = {ret}")
        if ret and frame is not None:
            print(f"SUCCESS! Shape = {frame.shape}")
            cv2.imwrite("scratch/camera_success.jpg", frame)
            cap.release()
            break
        cap.release()
    else:
        print("cap.isOpened() = False")
