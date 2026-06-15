import cv2
from ultralytics import YOLO
import time

model = YOLO("yolov8s.pt")
video = cv2.VideoCapture("traffic.mp4")
prev_time = time.time()

while True:
    ret, frame = video.read()

    if not ret:
        break

    results = model(frame, stream=True)

    vehicle_count = 0

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            if cls in [2, 3, 5, 7]and float(box.conf[0])>0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                conf=float(box.conf[0])
                label = result.names[cls]+ " "+str(round(conf,2))
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                vehicle_count += 1

    density = min(int((vehicle_count / 20) * 100), 100)

    if vehicle_count > 15:
        status = "HEAVY TRAFFIC"
        status_color = (0, 0, 255)
        light_color = (0, 0, 255)
    elif vehicle_count > 8:
        status = "MODERATE TRAFFIC"
        status_color = (0, 165, 255)
        light_color = (0, 165, 255)
    else:
        status = "NORMAL TRAFFIC"
        status_color = (0, 255, 0)
        light_color = (0, 255, 0)

    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time))
    prev_time = curr_time

    cv2.rectangle(frame, (0, 0), (340, 130), (0, 0, 0), -1)
    cv2.putText(frame, "TRAFFIC CONGESTION DETECTION", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "Vehicles: " + str(vehicle_count), (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, "Density: " + str(density) + "%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, status, (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(frame, "FPS: " + str(fps), (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.circle(frame, (580, 50), 30, (50, 50, 50), -1)
    cv2.circle(frame, (580, 50), 30, light_color, -1)
    cv2.putText(frame, "TRAFFIC", (545, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "LIGHT", (553, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Traffic Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()