from ultralytics import YOLO
import cv2
import time
model = YOLO("best.pt")
print(model.names)
COLORS = {0: (255, 80, 0), 1: (0, 220, 255)}   # Yellow, Blue

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)



# Przed pętlą:
fps = 0
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model(frame, conf=0.4, iou=0.45, verbose=False)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        label  = model.names[cls_id]
        color  = COLORS.get(cls_id, (0, 255, 0))


        # Centrum pachołka — gotowe do sterowania
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        frame_cx = frame.shape[1] // 2
        error = cx - frame_cx   # ujemny = pachołek po lewej, dodatni = po prawej

        cv2.circle(frame, (cx, cy), 5, color, -1)
        cv2.line(frame, (frame_cx, 0), (frame_cx, frame.shape[0]), (255,255,255), 1)
        # Zastąp putText dla error i FPS tym blokiem:

        # Tło pod tekst (czytelność)
        cv2.rectangle(frame, (10, 10), (350, 70), (0, 0, 0), -1)   # czarne tło

        # Error w linii 1
        cv2.putText(frame, f"error: {error}px",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # FPS w linii 2
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, f"{label} {conf:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("YOLO Cone Detector", frame)
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()