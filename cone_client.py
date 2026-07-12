import cv2, json, os, time
from collections import deque

FRAME_PATH  = r"C:\Temp\yolox_frame.jpg"
RESULT_PATH = r"C:\Temp\yolox_result.json"
READY_FLAG  = r"C:\Temp\yolox_ready.flag"
TMP_PATH    = r"C:\Temp\yolox_frame_tmp.jpg"   # ← tymczasowy plik .jpg
COLORS = {"ConeYellow": (0, 220, 255), "ConeBlue": (255, 80, 0)}

os.makedirs(r"C:\Temp", exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

bboxes, scores, labels = [], [], []
bbox_history = deque(maxlen=3)   # stabilizacja — pamięta ostatnie 3 detekcje
last_send = 0
Y_OFFSET = 0   # offset jeśli przycinasz klatkę

print("Klient startuje. Q = wyjście")
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    now = time.time()
    matlab_busy = os.path.exists(READY_FLAG) or os.path.exists(RESULT_PATH)

    if now - last_send > 0.15 and not matlab_busy:
        try:
            h_frame = frame.shape[0]
            Y_OFFSET = int(h_frame * 0.3)
            frame_to_send = frame[Y_OFFSET:, :]   # dolne 70% — gdzie są pachołki

            cv2.imwrite(TMP_PATH, frame_to_send, [cv2.IMWRITE_JPEG_QUALITY, 95])
            time.sleep(0.02)   # poczekaj na zamknięcie pliku

            if os.path.exists(FRAME_PATH):
                os.remove(FRAME_PATH)
            os.rename(TMP_PATH, FRAME_PATH)

            open(READY_FLAG, 'w').close()
            last_send = now
        except (PermissionError, OSError):
            pass

    # Odbierz wyniki
    if os.path.exists(RESULT_PATH):
        try:
            with open(RESULT_PATH, 'r') as f:
                result = json.load(f)
            new_bboxes = result.get("bboxes", [])
            new_scores = result.get("scores", [])
            new_labels = result.get("labels", [])
            if new_bboxes:   # zapisz tylko gdy coś wykryto
                bbox_history.append((new_bboxes, new_scores, new_labels))
                bboxes, scores, labels = new_bboxes, new_scores, new_labels
            os.remove(RESULT_PATH)
        except (json.JSONDecodeError, PermissionError, OSError):
            pass

    # Rysuj detekcje (z Y_OFFSET żeby boxy były na właściwym miejscu)
    for i, bbox in enumerate(bboxes):
        x, y, w, h = [int(v) for v in bbox]
        y += Y_OFFSET   # ← przywróć oryginalną pozycję w pełnym kadrze
        label = labels[i] if i < len(labels) else "Cone"
        score = float(scores[i]) if i < len(scores) else 0.0
        color = COLORS.get(label, (0, 255, 0))
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        cv2.putText(frame, f"{label} {score:.2f}",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("YOLOX Cone Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()