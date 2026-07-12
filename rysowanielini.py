import cv2
import numpy as np
import time
from collections import deque
from ultralytics import YOLO

try:
    from scipy.interpolate import splprep, splev
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ──────────────────────────────────────────────
#  KONFIGURACJA
# ──────────────────────────────────────────────
MODEL_PATH   = "best.pt"
CONF         = 0.4
IOU          = 0.45
CAM_INDEX    = 0
CAM_W, CAM_H = 640, 480

# Indeksy klas — sprawdź przez print(model.names) i dopasuj
YELLOW_CLS   = 1   # yellow cone
BLUE_CLS     = 0   # blue cone
COLORS       = {YELLOW_CLS: (0, 215, 255), BLUE_CLS: (255, 80, 0)}

# Kalibracja odleglosci (zmierz empirycznie):
# Ustaw pacholek w odleglosci CALIB_DIST_M i odczytaj wysokosc boxa w px
CALIB_BBOX_H_PX = 280
CALIB_DIST_M    = 1.0
CONE_HEIGHT_M   = 0.325    # rzeczywista wysokosc pachołka FS [m]
FOCAL_PX        = CALIB_BBOX_H_PX * CALIB_DIST_M


# ──────────────────────────────────────────────
#  FUNKCJE POMOCNICZE
# ──────────────────────────────────────────────
def estimate_distance(bbox_h_px: int) -> float:
    if bbox_h_px < 5:
        return float("inf")
    return round((CONE_HEIGHT_M * FOCAL_PX) / bbox_h_px, 2)


def draw_spline(frame, points, color, thickness=2, n_pts=120):
    if len(points) < 2:
        return
    if len(points) == 2 or not SCIPY_AVAILABLE:
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], color, thickness, cv2.LINE_AA)
        return
    pts = np.array(points, dtype=np.float32)
    try:
        k = min(3, len(points) - 1)
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, k=k)
        t_new  = np.linspace(0, 1, n_pts)
        xs, ys = splev(t_new, tck)
        curve  = np.column_stack([xs, ys]).astype(np.int32)
        cv2.polylines(frame, [curve], False, color, thickness, cv2.LINE_AA)
    except Exception:
        pass


def draw_hud(frame, error_px: int, fps: float, n_yellow: int, n_blue: int):
    cv2.rectangle(frame, (8, 8), (320, 110), (0, 0, 0), -1)
    texts = [
        (f"FPS:    {fps:5.1f}",                   (0, 255, 0)),
        (f"Error:  {error_px:+5}px",               (255, 255, 255)),
        (f"Yellow: {n_yellow}  Blue: {n_blue}",    (180, 180, 180)),
    ]
    for i, (txt, col) in enumerate(texts):
        cv2.putText(frame, txt, (14, 34 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)

    # Wskaznik kierunku
    bar_cx = 164
    bar_y  = 95
    bar_w  = 140
    cv2.line(frame, (bar_cx - bar_w // 2, bar_y),
             (bar_cx + bar_w // 2, bar_y), (80, 80, 80), 6, cv2.LINE_AA)
    indicator_x = int(np.clip(bar_cx + error_px * bar_w / (CAM_W // 2),
                               bar_cx - bar_w // 2, bar_cx + bar_w // 2))
    col_dir = (0, 80, 255) if abs(error_px) > 60 else (0, 255, 80)
    cv2.circle(frame, (indicator_x, bar_y), 7, col_dir, -1)
    cv2.circle(frame, (bar_cx, bar_y), 4, (255, 255, 255), -1)


# ──────────────────────────────────────────────
#  INICJALIZACJA
# ──────────────────────────────────────────────
model = YOLO(MODEL_PATH)
print(f"Klasy modelu: {model.names}")

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)


# Sprawdź czy się przyjęło
print(f"FPS:  {cap.get(cv2.CAP_PROP_FPS)}")
print(f"W×H:  {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}×{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")
print(f"FOURCC ustawiony: {cap.get(cv2.CAP_PROP_FOURCC)}")

bbox_history = deque(maxlen=3)
prev_time    = time.time()
fps          = 0.0
error_px     = 0

print("Start. Nacisnij Q aby wyjsc.")

# ──────────────────────────────────────────────
#  PETLA GLOWNA
# ──────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Detekcja YOLO
    results = model(frame, conf=CONF, iou=IOU, verbose=False)
    boxes   = results[0].boxes

    yellow_pts, blue_pts = [], []
    all_detections = list(boxes) if boxes is not None else []

    # Stabilizacja: uzyj poprzednich detekcji gdy brak nowych
    if all_detections:
        bbox_history.append(all_detections)
    draw_boxes = bbox_history[-1] if bbox_history else []

    for box in draw_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        label  = model.names.get(cls_id, f"cls{cls_id}")
        color  = COLORS.get(cls_id, (0, 255, 0))
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        dist   = estimate_distance(y2 - y1)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        dist_str = f" {dist:.1f}m" if dist != float("inf") else ""
        cv2.putText(frame, f"{label} {conf:.2f}{dist_str}",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 4, color, -1)

        if cls_id == YELLOW_CLS:
            yellow_pts.append((cx, cy))
        elif cls_id == BLUE_CLS:
            blue_pts.append((cx, cy))

    # Linie toru — sortuj od dolu klatki (najblizszy pierwszy)
    yellow_pts.sort(key=lambda p: -p[1])
    blue_pts.sort(key=lambda p: -p[1])

    draw_spline(frame, yellow_pts, (0, 215, 255), thickness=2)
    draw_spline(frame, blue_pts,   (255, 80,   0), thickness=2)

    # Srodkowa linia toru
    midpoints = []
    n_pairs = min(len(yellow_pts), len(blue_pts))
    for i in range(n_pairs):
        mx = (yellow_pts[i][0] + blue_pts[i][0]) // 2
        my = (yellow_pts[i][1] + blue_pts[i][1]) // 2
        midpoints.append((mx, my))
        cv2.circle(frame, (mx, my), 5, (255, 255, 255), -1)

    draw_spline(frame, midpoints, (255, 255, 255), thickness=2)

    # Error sterowania
    frame_cx = CAM_W // 2
    cv2.line(frame, (frame_cx, 0), (frame_cx, CAM_H),
             (255, 255, 255), 1, cv2.LINE_AA)

    if midpoints:
        error_px = midpoints[-1][0] - frame_cx
    elif yellow_pts and not blue_pts:
        error_px = yellow_pts[-1][0] - frame_cx + 120
    elif blue_pts and not yellow_pts:
        error_px = blue_pts[-1][0] - frame_cx - 120
    else:
        error_px = 0

    # FPS i HUD
    curr_time = time.time()
    fps       = 1.0 / max(curr_time - prev_time, 1e-6)
    prev_time = curr_time

    draw_hud(frame, error_px, fps, len(yellow_pts), len(blue_pts))

    cv2.imshow("FS Cone Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()