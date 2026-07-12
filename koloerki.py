import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([18, 80, 80])
    upper = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Krok 1: Usuń szum (małe kropki)
    kernel_open = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    # Krok 2: Rozciągnij maskę PIONOWO — wypełnia czarny pas między żółtymi częściami
    # Wysoki kernel (np. 60px wysokości) "przeskakuje" czarny pas
    kernel_dilate = np.ones((60, 5), np.uint8)  # (wysokość, szerokość)
    mask_filled = cv2.dilate(mask, kernel_dilate, iterations=1)

    # Krok 3: Ponownie zamknij dziury po dilacji
    kernel_close = np.ones((5, 5), np.uint8)
    mask_filled = cv2.morphologyEx(mask_filled, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 3000:  # Zwiększony próg — po dilacji kontury są większe
            x, y, w, h = cv2.boundingRect(cnt)
            if h > w:
                # Oblicz środek X pachołka
                cx = x + w // 2
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(
                    frame, f"Pacholek ({area:.0f}px)",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                # Linia środkowa — przydatna do centrowania pojazdu
                cv2.line(frame, (cx, y), (cx, y + h), (0, 0, 255), 2)

    result = cv2.bitwise_and(frame, frame, mask=mask)        # oryginalna maska
    result_filled = cv2.bitwise_and(frame, frame, mask=mask_filled)  # po dilacji

    cv2.imshow("Oryginal", frame)
    cv2.imshow("Maska oryginalna", mask)
    cv2.imshow("Maska po wypelnieniu", mask_filled)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()