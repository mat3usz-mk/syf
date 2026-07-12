import cv2

# Zmieniamy CAP_DSHOW na CAP_MSMF
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

# Ustawiamy parametry, które potwierdziliśmy w PotPlayer
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

if not cap.isOpened():
    print("Błąd: Kamera nadal zablokowana. Upewnij się, że PotPlayer jest zamknięty!")
else:
    print("Sukces! Widzisz obraz?")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Arducam OV9782', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()