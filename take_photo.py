import cv2
import time;

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

if not ret:
    print("Failed to capture image.")
else:
    cv2.imshow('Captured Image', frame)
    cv2.waitKey(0)

    currtime = time.time()
    cv2.imwrite(f'captured_image_{currtime}.jpg', frame)

cap.release()
cv2.destroyAllWindows()
