import cv2
import imutils
from time import sleep
from imutils.video import VideoStream
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
arucoParams = cv2.aruco.DetectorParameters()

arucoDetector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

vs = VideoStream(src=0).start()
sleep(2)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)


    corners, ids, rejected = arucoDetector.detectMarkers(frame)

    if ids is not None and len(corners) > 0:
        ids = ids.flatten()
        print(f"Aruco ID: {ids}")
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):
        break

cv2.destroyAllWindows()
vs.stop()