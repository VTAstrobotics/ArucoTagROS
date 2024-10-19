#took a lot of this from an online video, will link the video on Monday.com
import cv2
import os
import imutils
import numpy as np
from time import sleep
from imutils.video import VideoStream

arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
arucoParams = cv2.aruco.DetectorParameters()
arucoDetector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

root = r"C:\Users\jeron\OneDrive\Documents\GitHub\ArucoTagROS" #bad practice change later
paramPath = os.path.join(root, "matrixanddist.npz") # before using, you must run calibrate.py to obtain calibration matrix

if not os.path.exists(paramPath):
    print(".npz path does not exist")
else:
    data = np.load(paramPath)
    cameraMatrix = data['matrix']
    distCoeffs = data['distortion']

markerLength = 0.1  # our tags are roughly 10 cm in width

objectPoints = np.array([
    [-markerLength / 2, markerLength / 2, 0],  
    [markerLength / 2, markerLength / 2, 0],   
    [markerLength / 2, -markerLength / 2, 0], 
    [-markerLength / 2, -markerLength / 2, 0]  
], dtype=np.float32)

vs = VideoStream(src=0).start()
sleep(2)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)

    corners, ids, rejected = arucoDetector.detectMarkers(frame)

    if len(corners) > 0:
        for markerCorner, markerID in zip(corners, ids.flatten()): # converting to 1-D arrary
            imagePoints = markerCorner.reshape((4, 2)) 

            success, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs) # obtaining rotation vector and translation vector from solvePnP

            if success:
                cv2.aruco.drawDetectedMarkers(frame, corners)
                cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.1)  #can change axis length from 0.1 if desired

                # print(f"marker ID: {markerID}")
                # print(f"translation vector :\n {tvec}")
                # print(f"rotation vector :\n {rvec}")

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == ord('c'):
        break

cv2.destroyAllWindows()
vs.stop()

