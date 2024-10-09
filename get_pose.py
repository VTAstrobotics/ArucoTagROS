import cv2
import imutils
from time import sleep
from imutils.video import VideoStream
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
arucoParams = cv2.aruco.DetectorParameters()


vs = VideoStream(src=0).start()
sleep(2)

while True:
    