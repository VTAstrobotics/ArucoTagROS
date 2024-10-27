import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String, Bool, Image, Pose
import cv2
import os
import imutils
import numpy as np
from imutils.video import VideoStream


class PosePublisher(Node):
    def __init__(self):
        super().__init__('pose_publisher')
        
        # Publishers and Subscribers
        self.pose_publisher = self.create_publisher(String, 'pose', 10)
        self.stop_subscription = self.create_subscription(
            Bool,
            'stop_aruco',
            self.stop_callback,
            10)
        self.image_subscription = self.create_subscription(
            Image,
            'camera/image',
            self.image_callback,
            10) #try image subscription with CvBridge
            
        self.bridge = CvBridge()
        self.stop = False

        self.arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
        self.arucoParams = cv2.aruco.DetectorParameters()
        self.arucoDetector = cv2.aruco.ArucoDetector(self.arucoDict, self.arucoParams)

        paramPath = os.path.join('.', "matrixanddist.npz")
        if not os.path.exists(paramPath):
            self.get_logger().error(".npz path does not exist")
            return
        data = np.load(paramPath)
        self.cameraMatrix = data['matrix']
        self.distCoeffs = data['distortion']
        
        self.markerLength = 0.1  # 10 cm tag size
        self.objectPoints = np.array([
            [-self.markerLength / 2, self.markerLength / 2, 0],
            [self.markerLength / 2, self.markerLength / 2, 0],
            [self.markerLength / 2, -self.markerLength / 2, 0],
            [-self.markerLength / 2, -self.markerLength / 2, 0]
        ], dtype=np.float32)
        #self.vs = VideoStream(src=0).start()

    def stop_callback(self, msg):
        self.stop = msg.data
        self.get_logger().info(f'Detection {"stopped" if self.stop else "resumed"}')

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgra8')

        if self.stop:
            return
        try:
            frame = imutils.resize(frame, width=600)
            corners, ids, _ = self.arucoDetector.detectMarkers(frame)

            if ids is not None and len(corners) > 0:
                for markerCorner, markerID in zip(corners, ids.flatten()):
                    imagePoints = markerCorner.reshape((4, 2))
                    success, rvec, tvec = cv2.solvePnP(
                        self.objectPoints,
                        imagePoints,
                        self.cameraMatrix,
                        self.distCoeffs
                    )
                    
                    if success:
                        pose_data = f"Marker ID: {markerID}, Translation: {tvec.flatten()}, Rotation: {rvec.flatten()}"
                        self.pose_publisher.publish(String(data=pose_data))
                        self.get_logger().info(f"Published pose: {pose_data}")
                    
        except Exception as e:
            self.get_logger().error(f'Error processing frame: {str(e)}')

    def destroy_node(self):
        self.vs.stop()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    pose_publisher = PosePublisher()

    rclpy.spin(pose_publisher)

    pose_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
