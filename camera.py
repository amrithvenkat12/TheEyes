#!/usr/bin/env python

#This code is partly based on an example found at:
#https://github.com/isarlab-department-engineering/ros_dt_lane_follower/blob/master/src/lane_detection.py

import rospy # ROS client library for Python
import numpy as np #NumPY is a package for scientific computing with Python.
import cv2 ##OpenCV - Written natively in C++, supported by C++, Python, Java and Matlab
import math #Math functions accoring to C standard.
from cv_bridge import CvBridge #Bridges between ROS images and OpenCV images
from sensor_msgs.msg import Image

def lane_detect():
	rospy.init_node('HIL_Control',anonymous=True)
	rospy.Subscriber("/image_raw",Image,monox_callback,queue_size=1,buff_size=2**24)
	try:
		rospy.loginfo("Pruebas 1")
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")


#callback is executed once for each frame
def monox_callback(data):
    #make OpenCV able to process image
    img = bridge.imgmsg_to_cv2(data)
    #Median Filter against noise
    blurred = cv2.medianBlur(img, 5)
    rgb = cv2.cvtColor(blurred,cv2.COLOR_BGR2RGB)
    #Concatenating image to compare
    image = np.concatenate((rgb, blurred), axis=1) # 0 for vertical, 1 for horizontal

    #publish processed image in ROS Topic
    image = bridge.cv2_to_imgmsg(image)
    pub_image.publish(image)


#definitions and declarations
bridge = CvBridge()
pub_image = rospy.Publisher('/hello_world_image_2',Image,queue_size=1)

if __name__ == '__main__':
    try:
        lane_detect()
    except KeyboardInterrupt:
		print("Shutting down")
    #except rospy.ROSInterruptException:
     #   pass





