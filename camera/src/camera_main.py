#!/usr/bin/env python
#This code is partly based on an example found at:
#https://github.com/liwangGT/CarND-Advanced-Lane-Lines
#https://github.com/isarlab-department-engineering/ros_dt_lane_follower/blob/master/src/lane_detection.py

#Definitions and declarations
import rospy # ROS client library for Python
import numpy as np # NumPY is a package for scientific computing with Python.
import cv2 ## OpenCV - Written natively in C++, supported by C++, Python, Java and Matlab
import math # Math functions accoring to C standard.
import os # For Terminal stuff
import pyzbar.pyzbar as pyzbar ## For QR Codes
from cv_bridge import CvBridge # Bridges between ROS images and OpenCV images
from sensor_msgs.msg import Image
#from sensor_msgs.msg import CameraInfo
import pyzbar.pyzbar as pyzbar
#from warp_transformer import thresholding
from line import Line
from load_parameters import load_perspective_transform_from_pickle as load_M_Minv
import time
from scipy import stats
import warnings
warnings.simplefilter('ignore', np.RankWarning) ## I need to fix the 4 default polyfit leftx, lefty, rightx, righty

def Pylon_Camera_tests():
    rospy.init_node('node_pruebas',anonymous=True)
    rospy.loginfo("Running Pylon_Camera_tests")
    rospy.Subscriber("/camera_image_undistorted",Image,TakingSnapshots_frame,queue_size=1,buff_size=2**24)
    rospy.spin()


def Hello_World():
    rospy.init_node('node_pruebas',anonymous=True)
    rospy.loginfo("Running Hello_World")
    rospy.Subscriber("/image_raw",Image,Hello_World_frame,queue_size=1,buff_size=2**24)
    rospy.spin() # keeps python from exiting until this node is stopped
    #ParanNames = rospy.get_param_names()
    #print('\n'.join(ParanNames)) #Converts a list variable into a str variable and print it
    #rosservice list (TERMINAL)
    #rostopic list (TERMINAL)

def lane_detection():
    rospy.init_node('node_LaneDetection',anonymous=True)
    rospy.loginfo("Running lane_detection on /image_raw Node")
    rospy.Subscriber("/image_raw",Image,lane_detection_frame,queue_size=1,buff_size=2**24)
    rospy.spin() # keeps python from exiting until this node is stopped

def lane_detection2():
    rospy.init_node('node_LaneDetection',anonymous=True)
    rospy.loginfo("Running lane_detection on /camera_image_undistorted Node")
    rospy.Subscriber("/camera_image_undistorted",Image,lane_detection_frame,queue_size=1,buff_size=2**24)
    rospy.spin() # keeps python from exiting until this node is stopped

def QR_code_detection():
    rospy.init_node('node_QR_code_detection',anonymous=True)
    rospy.loginfo("Running QR_code_detection")
    rospy.Subscriber("/image_raw",Image,QR_code_detection_frame,queue_size=1,buff_size=2**24)
    rospy.spin() # keeps python from exiting until this node is stopped

def TakingSnapshots():
    rospy.init_node('node_TakingSnapshots',anonymous=True)
    rospy.loginfo("Running TakingSnapshots")
    rospy.Subscriber("/image_raw",Image,TakingSnapshots_frame,queue_size=1,buff_size=2**24)
    rospy.spin() # keeps python from exiting until this node is stopped
    

#This is executed once for each frame
def lane_detection_frame(data, i=[0]):
    start = time.time()
    #os.system('clear') #Clear the console.
    i[0]+=1
        
    import glob
    # Make a list of images
    if mode == 1: #1 Original
        images = glob.glob(FolderPath +'/test_images/test*.jpg')
        images.sort()
        img = cv2.imread(images[4])
        img01 = img.copy()
        img10 = img.copy()
        putText2(img01, "Original RGB")

        #Color Conversion
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Today's images use RGB but OpenCV uses BGR format.
        img02 = img.copy()
        putText2(img02, "OpenCV BGR")
    elif mode == 2: #2 webcam test pixture
        images = glob.glob(FolderPath +'/test_images/ADAS_straight*.jpeg')
        images.sort()
        img = cv2.imread(images[0])
        img01 = img.copy()
        img10 = img.copy()
        putText2(img01, "Test 1 RGB")

        #Color Conversion
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Today's images use RGB but OpenCV uses BGR format.
        img02 = img.copy()
        putText2(img02, "OpenCV BGR")
    elif mode == 3: #3 ADAS car test picture
        images = glob.glob(FolderPath +'/test_images/ADAS_straight*.jpeg')
        images.sort()
        img = cv2.imread(images[1])
        img02 = img.copy()
        putText2(img02, "OpenCV BGR")

        #Color Conversion
        img01 = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB) # Today's images use RGB but OpenCV uses BGR format.
        img10 = img01.copy()
        putText2(img01, "Test 2 RGB")
    elif mode == 4: #ADAS Bag File, Simulated Node.
        img = bridge.imgmsg_to_cv2(data) #Although the image comes in BGR already, bridge is inverting it back to RGB
        img01 = img.copy()
        img10 = img.copy()
        putText2(img01, "Test 2 RGB")

        #Color Conversion
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img02 = img.copy()
        putText2(img02, "OpenCV BGR")
    else: #webcam (It has to match the mode in perspective_transform)
        img = bridge.imgmsg_to_cv2(data)
        img = cv2.flip(img, 1) #Flip it through vertical axis 
        img01 = img.copy()
        img01 = cv2.cvtColor(img01, cv2.COLOR_BGR2RGB) # imgmsg_to_cv2 converts data to OpenCV's BGR format
        img10 = img.copy()
        putText2(img01, "Camera RGB")
        
        img02 = img.copy()
        putText2(img02, "OpenCV BGR")
        
    '''##print(images)
    if i[0] == 6:
        i[0] = 0
    #img = cv2.imread(images[int(i[0])])#Funciona!!  
    #'''

    # Area to Warp - These values come from perspective_transform.py
    cv2.circle(img10, (pts[0,0], pts[0,1]), 10, (255,0,0), thickness=5, lineType=8, shift=0) 
    cv2.circle(img10, (pts[1,0], pts[1,1]), 10, (255,0,0), thickness=5, lineType=8, shift=0) 
    cv2.circle(img10, (pts[2,0], pts[2,1]), 10, (255,0,0), thickness=5, lineType=8, shift=0) 
    cv2.circle(img10, (pts[3,0], pts[3,1]), 10, (255,0,0), thickness=5, lineType=8, shift=0)
    putText2(img10, "Region of Interest - ROI")
    
    # Warp the image
    #binary_warped = cv2.warpPerspective(binary_threshold, M, img_size, flags=cv2.INTER_LINEAR)
    img_size = (img.shape[1], img.shape[0])
    warped_img = cv2.warpPerspective(img, M, img_size)
    img13 = warped_img.copy()
    #img13 = cv2.cvtColor(img13,cv2.COLOR_GRAY2RGB)
    putText2(img13, "Warped & BGR")

    '''
    # Filter White. To make it easier for more accurate Canny detection
    threshold = 200 # Original 200, working fine 160
    high_threshold = np.array([255, 255, 255]) #Bright white
    low_threshold = np.array([threshold, threshold, threshold]) #Soft White
    mask = cv2.inRange(warped_img, low_threshold, high_threshold)
    white_img = cv2.bitwise_and(warped_img, warped_img, mask=mask)
    img03 = white_img.copy() # Yellow image
    putText2(img03, "White Lanes Filter")


    # Filter Yellow
    hsv_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV) #Changing Color-space, HSV is better for object detection
    img09 = hsv_img.copy() #Filtered image
    putText2(img09, "HSV")
    #For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]. 
    high_threshold = np.array([110,255,255]) #Bright Yellow
    low_threshold = np.array([50,50,50]) #Soft Yellow   
    mask = cv2.inRange(hsv_img, low_threshold, high_threshold)
    yellow_img = cv2.bitwise_and(warped_img, warped_img, mask=mask)
    img04 = yellow_img.copy() # Yellow image
    putText2(img04, "Yellow Lanes Filter")
    
    # Combine the two above images
    filtered_img = cv2.addWeighted(white_img, 1., yellow_img, 1., 0.)
    img05 = filtered_img.copy() #Filtered image
    putText2(img05, "Combined Image")
    gray = cv2.cvtColor(filtered_img, cv2.COLOR_RGB2GRAY)
    '''

    # Convert image to gray scale
    gray = cv2.cvtColor(warped_img, cv2.COLOR_RGB2GRAY)
    img06 = gray.copy()
    img06 = cv2.cvtColor(img06,cv2.COLOR_GRAY2RGB)
    putText2(img06, "Gray Image")
    
    # Create binary based on detected pixels
    thresholdingLevel = 180 # 195 seems to yield best results between 180, 195 and 210
    _, binary_threshold = cv2.threshold(gray.copy(),thresholdingLevel,1,cv2.THRESH_BINARY) #1st parameter is the thresholding value used.
    img07 = binary_threshold*255
    img07 = cv2.cvtColor(img07,cv2.COLOR_GRAY2RGB)
    putText2(img07, "Threshold = " + str(thresholdingLevel))

    thresholdingLevel = 195
    _, binary_threshold2 = cv2.threshold(gray.copy(),thresholdingLevel,1,cv2.THRESH_BINARY)
    img17 = binary_threshold2*255
    img17 = cv2.cvtColor(img17,cv2.COLOR_GRAY2RGB)
    putText2(img17, "Threshold = " + str(thresholdingLevel))

    thresholdingLevel = 210
    _, binary_threshold3 = cv2.threshold(gray.copy(),thresholdingLevel,1,cv2.THRESH_BINARY)
    img18 = binary_threshold3*255
    img18 = cv2.cvtColor(img18,cv2.COLOR_GRAY2RGB)
    putText2(img18, "Threshold = " + str(thresholdingLevel))

    neighbourhoodArea1 = 9
    neighbourhoodArea2 = 11
    neighbourhoodArea3 = 13
    neighbourhoodArea4 = 15
    RandomConstant1 = 13
    RandomConstant2 = 15
    RandomConstant3 = 17
    RandomConstant4 = 19

    binary_threshold4 = cv2.adaptiveThreshold(gray.copy(),1,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,neighbourhoodArea1, RandomConstant1)
    img08 = binary_threshold4*255
    img08 = cv2.cvtColor(img08,cv2.COLOR_GRAY2RGB)
    putText2(img08, "n=" + str(neighbourhoodArea1) + " C=" + str(RandomConstant1))

    binary_threshold5 = cv2.adaptiveThreshold(gray.copy(),1,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,neighbourhoodArea2, RandomConstant2)
    img14 = binary_threshold5*255
    img14 = cv2.cvtColor(img14,cv2.COLOR_GRAY2RGB)
    putText2(img14, "n=" + str(neighbourhoodArea2) + " C=" + str(RandomConstant2))

    binary_threshold6 = cv2.adaptiveThreshold(gray.copy(),1,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,neighbourhoodArea3, RandomConstant3)
    img15 = binary_threshold6*255
    img15 = cv2.cvtColor(img15,cv2.COLOR_GRAY2RGB)
    putText2(img15, "n=" + str(neighbourhoodArea3) + " C=" + str(RandomConstant3))

    binary_threshold7 = cv2.adaptiveThreshold(gray.copy(),1,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,neighbourhoodArea4, RandomConstant4)
    img16 = binary_threshold7*255
    img16 = cv2.cvtColor(img16,cv2.COLOR_GRAY2RGB)
    putText2(img16, "n=" + str(neighbourhoodArea4) + " C=" + str(RandomConstant4))

    #Soution 1
    left_fit, right_fit, left_curverad, right_curverad, img11 = sliding_window(binary_threshold)
    img12 = draw_lane(img, binary_threshold, left_fit, right_fit, left_curverad, right_curverad)
    img12 = cv2.cvtColor(img12, cv2.COLOR_BGR2RGB) # Return from OpenCv's BGR to normal display in RGB. 

    #Solution 2
    left_fit, right_fit, left_curverad, right_curverad, img19 = sliding_window(binary_threshold2)
    img20 = draw_lane(img, binary_threshold2, left_fit, right_fit, left_curverad, right_curverad)
    img20 = cv2.cvtColor(img20, cv2.COLOR_BGR2RGB) # Return from OpenCv's BGR to normal display in RGB. 

    #Solution 2
    left_fit, right_fit, left_curverad, right_curverad, img21 = sliding_window(binary_threshold3)
    img22 = draw_lane(img, binary_threshold3, left_fit, right_fit, left_curverad, right_curverad)
    img22 = cv2.cvtColor(img22, cv2.COLOR_BGR2RGB) # Return from OpenCv's BGR to normal display in RGB. 

    ## List of images
    #img01 = Original RGB
    #img02 = OpenCV BGR
    #img03 = White Lanes Filter
    #img04 = Yellow Lanes Filter
    #img05 = Combined Image
    #img06 = Gray Image
    #img07 = Threshold 1
    #img08 = Adaptative Threshold 1
    #img09 = HSV Color Space
    #img10 = Area of Interest
    #img11 = Sliding Boxes 1
    #img12 = Returned to Original 1
    #img13 = Warped and BGR
    #img14 = Adaptative Threshold 2
    #img15 = Adaptative Threshold 3
    #img16 = Adaptative Threshold 4
    #img17 = Threshold 2
    #img18 = Threshold 3
    #img19 = Sliding Boxes 2
    #img20 = Returned to Original 2
    #img21 = Sliding Boxes 3
    #img22 = Returned to Original 3
    
    
    ##Screen Selector
    '''
    #For comparing 3 threshold with 3 results at the same time
    concatenation_1 = np.concatenate((img12, img20, img22), axis=1)
    concatenation_2 = np.concatenate((img07, img17, img18) , axis=1)
    concatenation_3 = np.concatenate((img11, img19, img21), axis=1)
    concatenation_3 = np.concatenate((concatenation_1, concatenation_2, concatenation_3), axis=0)
    #'''
    #'''
    #Current Screens
    concatenation_1 = np.concatenate((img12, img20, img22), axis=1)
    concatenation_2 = np.concatenate((img07, img17, img18) , axis=1)
    concatenation_3 = np.concatenate((img11, img19, img21), axis=1)
    concatenation_3 = np.concatenate((concatenation_1, concatenation_2, concatenation_3), axis=0)
    #'''

    #cv2.imwrite(FolderPath + '/snapshots/current_screen.jpeg', concatenation_3) #Printing a screenshot for the report. 
    
    '''
    #For Video
    imageName = FolderPath + '/snapshots/video_' + str(i[0]).zfill(3) + '.jpeg'
    cv2.imwrite(imageName, concatenation_3)
    #'''
    ROS_Frame = bridge.cv2_to_imgmsg(concatenation_3) #Convert cv2 image to ROS image
    ## Stats Calculation before publishing image
    end = time.time()
    timeElapsed = end - start
    MaxFPS = round(1/timeElapsed,1)
    InputSpeed = 1 #km/h
    InputSpeed = InputSpeed*3600/1000 #m/s
    BlindDriving = round(InputSpeed*timeElapsed*100,2)
    print("\nTime elapsed:" + str(round(timeElapsed*1000,1)) + " ms." + "\n" + "FPS: " + str(MaxFPS) + "\n" + "Blind Driving: " + str(BlindDriving) + " cm.")
    #publish processed image in ROS Topic
    pub_image.publish(ROS_Frame)

def putText2(img, text):
    origin = (0, 50)
    fontStyle = cv2.FONT_HERSHEY_COMPLEX #PLAIN COMPLEX
    fontSize = 1.5
    fontColor = (0, 0, 255) # OpenCV uses BGR format. Yellow = Green + Red
    fontThickness = 3
    cv2.putText(img, text, origin, fontStyle, fontSize, fontColor, fontThickness)
    return img

#This is executed once for each frame
def QR_code_detection_frame(data):
    cv2_Frame = bridge.imgmsg_to_cv2(data)
    currentFrame = cv2.cvtColor(cv2_Frame, cv2.COLOR_BGR2GRAY)
    decodedObjects = pyzbar.decode(currentFrame)
    
    for obj in decodedObjects:
        rospy.loginfo("QR Code Read: " + obj.data)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(currentFrame, str(obj.data), (50, 50), font, 2,(255, 0, 0), 3)

    ROS_Frame = bridge.cv2_to_imgmsg(currentFrame)
    pub_image.publish(ROS_Frame) #publish processed image in ROS Topic

    #This is executed once for each frame
def Hello_World_frame(data):
    cv2_Frame = bridge.imgmsg_to_cv2(data)
    currentFrame = cv2.cvtColor(cv2_Frame, cv2.COLOR_BGR2RGB)
    currentFrame = cv2.flip(currentFrame, 1)
    ROS_Frame = bridge.cv2_to_imgmsg(currentFrame)
    pub_image.publish(ROS_Frame) #publish processed image in ROS Topic


#This is executed once for each frame
def TakingSnapshots_frame(data, i=[0]):
    i[0]+=1 
    cv2_Frame = bridge.imgmsg_to_cv2(data)
    cv2_Frame = cv2.cvtColor(cv2_Frame, cv2.COLOR_BGR2RGB)

    if i[0] % 5 == 0:
        imageName = FolderPath + '/snapshots/snapshot_' + str(i[0]) + '.jpeg'
        cv2.imwrite(imageName, cv2_Frame)
        rospy.loginfo('Writing image ' + imageName)
    ROS_Frame = bridge.cv2_to_imgmsg(cv2_Frame) #Convert cv2 image to ROS image
    pub_image.publish(ROS_Frame) #publish processed image in ROS Topic

def sliding_window(binary_warped):
    out_img = (np.dstack((binary_warped, binary_warped, binary_warped)) * 255).astype(np.uint8)
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]*0.60):,:], axis=0) # orginal /2 # From 75% to 100% of height.

    #print "Shape " + str(np.shape(binary_warped)) # Resolution is 480 rows (height) x 640 columns (width)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - windowMargin
        win_xleft_high = leftx_current + windowMargin
        win_xright_low = rightx_current - windowMargin
        win_xright_high = rightx_current + windowMargin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low,win_y_low), (win_xleft_high,win_y_high), color=(0,255,0), thickness=2) # Green
        cv2.rectangle(out_img, (win_xright_low,win_y_low), (win_xright_high,win_y_high), color=(0,255,0), thickness=2) # Green
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            '''yValues = np.arange(len(good_left_inds))
            slope_0, _, _, _, _ = stats.linregress(good_left_inds, yValues)
            leftx_found = np.int(np.mean(nonzerox[good_left_inds]))
            if slope_0 > 0:
                leftx_current = leftx_found + int(10 + 10*slope_0)
            else:
                leftx_current = leftx_found - int(10 + 10*slope_0)
            '''
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            '''
            yValues = np.arange(len(good_right_inds))
            slope_1, _, _, _, _ = stats.linregress(good_right_inds, yValues)
            rightx_found = np.int(np.mean(nonzerox[good_right_inds]))
            if slope_1 < 0:
                rightx_current = rightx_found + int(10 + 10*slope_1)
            else:
                rightx_current = rightx_found - int(10 + 10*slope_1)
            '''
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]  

    #Resolution is 480 (height) x 640 (width)
    tx1Shape = leftx.shape
    if tx1Shape[0] == 0:
        leftx = [0,1,2,3] 

    ty1Shape = lefty.shape
    if ty1Shape[0] == 0:
        lefty = [1,1,479,479]
    
    tx2Shape = rightx.shape
    if tx2Shape[0] == 0:
        rightx = [636,637,638,639]

    ty2Shape = righty.shape
    if ty2Shape[0] == 0:
        righty = [1,1,479,479]
    

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2) # RankWarning: Polyfit may be poorly conditioned
    right_fit = np.polyfit(righty, rightx, 2) # RankWarning: Polyfit may be poorly conditioned

    # Stash away polynomials
    left_line.current_fit = left_fit
    right_line.current_fit = right_fit
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    left_fitx = np.clip(left_fitx, 0, 639) #Limiting the fits so that they don't exceed the maximum 640 height.
    right_fitx = np.clip(right_fitx, 0, 639) #Limiting the fits so that they don't exceed the maximum 640 height.
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0] # Left Lane = Blue
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255] # Right Lane = Red
    out_img[ploty.astype('int'),left_fitx.astype('int')] = [0, 255, 255] #G+R = Yellow Line
    out_img[ploty.astype('int'),right_fitx.astype('int')] = [0, 255, 255] #G+R = Yellow Line

    # Fit new polynomials to x,y in world space
    y_eval = np.max(ploty)  # Where radius of curvature is measured
    try:
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, deg=2)
        # Calculate radii of curvature in meters
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    except:
        left_curverad = 0

    try:
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, deg=2)
        # Calculate radii of curvature in meters
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    except:
        right_curverad = 0

    
    # Stash away the curvatures  
    left_line.radius_of_curvature = left_curverad  
    right_line.radius_of_curvature = right_curverad

    
    return left_fit, right_fit, left_curverad, right_curverad, out_img

def draw_lane(undistorted, binary_warped, left_fit, right_fit, left_curverad, right_curverad):
    # Create an image to draw the lines on
    warped_zero = np.zeros_like(binary_warped)
    color_warped = np.dstack((warped_zero, warped_zero, warped_zero))    
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]   

    #print "pruebas2" + str(np.shape(left_fitx)) + "  " + str(np.shape(right_fitx))

    # ?
    midpoint = np.int(undistorted.shape[1]/2)
    middle_of_lane = (right_fitx[-1] - left_fitx[-1]) / 2.0 + left_fitx[-1]
    offset = (midpoint - middle_of_lane) * xm_per_pix
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    
    if (left_curverad == 0) or (right_curverad == 0):
        cv2.fillPoly(color_warped, np.int_([pts]), (255,0,0) )
    else:
        cv2.fillPoly(color_warped, np.int_([pts]), (0,255,0) )

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    img_size = (undistorted.shape[1], undistorted.shape[0])
    unwarped = cv2.warpPerspective(color_warped, Minv, img_size, flags=cv2.INTER_LINEAR)
    
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted, 1, unwarped, 0.3, 0)
    
    # Add radius and offset calculations to top of video
    cv2.putText(result,"L. Lane Radius: " + "{:0.2f}".format(left_curverad/1000) + 'km', org=(50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1, color=(255,255,255), lineType = cv2.LINE_AA, thickness=2)
    cv2.putText(result,"R. Lane Radius: " + "{:0.2f}".format(right_curverad/1000) + 'km', org=(50,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1, color=(255,255,255), lineType = cv2.LINE_AA, thickness=2)
    cv2.putText(result,"C. Position: " + "{:0.2f}".format(offset) + 'm', org=(50,150), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1, color=(255,255,255), lineType = cv2.LINE_AA, thickness=2)
    return result


## Main code begins here:
if __name__ == '__main__':
    ## Define Gloabl Variables
    os.system('clear') #Clear the console.
    bridge = CvBridge()
    pub_image = rospy.Publisher('/image_processed', Image, queue_size=1)
    left_line = Line()
    right_line = Line() 
    M, Minv, pts, mode = load_M_Minv()
    FolderPath = os.path.dirname(__file__)
    font = cv2.FONT_HERSHEY_PLAIN  
    
    ## Important constants associated to Lane Detection
    windowMargin = 28 # Works with 25. Set the width of the windows +/- windowMargin
    minpix = 20 # Original = 50. Set minimum number of pixels found to recenter window
    nwindows = 25  # Original = 9. Choose the number of sliding windows
    xm_per_pix = 3.7/700.0 # meters per pixel in x dimension
    ym_per_pix = 30.0/720.0 # meters per pixel in y dimension
    LaneAreaColor = (0, 255, 0)


    try:
        #lane_detection() # 1. Code for detecting lane.
        lane_detection2() #1B. Same as 1 but on the simulated camera_image_undistorted node.
        #QR_code_detection() # 2. Code for detecting QR codes.
        #TakingSnapshots() # 3. Code for taking snapshoots.
        #Hello_World() #4. Experiments
        #Pylon_Camera_tests() #5. Taking snapshots from bagfile.
        rospy.loginfo("Shutting down: " + os.path.basename(__file__))
    except rospy.ROSInterruptException:
        pass
  

