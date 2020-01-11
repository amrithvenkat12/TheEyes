#!/bin/bash 
# To run this file: ./RunRos.sh
echo "Initializing ROS, rviz and running nodes..."
#DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )" #Gets the current script directory.
#source $DIR/catkin_ws/devel/setup.bash
source ~/.bashrc
source /home/$USER/catkin_ws/devel/setup.bash #source /home/monoxubu/catkin_ws/devel/setup.bash
bash -c "roslaunch camera mycamera.launch"
echo "Initialization completed."
#$SHELL
##IMPORTANT NOTES FOR FINE TUNNING AND TEST DAY
# 0. Adjust camera focus on the lanes. Play also with brightness, Sharpness, Hue, etc.
# 1. Lane lines have to be FLAT. Make sure the tape is flat, not twisted or else it won't be seen.
# 2. Adjust Perpesctve points according to paralel and centered lanes.
#	The distortion is greatly blurring the lane, especially those points more further away.
# 3. Adjust thresholding level acording to current lighthing settings.



# Source makes these commands work
# rosrun camera camera_main.py # rosrun = Individual node
# roslaunch camera mycamera.launch # roslaunch = Multiple Nodes

#rosrun opens the camera package and pass parameters: http://wiki.ros.org/rosbash#rosrun
#For uvc_camera package read documentation of available parameters in http://wiki.ros.org/uvc_camera
#Original
#gnome-terminal \
#--tab -e 'bash -c "roscore"' \
#--tab -e 'bash -c "rosrun uvc_camera uvc_camera_node _fps:10"' \
#--tab -e 'bash -c "rviz"'

#Pruebas
#gnome-terminal \
#--tab -e 'bash -c "roslaunch camera mycamera.launch"'
#--tab -e 'bash -c "roscore"' #Aparentemente innecesario.

## OTHER USEFUL COMMANDS
# rostopic echo /camera_info
# rostopic list

##CAMERA PARAMETERS
#http://ros-developer.com/2017/04/23/camera-calibration-with-ros/
#uvcdynctrl --device=/dev/video0 --clist
#uvcdynctrl --device=/dev/video0 --get='Brightness'
#uvcdynctrl --device=/dev/video0 --set='Brightness' "60"


##CAMERA CALIBRATION
#http://wiki.ros.org/camera_calibration
#https://wiki.ros.org/camera_calibration/Tutorials/MonocularCalibration
#http://wiki.ros.org/camera_calibration_parsers to convert bewteen file formats (eg: cal.yml to cal.ini)
#Tutorial on camera calibration 
#To run the cameracalibrator.py node for a monocular camera using an 8x6 chessboard with 108mm squares:
#size= chessboard size as NxM, counting interior corners, e.g. a standard chessboard is 7x7
#square = chessboard square size in meters
#last two parameters are node and topic name
#rosrun camera_calibration cameracalibrator.py --size=8x12 --square=0.035 --pattern='chessboard' image:=/image_raw camera:=/
#SAVE button: click it after a succesfull calibration, the data (calibration data and images used for calibration) will be written to /tmp/calibrationdata.tar.gz.
#Maybe put the resulting calibration file here /home/<username>/.ros/camera_info/head_camera.yaml
#COMMIT button: click it to download data to the camera and then check it with Terminal run:
#rosrun camera_calibration cameracheck.py --size 8x6 monocular:=/forearm image:=image_rect


#LAUNCHING BAG FILES
# https://answers.ros.org/question/62811/how-to-play-rosbag-using-launch-file/


