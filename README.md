First and beforehand, I want to thank Li Wang for uploading a Lane Detection algorithm and making it publicly available. He has also, referenced those codes he used as well. 
You can find his code here: https://github.com/liwangGT/CarND-Advanced-Lane-Lines
Also he has a YouTube video here with the running program: https://www.youtube.com/watch?v=X8QN-qY7uIo
In a brief, I took Li Wang's code, migrated it into ROS platform and adjusted/changed/added many things to make it work live on the ADAS car.

With that said, welcome to our Lane Detection Project. The main goal here is detect lanes in a racing track utilizing the ADAS car and its camera. 
You can find info about the ADAS car here: https://www.digitalwerk.net/adas-modellauto/

If you are in need of a similar project running on ROS or using Machine Vision, feel free to use this code. 
_____________________________
What Knowledge do you require before running this code?
- Watch the 38 minute video "Ross Kippenbrock - Finding Lane Lines for Self Driving Cars" on YouTube. Then, you will understand clearly what is this code about.
You can find the video here: https://www.youtube.com/watch?v=VyLihutdsPk
- It is advised that you understand what is ROS, making catkin packages and general machine vision knowledge.
- Watch the Working Videos to see the project in action.
_____________________________
What setup do you need?
 - Ubuntu 16.04
 - Python 2.7
 - Download only the camera folder. Convert it into a catkin package in src. It should look something like "/home/YOUR_USERNAME/catkin_ws/src/camera"
 - Note: the "/camera/src/bagfiles" folder is empty. I can't upload the bag file here in gitLab because the file size (336Mb) is larger than the 10Mb limit. Instead, you can download the file, name it "with_cover.bag" and put into the bagfiles folder. The bagfile is not mandatory though. https://nextcloud.th-deg.de/s/ETzpYfEtW4SHRwX
 - You can optionally run the algorithm on a live camera or a video file without the bag file. You must edit the camera_main.py file.

_____________________________
How to run the project?

1) Run this file "/camera/src/Run_Me.sh". This is a Bash file that will source the terminal and Start ROS with the camera launch file. It is possible that you need to edit file. Make sure it is an executable file and run it.

2) Running the previous file will automatically execute the launch file "/camera/launch/mycamera.launch". Put here all the nodes that you are going to be running (much better than running 1 node per terminal). I have commented some of them. The nodes are:
    - rviz: For visualizing the live camera or video output, with the code running.
    - rqt_bag: This are bag files recorded from the ADAS car. We use this recorded nodes to run them as simulations where we test the lane code.
    - uvc_camera: This node connects to an usb webcam if you have one.
    - camera: This node is our main python file camera_main.py where the code starts. I recommend you to leave this node commented, and instead start and run this Python file when needed from your Text editor of from the Terminal.

3) Having run the bash file will open two softwares:
    - The rqt_bag software: All the recorded nodes are here. Right click the "camera_image_undistorted" node and click on publish. The node will now appear on ROS. At anytime from now on, you can play and pause the recording. The data will be published to the ROS node as if it were live.
    - Rviz software: It will using the default config file "/camera/src/rviz_config.rviz". This might not work for you because it's meant to run on my laptop specifics. So you may need to create a new rviz config file.
If the configuration file is recognized, you should see a big white screen with the label "no image". This is good. (The "no image" text will be replaced with the output data once you run step #4)
If it didn't work, rviz will be open and you will see the default workspace windows. Make sure to add the Image topic of the "camera_image_undistorted" node (You can only see the node one it has been launched on step #4). 

    If this step was succesful, rviz should be ready to display the video output once you start the python file on the next step and rqt_bag should be publishing the data as a node visible by ROS commands.
  
4) Run the main Python file: "/camera/src/camera_main.py". You can do it using the terminal or using your favorite text editor program. I personally used Visual Studio Code and click on the  button "Run Python File in Terminal". Remember to play the node on "rqt_bag". If the Python file is succesfully running, you should see on rviz an endless 20s loop with the Lane Algorithm running on it.


Congratulations. You are ready to start modifying the code. Take anything you need!
_________________________
Motto: We See Everything.

Song: The Eye of The Tiger.

Project tested on Ubuntu 16.04 with Python 2.7.12 64-bit. 12/Jan/2020
