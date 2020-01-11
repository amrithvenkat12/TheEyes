import pickle
import cv2
import numpy as np
from os.path import join, basename
import os 

# Program local libraries
#from load_parameters import load_camera_mtx_dist_from_pickle as load_mtx_dist


FolderPath = os.path.dirname(__file__)

mode = 4 #1 Original, 2 Webcam, 3 ADAS Car test Picture, 4 ADAS Car Simulation Node.

# Where are the road test images?
road_test_images_dir = FolderPath + '/test_images' 

# Point to a straight road image here

if mode == 1:
    #Original
    road_straight_image_filename = 'straight_lines2.jpg'
elif mode == 2:
    #Webcam Picture
    road_straight_image_filename = 'ADAS_straight0.jpeg'
else:
    #ADAS Car
    road_straight_image_filename = 'ADAS_straight1.jpeg'
    #road_straight_image_filename = 'ADAS_straight2.jpeg'
    
    

# Where you want to save warped straight image for check?
road_straight_warped_image_dir = FolderPath + '/output_images/bird_eye_test'

# Where you want to save the transformation matrices (M,Minv)?
M_Minv_output_dir = FolderPath + '/output_images/camera_cal'

# Play with trapezoid ratio until you get the proper bird's eye lane lines projection
# bottom_width = percentage of image width
# top_width = percentage of image width
# height = percentage of image height
# car_hood = number of pixels to be cropped from bottom meant to get rid of car's hood
bottom_width=0.4
top_width=0.092
height=0.4
car_hood=45


# Sort coordinate points clock-wise, starting from top-left
# Inspired by the following discussion:
# http://stackoverflow.com/questions/1709283/how-can-i-sort-a-coordinate-list-for-a-rectangle-counterclockwise
def order_points(pts):
    # Normalises the input into the [0, 2pi] space, added 0.5*pi to initiate from top left
    # In this space, it will be naturally sorted "counter-clockwise", so we inverse order in the return
    mx = np.sum(pts.T[0]/len(pts))
    my = np.sum(pts.T[1]/len(pts))

    l = []
    for i in range(len(pts)):
        l.append(  (np.math.atan2(pts.T[0][i] - mx, pts.T[1][i] - my) + 2 * np.pi + 0.5 * np.pi) % (2*np.pi)  )
    sort_idx = np.argsort(l)
    
    return pts[sort_idx[::-1]]


def get_transform_matrices(pts, img_size):
    # Obtain a consistent order of the points and unpack them individually
    src = order_points(pts)
    
    #Give user some data to check
    print('Here are the ordered src pts:', src)
    
    # Destination points
    dst = np.float32([[src[3][0], 0],
                      [src[2][0], 0],
                      [src[2][0], img_size[1]],
                      [src[3][0], img_size[1]]])
    
    #Give user some data to check
    print('Here are the dst pts:', dst)
    
    # Compute the perspective transform matrix and the inverse of it
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv

def get_transform_matrices_2(source_pts, destination_pts):

    # Compute the perspective transform matrix and the inverse of it
    M = cv2.getPerspectiveTransform(source_pts, destination_pts)
    Minv = cv2.getPerspectiveTransform(destination_pts, source_pts)
    return M, Minv


# Re-using one of my functions used in the first detection project
# Modified to crop car hood
def trapezoid_vertices(image, bottom_width=0.85,top_width=0.07,height=0.40, car_hood=45):
    """
    Create trapezoid vertices for mask. 
    Inputs:
    image
    bottom_width = percentage of image width
    top_width = percentage of image width
    height = percentage of image height
    car_hood = number of pixels to be cropped from bottom meant to get rid of car's hood
    """   
    imshape = image.shape
    
    vertices = np.array([[\
        ((imshape[1] * (1 - bottom_width)) // 2, imshape[0]-car_hood),\
        ((imshape[1] * (1 - top_width)) // 2, imshape[0] - imshape[0] * height + car_hood),\
        (imshape[1] - (imshape[1] * (1 - top_width)) // 2, imshape[0] - imshape[0] * height + car_hood),\
        (imshape[1] - (imshape[1] * (1 - bottom_width)) // 2, imshape[0] - car_hood)]]\
        , dtype=np.int32)
    
    return vertices



def get_perspective_and_pickle_M_Minv():
    # Optimize source points by using straight road test image
    # Load image
    Readname = join(road_test_images_dir, road_straight_image_filename)
    img = cv2.imread(Readname)
    if mode == 1:
        #Original
        img_size = (img.shape[1], img.shape[0])
        testImage = img.copy()
        # Get the points by image ratios
        pts = trapezoid_vertices(img, bottom_width=bottom_width,top_width=top_width,height=height, car_hood=car_hood)
        # Modify it to expected format
        pts = pts.reshape(pts.shape[1:])
        pts = pts.astype(np.float32)
        # Give user some data to check
        print('Here are the initial src pts:', pts)
        # get the transform matrices
        M, Minv = get_transform_matrices(pts, img_size) 
        cv2.circle(testImage, (pts[0,0], pts[0,1]), 10, (255,0,0), thickness=5, lineType=8, shift=0) 
        cv2.circle(testImage, (pts[1,0], pts[1,1]), 10, (255,0,0), thickness=5, lineType=8, shift=0) 
        cv2.circle(testImage, (pts[2,0], pts[2,1]), 10, (255,0,0), thickness=5, lineType=8, shift=0) 
        cv2.circle(testImage, (pts[3,0], pts[3,1]), 10, (255,0,0), thickness=5, lineType=8, shift=0)
        # transform image and save it
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
        write_name1 = join(road_straight_warped_image_dir, 'Warped_' + basename(Readname) )
        cv2.imwrite(write_name1,warped)
    else:
        testImage = img[0:img.shape[0], 0:img.shape[1]].copy() #y,x # 30 y 25
        img_size = (testImage.shape[1], testImage.shape[0])
        #Order is: topLeft, topRight, bottomLeft, bottomRight
        if mode == 2: #Webcam
            print "Tests"
            #pts = np.array(((135,200),[458,200],[0, 270],[608, 270]), dtype=np.int32)
        elif (mode == 3) or (mode == 4): #ADAS Camera
            pts = np.array(([20,265],[620,265],[30, 340],[610, 340]), dtype=np.int32)
            #pts = np.array(((180,265),[458,265],[30, 340],[608, 340]), dtype=np.int32) #backup
        else:
            print "Pending"
            #pts = np.array(((135,300),[458,300],[0, 400],[608, 400]), dtype=np.int32)

        cv2.circle(testImage, (pts[0,0], pts[0,1]), 10, (255,0,0), thickness=5, lineType=8, shift=0) 
        cv2.circle(testImage, (pts[1,0], pts[1,1]), 10, (255,0,0), thickness=5, lineType=8, shift=0) 
        cv2.circle(testImage, (pts[2,0], pts[2,1]), 10, (255,0,0), thickness=5, lineType=8, shift=0) 
        cv2.circle(testImage, (pts[3,0], pts[3,1]), 10, (255,0,0), thickness=5, lineType=8, shift=0)
        source_pts = np.float32([(pts[0,0], pts[0,1]), (pts[1,0], pts[1,1]), (pts[2,0], pts[2,1]), (pts[3,0], pts[3,1])])
        destination_pts = np.float32([[0, 0], [img_size[0],0], [0,img_size[1]], [img_size[0], img_size[1]]])
        M, Minv = get_transform_matrices_2(source_pts, destination_pts) 
        # transform image and save it
        warped = cv2.warpPerspective(img, M, img_size)
        write_name1 = join(road_straight_warped_image_dir, 'Warped_' + basename(Readname) )
        cv2.imwrite(write_name1,warped)
 
    

    # Monox edits
    write_name2 = join(road_straight_warped_image_dir, 'Warped2_' + basename(Readname) )
    cv2.imwrite(write_name2, testImage)


    # Save the transformation matrices for later use
    dist_pickle = {}
    dist_pickle["M"] = M
    dist_pickle["Minv"] = Minv
    dist_pickle["pts"] = pts
    dist_pickle["mode"]= mode
    write_name2 = join(M_Minv_output_dir,'perspective_trans_matrices.p')
    pickle.dump( dist_pickle, open( write_name2, "wb" ) )
    
    print('Done!')
    print("Warped image test: from [" + basename(Readname)  + "] to [" + basename(write_name1) + "]")
    print("Here is the warped image: [" + write_name1  + "]")
    print("M and Minv saved: [pickled file saved to: " + write_name2  + "]")
    
    
if __name__ == '__main__':
    get_perspective_and_pickle_M_Minv()
    
    
