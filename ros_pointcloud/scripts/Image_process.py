import numpy as np
import cv2
import math

def stereo_rectification(left_image, right_image, left_cameraMatrix, left_distCoeffs, right_cameraMatrix, right_distCoeffs, image_size, R, T):
    # distortion correct
    left_undistort_image = cv2.undistort(left_image, left_cameraMatrix, left_distCoeffs, None, left_cameraMatrix)
    right_undistort_image = cv2.undistort(right_image, right_cameraMatrix, right_distCoeffs, None, right_cameraMatrix)
    # stereo rectify
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(left_cameraMatrix, left_distCoeffs, right_cameraMatrix, right_distCoeffs, image_size, R, T, alpha=0)
    
    map1_x, map1_y = cv2.initUndistortRectifyMap(left_cameraMatrix, left_distCoeffs, R1, P1, image_size, cv2.CV_32FC1)
    map2_x, map2_y = cv2.initUndistortRectifyMap(right_cameraMatrix, right_distCoeffs, R2, P2, image_size, cv2.CV_32FC1)
    
    rectified_left = cv2.remap(left_undistort_image, map1_x, map1_y, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(right_undistort_image, map2_x, map2_y, cv2.INTER_LINEAR)
    
    baseline = math.sqrt(np.dot(T,T.T))/1000
    
    return rectified_left, rectified_right, baseline

def resize_crop(input_image, cameraMatrix, width, height, ratio):

    fx, fy, cx, cy = cameraMatrix[0][0], cameraMatrix[1][1], cameraMatrix[0][2], cameraMatrix[1][2]
   
    # resize after stereo rectification
    resized_image = cv2.resize(input_image, (0,0), fx=ratio, fy=ratio)
    fx, fy, cx, cy = fx*ratio, fy*ratio, cx*ratio, cy*ratio
    focal_len = (fx+fy)/2
    # compute location
    start_x = int(max(cx - width/2 , 0))
    start_y = int(max(cy - height/2 , 0))
    end_x = start_x + width
    end_y = start_y + height
    # crop
    cropped_image = resized_image[start_y:end_y, start_x:end_x]
    #cv2.imwrite(f"1.png", cropped_image)
    cx, cy = cx - start_x, cy - start_y
    outimg = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    #print(outimg.shape[1],outimg.shape[0])
    
    return outimg, focal_len
     
    
    
    
    



