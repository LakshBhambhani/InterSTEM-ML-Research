#https://www.geeksforgeeks.org/opencv-motion-blur-in-python/

# loading library 
import cv2 
import numpy as np 
import glob
import os

# Specify the kernel size. 
# The greater the size, the more the motion. 
kernel_size = 5
    
# Create the vertical kernel. 
kernel_v = np.zeros((kernel_size, kernel_size)) 
    
# Create a copy of the same for creating the horizontal kernel. 
kernel_h = np.copy(kernel_v) 
    
# Fill the middle row with ones. 
kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size) 
kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size) 
    
# Normalize. 
kernel_v /= kernel_size 
kernel_h /= kernel_size 

for origImg in glob.glob('../LungCancerDetection/data/test/*'):

    img = cv2.imread(origImg) 
    
    # Apply the vertical kernel. 
    # vertical_mb = cv2.filter2D(img, -1, kernel_v) 
    
    # Apply the horizontal kernel. 
    horizonal_mb = cv2.filter2D(img, -1, kernel_h) 

    filename = os.path.basename(origImg)
    
    # Save the outputs. 
    # cv2.imwrite('car_vertical.jpg', vertical_mb) 
    cv2.imwrite('../data/blurred_scans/test/' + filename, horizonal_mb) 

