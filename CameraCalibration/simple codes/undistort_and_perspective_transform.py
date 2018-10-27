import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
img = cv2.imread('calibration_wide/test_image2.png')
plt.imshow(img)
nx = 8 # the number of inside corners in x
ny = 6 # the number of inside corners in y

def corners_unwarp(img, nx, ny, mtx, dist):
    #find optimal matrix for output image
    imageSize = (img.shape[1],img.shape[0])
    #alpha = 1
    #newCameraMatrix,roi = cv2.getOptimalNewCameraMatrix(mtx, dist, imageSize, alpha, imageSize)
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)  
    # crop the image
    #x,y,w,h = roi
    #undistorted = undistorted[y:y+h, x:x+w]
    
    undistorted_gray = cv2.cvtColor(undistorted,cv2.COLOR_BGR2GRAY)
    #Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(undistorted_gray, (nx, ny), None)
    # If found, draw corners
    if ret == True:
    # Draw the corners
        cv2.drawChessboardCorners(undistorted_gray, (nx, ny), corners, ret)
    src = np.float32([corners[7][0],corners[47][0],corners[40][0],corners[0][0]])
    des = np.float32([[1030,200],[1030,760],[250,760],[250,200]])
    M = cv2.getPerspectiveTransform(src, des)
    warped = cv2.warpPerspective(undistorted_gray, M, imageSize, flags=cv2.INTER_LINEAR)
    return warped, M
	
top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down, cmap='gray')
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	
	