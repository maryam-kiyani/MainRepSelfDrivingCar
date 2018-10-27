import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

# Load our image
binary_warped = mpimg.imread('warped_example.jpg')

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 200
    # Set minimum number of pixels found to recenter window
    minpix = 60

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    #left_lane_inds = []
    #right_lane_inds = []
    left_lane_inds = np.zeros_like(nonzerox, dtype=bool)
    right_lane_inds = np.zeros_like(nonzerox, dtype=bool)

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - (margin//2)  
        win_xleft_high = leftx_current + (margin//2)
        win_xright_low = rightx_current - (margin//2) 
        win_xright_high = rightx_current + (margin//2) 
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        #return out_img
        
        ### Done: Identify the nonzero pixels in x and y within the window ###
        boolYmin = (nonzeroy >= win_y_low)
        boolYmax = (nonzeroy <= win_y_high)
        booly = boolYmin & boolYmax
        
        boolXmin = (nonzerox >= win_xleft_low)
        boolXmax = (nonzerox <= win_xleft_high)
        boolx_left = boolXmin & boolXmax
        boolxy_left = booly & boolx_left
        good_left_inds = boolxy_left
        """
        good_left_inds = np.array([nonzeroy[boolxy_left],nonzeroy[boolxy_left]])
        good_left_inds=good_left_inds.tolist()
        """
        boolXmin = (nonzerox >= win_xright_low)
        boolXmax = (nonzerox <= win_xright_high)
        boolx_right = boolXmin & boolXmax
        boolxy_right = booly & boolx_right
        good_right_inds =boolxy_right
        
        """
        good_right_inds = np.array([nonzeroy[boolxy_right],nonzeroy[boolxy_right]])
        good_right_inds = good_right_inds.tolist()
        """
        # Append these indices to the lists
        left_lane_inds=left_lane_inds | good_left_inds
        right_lane_inds=right_lane_inds | good_right_inds
        
        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        val_idxes_in_left_win =  nonzerox[good_left_inds]
        if(np.size(val_idxes_in_left_win) > minpix):
            leftx_current = sum(val_idxes_in_left_win)//np.size(val_idxes_in_left_win)
        val_idxes_in_right_win =  nonzerox[good_right_inds]
        if(np.size(val_idxes_in_right_win) > minpix):
            rightx_current = sum(val_idxes_in_right_win)//np.size(val_idxes_in_right_win)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds] #this is eqv. to boolxy_left for all windows
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds] #this is ---  boolxy_right ---
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty,leftx,2)  
    right_fit = np.polyfit(righty,rightx,2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return out_img


out_img = fit_polynomial(binary_warped)

plt.imshow(out_img)