import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

image = mpimg.imread('warped-example.jpg')/255
plt.imshow(image)

def hist(img):
    # Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[int(img.shape[0]/2):,:]
    #Sum across image pixels vertically - make sure to set `axis`
    # i.e. the highest areas of vertical lines should be larger values#
    histogram = bottom_half.sum(0)
    return histogram
	
# Create histogram of image binary activations
histogram = hist(image)
# Visualize the resulting histogram
plt.plot(histogram)