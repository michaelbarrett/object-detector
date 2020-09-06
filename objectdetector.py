# Finds the largest shape components in an image and provides an analysis of their properties.

import argparse
import imutils
import cv2
import numpy as np

# Constants (configurable)
CONTOUR_MIN_PERIMETER = 22 # remove all contours with a smaller perimeter than this length
CONTOURS_TO_DISPLAY = 1 # analyze this many contours in output
IMAGE_RESIZE_WIDTH = 500 # resize given image to this width before processing
RED_HSV_BOUND_WIDTH = 20 # width of HSV slice considered "red" (greater = more objects considered red)
GREEN_HSV_BOUND_WIDTH = 10 # width of HSV slice considered "green" (greater = more objects considered green)
BLUE_HSV_BOUND_WIDTH = 40 # width of HSV slice considered "blue" (greater = more objects considered blue)

# This object detects the shape of a given contour
class ShapeDetector:
    def __init__(self):
        pass
    def detect(self, c): # param: c, the contour of the shape to id
        # initialize the shape name and approximate the contour
        shape = "Unidentified"
        peri = cv2.arcLength(c, True)
        # contour approximation using cv2.approxPolyDP method
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise the shape is a rectangle
            if (ar >= 0.95 and ar <= 1.05):
                shape = "Square"
            else:
                shape = "Rectangle"
        else:
            shape = "Unknown"
        return shape

# is_contour_too_small: used to filter out small contours.
# returns false if contour perimeter is less than bound specified by CONTOUR_MIN_PERIMETER
def is_contour_too_small(c):
    perimeter = cv2.arcLength(c, True)
    #print("perimeter: ", perimeter)
    return perimeter < CONTOUR_MIN_PERIMETER

sd = ShapeDetector()
# construct the arg parser and parse the args
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-c", "--color", required=True,
                help="the color to detect")
args = vars(ap.parse_args())

# load the input image (whose path was supplied via command line argument)
image = cv2.imread(args["image"])
# resize?
image = imutils.resize(image, width=IMAGE_RESIZE_WIDTH)
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # convert to HSV
#ratio = image.shape[0] / float(image.shape[0])
#cv2.imshow("Image", image)
#cv2.waitKey(0)

# find the colors within the specified boundaries and apply
# the color mask to the image
# RED
if "red" in args["color"]:
    mask1 = cv2.inRange(image_hsv, (0, 100, 0), (0 + (RED_HSV_BOUND_WIDTH / 2), 255, 255))
    mask2 = cv2.inRange(image_hsv, (180 - (RED_HSV_BOUND_WIDTH / 2), 100, 0), (180, 255, 255))
    mask = mask1 + mask2
# GREEN    
elif "green" in args["color"]:
    mask = cv2.inRange(image_hsv, (60 - (GREEN_HSV_BOUND_WIDTH / 2), 100, 0), (60 + (GREEN_HSV_BOUND_WIDTH / 2), 255, 255))    
# BLUE
elif "blue" in args["color"]:
    mask = cv2.inRange(image_hsv, (120 - (BLUE_HSV_BOUND_WIDTH / 2), 100, 0), (120 + (BLUE_HSV_BOUND_WIDTH / 2), 255, 255))
else:
    print("Error: color not supported yet")
    exit(0)
                
output = cv2.bitwise_and(image, image, mask=mask)

cv2.imshow("Masked", output)
cv2.waitKey(0)

# threshold the image by setting all pixel values less than 225
# to 225 (white; foreground) and all pixel values >= 225 to 0
# (black; background), thereby segmenting the image
# use experience + trial and error to determine threshold values
ret, thresh = cv2.threshold(mask, 70, 255, 0)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

cv2.imshow("Thresholded", thresh)
cv2.waitKey(0)

# add contours that fit size criteria to contours_list
contours_list = []
for c in contours:
    if (not is_contour_too_small(c)):
        contours_list.append(c)

num_contours = len(contours_list)
displaying_contours = num_contours
if num_contours != 0:

    # if num_contours < CONTOURS_TO_DISPLAY:
    # displaying_contours = num_contours
    
    print("We found this many objects: ", num_contours)
    print("Displaying this many objects: ", CONTOURS_TO_DISPLAY)
    
    # draw in blue the contours that were found
    cv2.drawContours(output, contours_list, -1, 255, 3)

    #contours.remove(max(contours, key = cv2.contourArea))
    contours_list = sorted(contours_list, key=cv2.contourArea, reverse=True)

    i = 0
    for c in contours:
        print("-----------")
        print("-----------")
        print("-----------")
        print("OBJECT", (i + 1))
        print("-----------")
        c = contours_list[i]
        shape = sd.detect(c)
        x,y,w,h = cv2.boundingRect(c)
        bounding_rect_area = w*h
        peri = cv2.arcLength(c, True)
        print("Object shape:", shape)
        print("Object perimeter:", peri)
        print("Object area: ", cv2.contourArea(c))        
        print("Top-left corner x: ", x)
        print("Top-left corner y: ", y)
        print("Center x: ", x + (w / 2))
        print("Center y: ", y + (h / 2))
        print("Width:", w)
        print("Height:", h)
        print("Aspect ratio:", (h / w))

        # draw the biggest contour (c) in green
        cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)

        if (i+1) >= CONTOURS_TO_DISPLAY: # displayed all desired contours
            break
        
        i += 1
else:
    print("No objects of that color found! Try a different image or a different color.")
        
# show the images
cv2.imshow("Result", np.hstack([image, output]))

cv2.waitKey(0)

exit()

