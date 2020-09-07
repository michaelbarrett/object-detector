# Object Detection Using HSV Color Detection In OpenCV

## Overview

The Object Detector detects objects in a photo by color and returns their coordinates in the photo as well as size, shape, and other properties. Currently, objects colored red, blue, and green are detectable. OpenCV's contour detection requires no machine learning making this a fast solution to the problem of detecting objects on a manufacturing table. For HSV color encoding, the level of light or shadow on the object is less important than it would be for color detection with RGB color encoding, allowing an object to be reliably detected as long as it is colored distinctly.

## Functionality

### Detecting a red object

!(redobj)[red-house-detect.png]

### Detecting a blue object

!(blueobj)[blue-house-detect.png]

## Other Possible Features

- The ability to detect objects with colors other than red, blue, and green
- The ability to compute the average color of a particular object in the photo as a rudimentary means of object recognition
- The use of machine learning (custom-trained CNNs) to detect the identity of an object as a finer means of object recognition
