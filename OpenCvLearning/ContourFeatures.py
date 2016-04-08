"""
To find the different features of contours, like area, perimeter, centroid, bounding box etc
You will see plenty of functions related to contours.
"""
import cv2
import numpy as np

img = cv2.imread('messi5.jpg',0)
ret,thresh = cv2.threshold(img,127,255,0)
contours,hierarchy = cv2.findContours(thresh, 1, 2)

cnt = contours[0]
M = cv2.moments(cnt)
print M