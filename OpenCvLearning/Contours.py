"""
Contours can be explained simply as a curve joining all the continuous points (along the boundary), having same color or intensity. The contours are a useful tool for shape analysis and object detection and recognition.

For better accuracy, use binary images. So before finding contours, apply threshold or canny edge detection.
findContours function modifies the source image. So if you want source image even after finding contours, already store it to some other variables.
In OpenCV, finding contours is like finding white object from black background. So remember, object to be found should be white and background should be black.
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

im = cv2.imread('messi5.jpg')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

"""
To draw the contours, cv2.drawContours function is used.
It can also be used to draw any shape provided you have its boundary points.
Its first argument is source image, second argument is the contours which should be passed as a Python list,
third argument is index of contours (useful when drawing individual contour.
To draw all contours, pass -1) and remaining arguments are color, thickness etc.
"""

cv2.drawContours(im, contours, -1, (0,255,0), 3)
# cv2.drawContours(img, contours, 3, (0,255,0), 3)
# # cnt = contours[4]
# # cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
#
# plt.imshow(a,cmap = 'gray')
# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
# #
# plt.show()