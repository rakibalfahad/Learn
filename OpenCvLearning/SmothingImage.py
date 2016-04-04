"""
http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_filtering/py_filtering.html#filtering
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('opencv_logo.png')
print img.shape
## 1st kernel
kernel1 = np.ones((5,5),np.float32)/25
dst1 = cv2.filter2D(img,-1,kernel1)

##2nd kernel

kernel2 = np.ones((3,3),np.float32)/9
dst2 = cv2.filter2D(img,-1,kernel2)

#Gaussian Blurring
blur = cv2.GaussianBlur(img,(5,5),0)
#Median Blurring
median = cv2.medianBlur(img,5)


plt.subplot(221),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(dst2),plt.title('Averaging K-1')
plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(dst2),plt.title('Averaging K-2')
plt.xticks([]), plt.yticks([])

plt.show()