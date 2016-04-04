import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('mnist-2.png')
kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)
dilation = cv2.dilate(img,kernel,iterations = 1)
plt.imshow(erosion),plt.title('Original'),plt.yticks([]),plt.xticks([])
plt.imshow(dilation),plt.title('Original1'),plt.yticks([]),plt.xticks([])
plt.show()