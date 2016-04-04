import cv2
import numpy as np
from matplotlib import pyplot as plt
## Translation:
img = cv2.imread('messi5.jpg',0)
rows,cols = img.shape
print  rows,cols
M = np.float32([[1,0,100],[0,1,50]])
dst = cv2.warpAffine(img,M,(cols,rows))
cv2.imshow('ScalledImage',dst)
cv2.imshow('OriginalImage',img)

## Rotation: Rotatiopn 90 deg

M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
dst1 = cv2.warpAffine(img,M,(cols,rows))
cv2.imshow('imgRotation',dst1)



## Example of Affine Transformation:
"""
In affine transformation, all parallel lines in the original image will still be parallel in the output image.
To find the transformation matrix, we need three points from input image and their corresponding locations in output image.
Then cv2.getAffineTransform will create a 2x3 matrix which is to be passed to cv2.warpAffine.
"""

img = cv2.imread('drawing.png')
rows,cols,ch = img.shape

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

M = cv2.getAffineTransform(pts1,pts2)

dst = cv2.warpAffine(img,M,(cols,rows))

# plt.subplot(121),plt.imshow(img),plt.title('Input')
# plt.subplot(122),plt.imshow(dst),plt.title('Output')
# plt.show()

##Waiting and destory after visualization
cv2.waitKey(0)
cv2.destroyAllWindows()