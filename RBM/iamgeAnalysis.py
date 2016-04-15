import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('C:\Users\Rakib\Documents\messi5.jpg')

r,g,b = cv2.split(img)
img_bgr = cv2.merge([r,g,b])
b=cv2.fromarray(r)
print b.shape
#A =np.asanyarray(r)
# print r.dtype
# print r.shape
# print type(r)
#print A.shape
# cv2.imshow('image',img_bgr)
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#img = cv2.imread('messi5.jpg',0)
#
fig = plt.figure(1)
plt.subplot(221),plt.imshow(r,'gray'),plt.title('Original')
plt.subplot(222),plt.imshow(g,'gray'),plt.title('Original')
plt.subplot(223),plt.imshow(b,'gray'),plt.title('Original')
plt.subplot(224),plt.imshow(img_bgr,'gray'),plt.title('Reconstract')
# #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis # to remove x and y levels
plt.show()

# #How to add another plot
# fig = plt.figure(5)
# plt.subplot(121),plt.imshow(img,'gray'),plt.title('Original')
# plt.subplot(122),plt.imshow(img_bgr,'gray'),plt.title('Reconstract')
# #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()