import cv2
from matplotlib import pyplot as plt

img = cv2.imread('fapiao.jpg')
img = img[1500:2100,0:400]
height,width = img.shape[:2]
size = (int(height*3),int(width*5))
img = cv2.resize(img,size)
plt.imshow(img)
plt.show()

cv2.namedWindow('image')
cv2.imshow('image',img)
cv2.waitKey(0)