import cv2
from matplotlib import pyplot as plt
import numpy as np
import urllib.request

# img = cv2.imread('fapiao1.jpg')
# plt.imshow(img)
# plt.show()
# length,width,dim = img.shape
# img = img[0:int(length*0.25),0:int(width*0.35),:]
# plt.imshow(img)
# plt.show()
# cv2.namedWindow('image')
# cv2.imshow('image',img)
# cv2.waitKey(0)


# img_dc = cv2.imread('fapiao.jpg')
# plt.imshow(img_dc)
# plt.show()
# length,width,dim1 = img_dc.shape
# img_dc = img_dc[int(length*0.8):int(length),0:int(width*0.7),:]
# cv2.namedWindow('image')
# cv2.imshow('image',img_dc)
# cv2.waitKey(0)
# plt.imshow(img_dc)
# plt.show()

# def url_conversion(url):
#     url_obtain = urllib.request.urlopen(url)
#     url_image = np.asarray(bytearray(url_obtain.read()),dtype = 'uint8')
#     url_image = cv2.imdecode(url_image,cv2.IMREAD_COLOR)
#     return url_image
#
# if __name__ == '__main__':
#     url_image = url_conversion('http://jfjun.img-cn-hangzhou.aliyuncs.com/resources/59f7ccc941e8779b4b3bb53f/2018-07/35b5692351ca2acbdf7018ce3ef651ec.jpg')
#     cv2.namedWindow('image')
#     cv2.imshow('image',url_image)
#     cv2.waitKey(0)

# img_dc = cv2.cvtColor(img_dc,cv2.COLOR_BGR2GRAY)
# ret,img_dc = cv2.threshold(img_dc,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# image,contours,hierarchy = cv2.findContours(img_dc,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
img = cv2.imread('fapiao.jpg')
# height,width = img.shape[:2]  #获取原图像的水平方向尺寸和垂直方向尺寸。
res = cv2.resize(img,(300,500),interpolation=cv2.INTER_CUBIC)
cv2.imshow('image',res)
cv2.waitKey(0)
# length,width,= img_dc.shape[:2]
#
# res = cv2.resize(img_dc,(0.2*width,0.2*length),interpolation=cv2.INTER_CUBIC)
# # size = [int(length*0.25),int(width*0.35)]
# # im = cv2.resize(img_dc,size)
# cv2.namedWindow('image')
# cv2.imshow('image',res)
# cv2.waitKey(0)
# print(len(contours))
# print(contours[0])
# print(hierarchy.ndim)
# print(hierarchy[0].ndim)
# print(hierarchy.shape)
# plt.imshow(image)
# plt.show()
# cv2.namedWindow('image')
# # cv2.imshow('image',image)
# # cv2.waitKey(0)




