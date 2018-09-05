# -*- coding:utf-8 -*-
import cv2
from matplotlib import pyplot as plt
import urllib
import numpy as np
import pyzbar.pyzbar as pyzbar
import webbrowser
from PIL import Image

def decode(im):
    objectt = pyzbar.decode(im)

    for obj in objectt:
        print('Data:',obj.data,'\n')
        webbrowser.open(obj.data)
# img_src = 'http://jfjun.img-cn-hangzhou.aliyuncs.com/resources/59f7ccc941e8779b4b3bb53f/2018-07/35b5692351ca2acbdf7018ce3ef651ec.jpg'
img_src = 'http://jfjun.img-cn-hangzhou.aliyuncs.com/resources/5b1f65996df6870522b7208d/2018-07/4dab9d62dbee370b5f0149e37bf32f9a.jpg'
# img_src = 'http://jfjun.img-cn-hangzhou.aliyuncs.com/resources/5a02cdc846b538d80f2fdde0/2018-07/ce020cc8cb2b877d266fcf253bf4ad26.jpg?x-oss-process=image/rotate,359'
# img_src = 'http://jfjun.img-cn-hangzhou.aliyuncs.com/resources/5b31ae528fc92d3a52bd6396/2018-07/e34e7679aa8cec5f6c5230c8fbadaa13.jpg?x-oss-process=image/rotate,92'
cap = cv2.VideoCapture(img_src)
if (cap.isOpened()):
    ret,img = cap.read()
    img = img[1000:2080, 0:400]

    img = cv2.resize(img, (480, 640))  # 缩放图像
    im = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图
    img_gb = cv2.GaussianBlur(img,(5,5),0)
    edges = cv2.Canny(img_gb, 100, 200)

    # thresh, i = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)  # 对灰度图进行二值化处理
    # i = cv2.bitwise_not(i)  # 将二值化图片进行效果反转，即：黑变白，白变黑
    # i = cv2.dilate(i, None, iterations=7)  # 腐蚀和膨胀，去除图片中的干扰像素，使背景与目标完全分离
    # i = cv2.erode(i, None, iterations=3)
    image, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓，采用层级嵌套的方式

    hierarchy = hierarchy[0]
    found = []
    for i in range(len(contours)):
        k = i
        c = 0
        while hierarchy[k][2] != -1:
            k = hierarchy[k][2]
            c = c + 1
        if c >= 5:
            found.append(i)

    for i in found:
        img_dc = img.copy()
        cv2.drawContours(img_dc, contours, i, (0, 255, 0), 3)
        plt.imshow(img_dc)
        plt.show()
        # show(img_dc)

    draw_img = img.copy()
    for i in found:
        rect = cv2.minAreaRect(contours[i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(draw_img, [box], 0, (0, 0, 255), 2)

    # cv2.namedWindow('image')
    # cv2.imshow('image',draw_img)
    # cv2.waitKey(0)
    # plt.imshow(draw_img)
    # plt.show()
    # show(draw_img)

    # if (contours):
    #     #     area = [cv2.contourArea(cnt) for cnt in contours]  # 得到轮廓面积
    #     #     index = np.argmax(area)
    #     #     cnt = contours[index]
    #     #     x, y, w, h = cv2.boundingRect(cnt)  # 计算边界矩形相关参数
    #     #     cv2.imwrite("barcode.jpg", im[y:y + h, x:x + w])  # 对目标进行标定
    #     #     new = im[y:y + h, x:x + w]
    #     #     decode(new)
    #     #     cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 用矩形框图框出目标
    #     #     cv2.imshow('fame', im)
    #     #     cv2.waitKey(0)
