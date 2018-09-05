# -*- coding:utf-8 -*-
import cv2
from matplotlib import pyplot as plt
import numpy as np

def show(img, code=cv2.COLOR_BGR2RGB):
    cv_rgb = cv2.cvtColor(img, code)
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(cv_rgb)
    fig.show()

# img = cv2.imread('1.jpg')
img = cv2.imread('fapiao.jpg')
img = img[1000:2080, 0:400]    # 剪切部分区域
# img = img[1500:2100,0:400]
# height,width = img.shape[:2]
# size = (int(height*2),int(width*3))
# img = cv2.resize(img,size)
# plt.imshow(img)
# plt.show()
# cv2.namedWindow('image')
# cv2.imshow('image',img)
# cv2.waitKey(0)
# show(img)

img = cv2.resize(img, (480, 640))      # 缩放图像
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gb = cv2.GaussianBlur(img_gray, (5, 5), 0)
edges = cv2.Canny(img_gray, 100 , 200)
# cv2.namedWindow('image')
# cv2.imshow('image',edges)
# cv2.waitKey(0)
# plt.imshow(edges)
# plt.show()

img_fc, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
        found.append(k)

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
    cv2.drawContours(draw_img,[box], 0, (0,0,255), 2)
# cv2.namedWindow('image')
# cv2.imshow('image',draw_img)
# cv2.waitKey(0)
# plt.imshow(draw_img)
# plt.show()
# show(draw_img)

boxes = []
for i in found:
    rect = cv2.minAreaRect(contours[i])
    box = cv2.boxPoints(rect)
    # box = np.int0(box)
    box = map(tuple, box)
    box = [tuple(x) for x in box]
    boxes.append(box)

def cv_distance(P, Q):
    return int(np.math.sqrt(pow((P[0] - Q[0]), 2) + pow((P[1] - Q[1]),2)))

def check(a, b):
    # 存储 ab 数组里最短的两点的组合
    s1_ab = ()
    s2_ab = ()
    # 存储 ab 数组里最短的两点的距离，用于比较
    s1 = np.iinfo('i').max
    s2 = s1
    for ai in a:
        for bi in b:
            d = cv_distance(ai, bi)
            if d < s2:
                if d < s1:
                    s1_ab, s2_ab = (ai, bi), s1_ab
                    s1, s2 = d, s1
                else:
                    s2_ab = (ai, bi)
                    s2 = d
    a1, a2 = s1_ab[0], s2_ab[0]
    b1, b2 = s1_ab[1], s2_ab[1]
    # 将最短的两个线画出来
    cv2.line(draw_img, a1, b1, (0,0,255), 3)
    cv2.line(draw_img, a2, b2, (0,0,255), 3)

    # a1 = (a1[0] + (a2[0] - a1[0]) * 1 / 14, a1[1] + (a2[1] - a1[1]) * 1 / 14)
    # b1 = (b1[0] + (b2[0] - b1[0]) * 1 / 14, b1[1] + (b2[1] - b1[1]) * 1 / 14)
    # a2 = (a2[0] + (a1[0] - a2[0]) * 1 / 14, a2[1] + (a1[1] - a2[1]) * 1 / 14)
    # b2 = (b2[0] + (b1[0] - b2[0]) * 1 / 14, b2[1] + (b1[1] - b2[1]) * 1 / 14)

for i in range(len(boxes)):
    for j in range(i+1, len(boxes)):
        check(boxes[i], boxes[j])

# plt.imshow(draw_img)
# plt.show()
# show(draw_img)

th, bi_img = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)

# plt.imshow(bi_img)
# plt.show()

# a1 = (a1[0] + (a2[0]-a1[0])*1/14, a1[1] + (a2[1]-a1[1])*1/14)
# b1 = (b1[0] + (b2[0]-b1[0])*1/14, b1[1] + (b2[1]-b1[1])*1/14)
# a2 = (a2[0] + (a1[0]-a2[0])*1/14, a2[1] + (a1[1]-a2[1])*1/14)
# b2 = (b2[0] + (b1[0]-b2[0])*1/14, b2[1] + (b1[1]-b2[1])*1/14)

def isTimingPattern(line):
    # 除去开头结尾的白色像素点
    while line[0] != 0:
        line = line[1:]
    while line[-1] != 0:
        line = line[:-1]
    # 计数连续的黑白像素点
    c = []
    count = 1
    l = line[0]
    for p in line[1:]:
        if p == l:
            count = count + 1
        else:
            c.append(count)
            count = 1
        l = p
    c.append(count)
    # 如果黑白间隔太少，直接排除
    if len(c) < 5:
        return False
    # 计算方差，根据离散程度判断是否是 Timing Pattern
    threshold = 5
    return np.var(c) < threshold

valid = set()
for i in range(len(boxes)):
    for j in range(i+1, len(boxes)):
        if check(boxes[i], boxes[j]):
            valid.add(i)
            valid.add(j)
print(valid)

contour_all = np.array([])
while len(valid) > 0:
    c = found[valid.pop()]
    for sublist in c:
        for p in sublist:
            contour_all.append(p)

rect = cv2.minAreaRect(contour_all)
box = cv2.boxPoints(rect)
box = np.array(box)

draw_img = img.copy()
cv2.polylines(draw_img, np.int32([box]), True, (0, 0, 255), 10)
# cv2.namedWindow('image')
# cv2.imshow('image',draw_img)
# cv2.waitKey(0)
# show(draw_img)



