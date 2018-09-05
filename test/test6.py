import cv2
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from pyzbar.pyzbar import decode

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir)))
TEMP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,'temp3'))
if not os.path.isdir(TEMP_PATH):
    os.mkdir(TEMP_PATH)

image_segmentation_erode = 25
image_segmentation_dilate = 25
image_segmentation_blur = 17  # 实现均值滤波
DEBUG_LEVEL = 1

# url_image = cv2.imread('fapiao.jpg')
def image_segmentation(url_image):
    """
         分割二维码
         :param url_image:原图片
         :return: 分割出来的图片
         """
    global image_segmentation_erode
    global image_segmentation_dilate
    global image_segmentation_blur
    image_length, image_width, image_dimension = url_image.shape  # 判断输入的是rgb图像还是灰度图像
    if image_dimension == 3:
        blue_channel, green_channel, red_channel = cv2.split(url_image)  # 取r通道的图像
        image_scale = red_channel
    else:
        image_scale = url_image
    enhanceimage = cv2.createCLAHE(clipLimit=2, tileGridSize=(10, 10))
    image_scale = enhanceimage.apply(image_scale)
    # cv2.namedWindow('image')
    # cv2.imshow('image',image_scale)
    # cv2.waitKey(0)
    # image_scale.show()
    image_filtering_x_direction = cv2.Sobel(image_scale, cv2.CV_16S, 1, 0)  # 使用sobel算子沿x、y方向进行滤波
    image_filtering_y_direction = cv2.Sobel(image_scale, cv2.CV_16S, 0, 1)
    image_subtract = cv2.subtract(image_filtering_x_direction, image_filtering_y_direction)
    image_format_conversion = cv2.convertScaleAbs(image_subtract)  # 转换格式
    image_blurred = cv2.blur(image_format_conversion, (image_segmentation_blur, image_segmentation_blur))
    (rew, thresh) = cv2.threshold(image_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, None)  # 闭运算消除二维码间的缝隙
    image_erode = cv2.erode(image_closed, None, iterations=image_segmentation_erode)  # 消除小斑点
    image_dilate = cv2.dilate(image_erode, None, iterations=image_segmentation_dilate)  # 剩余的像素扩张并重新增长回去
    image_canny = cv2.Canny(image_dilate, 150, 1000)  # canny边缘检测
    (rew, image_binary) = cv2.threshold(image_canny, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # if DEBUG_LEVEL:
    #     cv2.imshow('image_closed',image_closed)
    #     cv2.waitKey(0)
    #     cv2.imshow('image_erode',image_erode)
    #     cv2.waitKey(0)
    #     cv2.imshow('image_dilate',image_dilate)
    #     cv2.waitKey(0)
    #     cv2.imshow('image_canny',image_canny)
    #     cv2.waitKey(0)
    #     cv2.imshow('image_binary',image_binary)
    #     cv2.waitKey(0)
    image_area, contours, hierarchy = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找出最大面积
    if len(contours) > 0:
        image_outline = sorted(contours, key=cv2.contourArea, reverse=True)[0]  # 找到最大轮廓
        # if DEBUG_LEVEL:
        #     cv2.drawContours(url_image, contours, -1, (255, 0, 0), 2)
        #     cv2.imshow('contours', url_image)
        #     cv2.drawContours(url_image, [image_outline], -1, (0, 255, 0), 3)
        #     cv2.imshow('image_outline', url_image)
        #     cv2.waitKey(0)
        image_frame = cv2.minAreaRect(image_outline)  # 为最大轮廓确定最小边框
        segmentation_image = np.int0(cv2.boxPoints(image_frame))  # 输出图像
        image_top_left = abs(min(segmentation_image[2, 1], segmentation_image[0, 1]))
        image_top_right = max(segmentation_image[2, 1], segmentation_image[0, 1])
        image_bottom_left = abs(min(segmentation_image[0, 0], segmentation_image[2, 0]))
        image_bottom_right = max(segmentation_image[0, 0], segmentation_image[2, 0])
        if image_top_left <= 10:
            image_top_left = 10
        if image_bottom_left <= 10:
            image_bottom_left = 10
        Segmentation_image = url_image[image_top_left - 10:image_top_right + 10,
                             image_bottom_left - 10:image_bottom_right + 20, :]
    else:
        Segmentation_image = url_image
    # print(Segmentation_image)
    return Segmentation_image

def handle_im(Segmentation_image):
    # if DEBUG_LEVEL:
    #     cv2.imshow('Segmentation_image',Segmentation_image)
    #     cv2.waitKey(0)
    gray_image = cv2.cvtColor(Segmentation_image, cv2.COLOR_BGR2GRAY)  # 灰度化图像
    enhanceimage = cv2.createCLAHE(clipLimit=2, tileGridSize=(10, 10))
    gray_image = enhanceimage.apply(gray_image)
    (rew, binary_image) = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if DEBUG_LEVEL:
        cv2.imshow('gray_image',gray_image)
        cv2.imshow('binary_image',binary_image)
        cv2.waitKey(0)
    return binary_image

def image_handle(segmentation_image, number_one, number_two):
    """
          二维码图片预处理
          :param Segmentation_image:分割出来的图片
          :return: 预处理后的图片
          """
    gray_image = cv2.cvtColor(segmentation_image, cv2.COLOR_BGR2GRAY)  # 灰度化图像
    enhanceimage = cv2.createCLAHE(clipLimit=2, tileGridSize=(10, 10))
    histogram_image = enhanceimage.apply(gray_image)
    (rew, binary_image) = cv2.threshold(histogram_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_image_height, binary_image_width = binary_image.shape
    if DEBUG_LEVEL:
        cv2.imshow('gray_image', gray_image)
        cv2.waitKey(0)
        cv2.imshow('binary_image', binary_image)
        cv2.waitKey(0)
    zoom_image = cv2.resize(binary_image, (2 * binary_image_width, 2 * binary_image_height),
                            interpolation=cv2.INTER_CUBIC)  # 以4*4立方插值 使放大后的图像平滑
    image_closed = cv2.GaussianBlur(zoom_image, (3, 3), 0)
    (rew, image_turn) = cv2.threshold(image_closed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # 图像反转
    image_dilate = cv2.dilate(image_turn, None, iterations=number_one)
    image_erode = cv2.erode(image_dilate, None, iterations=number_two)
    (rew, image_back_binary) = cv2.threshold(image_erode, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    (rew, handle_image) = cv2.threshold(image_back_binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if DEBUG_LEVEL:
        cv2.imshow('zoom_image', zoom_image)
        cv2.waitKey(0)
        cv2.imshow('GaussianBlur', image_closed)
        cv2.waitKey(0)
        cv2.imshow('image_turn', image_turn)
        cv2.waitKey(0)
        cv2.imshow('image_dilate',image_dilate)
        cv2.waitKey(0)
        cv2.imshow('image_erode', image_erode)
        cv2.waitKey(0)
        cv2.imshow('image_back_binary', image_back_binary)
        cv2.waitKey(0)
        cv2.imshow('handle_image3', handle_image)
        cv2.waitKey(0)

    return handle_image

if __name__ == '__main__':
    url_image = cv2.imread('duoci.jpg')
    image_length,image_width,image_dim = url_image.shape
    sub_image = url_image[int(image_length*0.85):int(image_length),0:int(image_width*0.40),:]
    # cv2.namedWindow('image')
    # cv2.imshow('image', sub_image)
    # cv2.waitKey(0)

    # image_length, image_width, image_dim = url_image.shape     # 获取图片的三维
    # sub_image = url_image[0:int(image_length * 0.25), 0:int(image_width * 0.35), :]  # 对图片进行剪切

    # plt.imshow(url_image)
    # plt.show()
    # url_image = area_image(url_image)
    # cv2.namedWindow('image')
    # cv2.imshow('image',url_image)
    # cv2.waitKey(0)
    # plt.imshow(url_image)
    # plt.show()
    segmentation_image = image_segmentation(sub_image)
    # plt.imshow(Segmentation_image)
    # plt.show()
    # handle_image = handle_im(segmentation_image)
    handle_image = image_handle(segmentation_image, 2, 1)
    # if DEBUG_LEVEL:
    #     cv2.imshow('handle_image', handle_image)
    #     cv2.waitKey(0)
    # qr_image = handle_im(Segmentation_image)
    # print(qr_image.size)
    # plt.imshow(qr_image)
    # plt.show()
    # decode_data = decode_by_pyzbar(qr_image)
    # print(decode_data)



