# -*- coding: utf-8 -*-
"""
Created on Mon Jun 05 15:27:42 2017

@author: boblee
"""
import numpy as np
import urllib.request
import cv2
from PIL import Image
from pyzbar.pyzbar import decode
import zxing
import os
import sys
import re
from src.ali_qr.profile import *
from src.ali_qr.service import *
import json
from src.ali_qr.upload2 import upload2oss
from config import CONFIG

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import platform

TEMP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'temp4'))
if not os.path.isdir(TEMP_PATH):
    os.mkdir(TEMP_PATH)

image_segmentation_erode = 25  # 图像分割腐蚀的次数
image_segmentation_dilate = 25  # 图像分割膨胀的次数
image_segmentation_blur = 17  # 图像分割均值滤波


def qr_decode_name(url):
    """
    得到图片名字
    :param url: 图片地址
    :return: 图片名字
    """
    name_image = re.search(r'[^/]+(?=\.jpg)', url)
    name_image = name_image.group()
    return name_image


def area_image(image):
    """
           下载图片
           :param image: 原图片
           :return: 原图片去掉空白
           """
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (res, color_gray_bin) = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    length_image, width_image = color_gray_bin.shape
    width_sum = color_gray_bin.sum(axis=1)
    im_choose = np.where(width_sum > (width_image * 0.96) * 255)[0]
    i = 0
    im_bottom = []
    for i in range(len(im_choose) - 2):
        if im_choose[i + 1] - im_choose[i] > 1:  # 确定下一幅图的顶部位置
            im_bottom.append(i + 1)
    if len(im_bottom) != 0:
        cc = im_choose[im_bottom[0] - 1]
        if cc > 20:
            cc = cc - 20
        image = image[cc:im_choose[im_bottom[len(im_bottom) - 1]], :]
    return image


def url_conversion(url):
    """
       下载图片
       :param url: 图片地址
       :return: 原图片
       """
    url_obtain = urllib.request.urlopen(url)
    url_image = np.asarray(bytearray(url_obtain.read()), dtype="uint8")
    url_image = cv2.imdecode(url_image, cv2.IMREAD_COLOR)  # 将比特流转化为图片格式
    return url_image


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
    image_area, contours, hierarchy = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找出最大面积
    if len(contours) > 0:
        image_outline = sorted(contours, key=cv2.contourArea, reverse=True)[0]  # 找到最大轮廓
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
    return Segmentation_image


def handle_im(Segmentation_image):
    gray_image = cv2.cvtColor(Segmentation_image, cv2.COLOR_BGR2GRAY)  # 灰度化图像
    enhanceimage = cv2.createCLAHE(clipLimit=2, tileGridSize=(10, 10))
    gray_image = enhanceimage.apply(gray_image)
    (rew, binary_image) = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Segmentation_image


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
    zoom_image = cv2.resize(binary_image, (2 * binary_image_width, 2 * binary_image_height),
                            interpolation=cv2.INTER_CUBIC)  # 立方插值 使放大后的图像平滑
    image_closed = cv2.GaussianBlur(zoom_image, (3, 3), 0)
    (rew, image_turn) = cv2.threshold(image_closed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # 图像反转
    image_dilate = cv2.dilate(image_turn, None, iterations=number_one)
    image_erode = cv2.erode(image_dilate, None, iterations=number_two)
    (rew, image_back_binary) = cv2.threshold(image_erode, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    (rew, handle_image) = cv2.threshold(image_back_binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return handle_image


def hangle(image):
    (res, color_gray_bin) = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    length_sum = color_gray_bin.sum(axis=1)
    length_image, width_image = color_gray_bin.shape
    im_choose = np.where(length_sum > (width_image * 0.2) * 255)[0]
    width_sum = color_gray_bin.sum(axis=0)
    im_choose1 = np.where(width_sum > (length_image * 0.2) * 255)[0]
    color_gray_bin = color_gray_bin[im_choose[0]:im_choose[len(im_choose) - 1],
                     im_choose1[0]:im_choose1[len(im_choose1) - 1]]
    length, width = color_gray_bin.shape
    (res, color_gray_bin) = cv2.threshold(color_gray_bin, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return color_gray_bin, length, width


def merged_image(image_hand, image_template):
    image_hand = cv2.cvtColor(image_hand, cv2.COLOR_BGR2GRAY)
    image_hand, length, width = hangle(image_hand)
    image_template, length1, width1 = hangle(image_template)
    width1 = abs(width - width1)
    length = min(length1, length)
    image_template = image_template[0:length, 0:width1]
    image_hand = image_hand[0:length, 0:width]
    image_merge = np.hstack((image_template, image_hand))
    length = 255 * np.ones(image_merge.shape[0])
    length_zoom = np.tile(length, (10, 1))
    length_zoom = length_zoom.T
    image_merge_length = np.c_[length_zoom, image_merge, length_zoom]
    width = 255 * np.ones(image_merge_length.shape[1])
    width = width.T
    width_zoom = np.tile(width, (10, 1))
    image_merge_zoom = np.r_[width_zoom, image_merge_length, width_zoom]
    image_merge_zoom = cv2.pyrUp(image_merge_zoom)
    image_merge_zoom = cv2.blur(image_merge_zoom, (3, 3))
    image_merge_zoom = np.uint8(image_merge_zoom)
    (res, image_merge_zoom) = cv2.threshold(image_merge_zoom, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return image_merge_zoom


def decode_by_pyzbar(handle_image):
    """
                pyzbar识别二维码
                :param handle_image:预处理后的图片
                :return: 二维码内容
                """
    pyzbar_decoded = decode(handle_image)
    if not pyzbar_decoded:
        pyzbar_decoded_data = ''
    else:
        data = pyzbar_decoded[0].data
        pyzbar_decoded_data = data.decode("utf-8")

    return pyzbar_decoded_data


def decode_by_zxing(decode_filename):
    """
                pyzbar识别二维码
                :param decode_filename:预处理后的图片
                :return: 二维码内容
                 """

    operating_system = platform.system()
    if operating_system == 'Windows':
        zxfile_home = "E:\download\zxing"
    else:
        zxfile_home = "/root/zxing"
    reader = zxing.BarCodeReader(zxfile_home)  # 路径指向 zxing所git clone的位置
    barcode = reader.decode(decode_filename, True)
    if barcode is None:
        zxing_decoded_data = ''
    else:
        zxing_decoded_data = barcode.data

    return zxing_decoded_data


def ali_qr(image):
    access_key_id = "LTAIrCdaOnR7qxj3"
    access_key_secret = "WC0QoJXvO9K6Pg55vv9x51eEDdwYT9"
    profile = Profile(access_key_id, access_key_secret)

    api_path = r'/green/image/scan'

    client_info = {
        "ip": "127.0.0.1"
    }
    image_url = upload2oss(image)  # "http://jfjun4test.oss-cn-hangzhou.aliyuncs.com/qrcode_temp.jpg"
    post_data = {
        "bizType": "Green",
        "scenes": ["qrcode"],
        "tasks": [
            {
                "dataId": str(uuid.uuid1()),
                # "url": "http://pic.enorth.com.cn/003/017/057/00301705773_66bc4800.gif"
                # "url":"http://jfjun.img-cn-hangzhou.aliyuncs.com/resources/59828e9890b112c8458afe9b/2016-07/1c07ae3e9c5ba7c3af46448347b9513d.jpg"
                "url": image_url
            }
        ]
    }

    service = GreenService(profile)
    response = service.send_request(api_path, client_info, post_data)

    dd = json.loads(response)
    if dd['code'] == 200:
        cc = dd['data'][0]
        cc = cc['results'][0]
        if cc['label'] == 'qrcode':
            cc = cc['qrcodeData'][0]
        else:
            cc = ''
    else:
        cc = ''

    return cc


def image_distinguish(handle_image, file_name):
    """
             分割二维码
             :param handle_image:原图片
             :param file_name:图片文件名
             :return: 分割出来的图片
             """
    global decoderName
    qr_image = Image.fromarray(handle_image)  # 输入图像
    height, width = qr_image.size
    s = height * width  # 图像面ji
    decoder_data = decode_by_pyzbar(qr_image)
    decoderName = 'pyzbar'
    tmp_file = os.path.join(TEMP_PATH, file_name + '_temp.jpg')
    cv2.imwrite(tmp_file, handle_image)
    if decoder_data == '':
        decoder_data = decode_by_zxing(tmp_file)
        decoderName = 'zxing'
        # if decoder_data == '':

        # decoder_data = ali_qr(tmp_file)
        # decoderName = 'ali_qr'
    os.remove(tmp_file)
    if decoder_data != '':  # 返回结果
        return {
            "result": 0,
            "qrcode": decoder_data,
            "decoderName": decoderName
        }
    elif s <= 1000 or s >= 4000000:
        return {

            "result": 1,
            "qrcode": "",
        }
    else:
        return {
            "result": 2,
            "qrcode": "",
        }


def qr_main(_id, url):
    """
    主函数
    :param url:图片地址
    :return: 二维码的内容
    """
    file_name = qr_decode_name(url)
    image_url = url_conversion(url)
    # image_url = area_image(image_url)
    # image_length, image_width, image_dim = image_url.shape     # 获取图片的三维
    # image_url = image_url[0:int(image_length * 0.25), 0:int(image_width * 0.35), :]  # 对图片进行剪切

    image_length,image_width,image_dim = image_url.shape
    image_url = image_url[int(image_length*0.85):int(image_length),0:int(image_width*0.70),:]

    segmentate_image = image_segmentation(image_url)
    handle_image = handle_im(segmentate_image)
    qr_content = image_distinguish(handle_image, file_name)
    number1 = 1
    number2 = 1
    if qr_content['result'] == 2:
        while qr_content['result'] != 0:
            handle_image = image_handle(segmentate_image, number1, number2)
            qr_content = image_distinguish(handle_image, file_name)
            number1 += 1
            number2 += 1
            if number1 > 4:
                break
    if qr_content['result'] == 2:
        handle_image = image_handle(segmentate_image, 1, 2)
        qr_content = image_distinguish(handle_image, file_name)
    if qr_content['result'] == 2:
        handle_image = image_handle(segmentate_image, 2, 1)
        qr_content = image_distinguish(handle_image, file_name)
    # if qr_content['result'] == 2:
    #        image_template = cv2.imread('temo.jpg')
    #        image_merged = merged_image(segmentate_image, image_template)
    #        qr_content = image_distinguish(image_merged,file_name)
    if CONFIG.is_save_image:
        segmentation_name = file_name + "_seg.jpg"
        handle_name = file_name + "_handle.jpg"
        handle_error = file_name + "_handle_error.jpg"
        origin_name = file_name + "_origin.jpg"
        cv2.imwrite(os.path.join(TEMP_PATH, origin_name), image_url)
        cv2.imwrite(os.path.join(TEMP_PATH, segmentation_name), segmentate_image)
        if qr_content.get(0) == None:
            cv2.imwrite(os.path.join(TEMP_PATH, handle_error), handle_image)
        else:
            cv2.imwrite(os.path.join(TEMP_PATH, handle_name), handle_image)
    return qr_content

if __name__ == '__main__':
    # qr_main('','http://jfjun.img-cn-hangzhou.aliyuncs.com/resources/56cead9ba784f772607475d8/2016-09/7bb8b020dc012e01fc9a5c98c44fd958.jpg')
    qr_main('','http://jfjun.img-cn-hangzhou.aliyuncs.com/resources/59f7ccc941e8779b4b3bb53f/2018-07/35b5692351ca2acbdf7018ce3ef651ec.jpg')