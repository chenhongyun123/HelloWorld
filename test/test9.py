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
import json
from config import CONFIG
from config import ERROR

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import platform

TEMP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'temp'))
if not os.path.isdir(TEMP_PATH):
    os.mkdir(TEMP_PATH)

image_segmentation_erode = 25  # 图像分割腐蚀的次数
image_segmentation_dilate = 25  # 图像分割膨胀的次数
image_segmentation_blur = 17  # 图像分割均值滤波
DEBUG_LEVEL = 0


def get_md5_from_url(url):
    """
    得到图片名字
    :param url: 图片地址
    :return: 图片名字
    """
    ret = re.search(r'[^/]+(?=\.jpg)', url)
    return ret.group()


def download_image(url):
    """
    下载图片
    :param url: 图片地址
    :return: 原图片
    """
    url_obtain = urllib.request.urlopen(url)
    url_image = np.asarray(bytearray(url_obtain.read()), dtype="uint8")
    url_image = cv2.imdecode(url_image, cv2.IMREAD_COLOR)  # 将比特流转化为图片格式
    return url_image


def remove_blank(image):
    """
    去掉周边留白
    :param image: 原图片
    :return: 原图片去掉空白
    """
    if DEBUG_LEVEL:
        res = cv2.resize(image, (300, 500), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('image', res)
        cv2.waitKey(0)
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
        res1 = cv2.resize(image, (300, 500), interpolation=cv2.INTER_CUBIC)
        if DEBUG_LEVEL:
            cv2.imshow('res1', res1)
            cv2.waitKey(0)
        # length,width = image.shape[:2]
        # res = cv2.resize(image, (0.2 * width, 0.2 * length),
        #                  interpolation=cv2.INTER_CUBIC)
        # if DEBUG_LEVEL:
        #     cv2.imshow('image', res)
        #     cv2.waitKey(0)
    return image


def pre_crop_by_image_type(img):
    """
    根据业务类型对图像进行预裁剪，增值税专普票取左上角，增值税卷票取左下角
    :param img:图片地址
    :return: 二维码的内容
    """
    height, width, _ = img.shape
    if int(height) < int(width):
        img_pre_crop = img[0:int(height * 0.25), 0:int(width * 0.35), :]
        if DEBUG_LEVEL:
            cv2.imshow('image', img_pre_crop)
            cv2.waitKey(0)
    else:
        img_pre_crop = img[int(height * 0.75):int(height), 0:int(width * 0.40), :]
        if DEBUG_LEVEL:
            cv2.imshow('image', img_pre_crop)
            cv2.waitKey(0)
    return img_pre_crop


def crop_qr_box(img):
    """
    分割二维码
    :param url_image:原图片
    :return: 分割出来的图片
    """
    image_length, image_width, image_dimension = img.shape  # 判断输入的是rgb图像还是灰度图像
    if image_dimension == 3:
        blue_channel, green_channel, red_channel = cv2.split(img)  # 分离图像通道，取r通道的图像
        image_scale = red_channel
    else:
        image_scale = img
    if DEBUG_LEVEL:
        cv2.imshow('orig', img)
        cv2.imshow('red_channel', image_scale)
        cv2.waitKey(0)
    enhanceimage = cv2.createCLAHE(clipLimit=2, tileGridSize=(10, 10))  # 对每个小块进行均衡化，10*10小块
    image_scale = enhanceimage.apply(image_scale)
    if DEBUG_LEVEL:
        cv2.imshow('image_scale', image_scale)
        cv2.waitKey(0)
    image_filtering_x_direction = cv2.Sobel(image_scale, cv2.CV_16S, 1, 0)  # 使用sobel算子沿x、y方向进行滤波
    image_filtering_y_direction = cv2.Sobel(image_scale, cv2.CV_16S, 0, 1)
    image_subtract = cv2.subtract(image_filtering_x_direction, image_filtering_y_direction)
    image_format_conversion = cv2.convertScaleAbs(image_subtract)  # 转换格式，unit8
    image_blurred = cv2.blur(image_format_conversion,
                             (image_segmentation_blur, image_segmentation_blur))  # 进行均值滤波，模板大小是17*17
    (rew, thresh) = cv2.threshold(image_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if DEBUG_LEVEL:
        cv2.imshow('image_blurred', image_scale)
        cv2.imshow('thresh', thresh)
        cv2.waitKey(0)
    image_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, None)  # 闭运算消除二维码间的缝隙
    image_erode = cv2.erode(image_closed, None, iterations=image_segmentation_erode)  # 消除小斑点
    image_dilate = cv2.dilate(image_erode, None, iterations=image_segmentation_dilate)  # 剩余的像素扩张并重新增长回去
    image_canny = cv2.Canny(image_dilate, 150, 1000)  # canny边缘检测
    (rew, image_binary) = cv2.threshold(image_canny, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if DEBUG_LEVEL:
        cv2.imshow('image_closed', image_closed)
        cv2.waitKey(0)
        cv2.imshow('image_erode', image_erode)
        cv2.waitKey(0)
        cv2.imshow('image_dilate', image_dilate)
        cv2.waitKey(0)
        cv2.imshow('image_canny', image_canny)
        cv2.waitKey(0)
        cv2.imshow('image_binary', image_binary)
        cv2.waitKey(0)
    image_area, contours, hierarchy = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找出最大面积
    # if DEBUG_LEVEL:
    #     cv2.drawContours(img, contours, -1, (0, 255, 0), 3 )
    #     cv2.imshow('contours', img)
    #     cv2.waitKey(0)
    if len(contours) > 0:
        rect_list = []
        # 参考：[python opencv minAreaRect 生成最小外接矩形](https://blog.csdn.net/lanyuelvyun/article/details/76614872)
        for c in contours:
            rect = cv2.minAreaRect(c)  # 为每个轮廓生成最小外接矩形
            rect_list.append(rect)
        max_box = sorted(rect_list, key=lambda x: x[1][0] * x[1][1], reverse=True)[0]
        box1 = cv2.boxPoints(max_box)  # 输出图像 [ [x0,y0], [x1,y1], [x2,y2], [x3,y3] ]
        [x0, y0], [x1, y1], [x2, y2], [x3, y3] = np.int64(box1)  # float32转换为int64
        _x0, _y0 = min(x0, x1, x2, x3), min(y1, y2, y0, y3)  # ?????
        _x1, _y1 = max(x0, x1, x2, x3), max(y1, y2, y0, y3)  # ?????

        # 向边界拓展10个像素
        _x0, _y0 = max(0, _x0 - 15), max(0, _y0 - 15)
        _x1, _y1 = min(_x1 + 15, image_width - 1), min(_y1 + 15, image_length - 1)
        print("_y0,_y1,_x0,_x1=", _y0, _y1, _x0, _x1)
        img_qr = img[_y0:_y1, _x0:_x1, :]
    else:
        img_qr = img
    return img_qr


def optimize_qr_img(segmentation_image, dilate_iterations, erode_iterations):
    """
    二维码图片预处理
    :param Segmentation_image:分割出来的图片
    :return: 预处理后的图片
    """
    gray_image = cv2.cvtColor(segmentation_image, cv2.COLOR_BGR2GRAY)  # 灰度化图像
    enhanceimage = cv2.createCLAHE(clipLimit=2, tileGridSize=(10, 10))
    histogram_image = enhanceimage.apply(gray_image)
    (rew, binary_image) = cv2.threshold(histogram_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if DEBUG_LEVEL:
        cv2.imshow('histogram_image', histogram_image)
        cv2.imshow('binary_image', binary_image)
        cv2.waitKey(0)
    binary_image_height, binary_image_width = binary_image.shape
    # 放大两倍
    zoom_image = cv2.resize(binary_image, (2 * binary_image_width, 2 * binary_image_height),
                            interpolation=cv2.INTER_CUBIC)  # 立方插值 使放大后的图像平滑
    image_closed = cv2.GaussianBlur(zoom_image, (3, 3), 0)
    (rew, image_turn) = cv2.threshold(image_closed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # 图像反转
    image_dilate = cv2.dilate(image_turn, None, iterations=dilate_iterations)
    image_erode = cv2.erode(image_dilate, None, iterations=erode_iterations)
    (rew, image_back_binary) = cv2.threshold(image_erode, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    (rew, handle_image) = cv2.threshold(image_back_binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if DEBUG_LEVEL:
        cv2.imshow('zoom_image', zoom_image)
        cv2.waitKey(0)
        cv2.imshow('image_closed1', image_closed)
        cv2.waitKey(0)
        cv2.imshow('image_turn', image_turn)
        cv2.waitKey(0)
        cv2.imshow('image_dilate1', image_dilate)
        cv2.waitKey(0)
        cv2.imshow('image_erode1', image_erode)
        cv2.waitKey(0)
        cv2.imshow('image_back_binary', image_back_binary)
        cv2.waitKey(0)
        cv2.imshow('handle_image', handle_image)
        cv2.waitKey(0)
    return handle_image


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
    zxing_decoded_data = ''
    try:
        operating_system = platform.system()
        if operating_system == 'Windows':
            zxfile_home = "E:\download\zxing"
        else:
            zxfile_home = "/root/zxing"
        print('zxfile_home=', zxfile_home)
        reader = zxing.BarCodeReader(zxfile_home)  # 路径指向 zxing所git clone的位置
        barcode = reader.decode(decode_filename, True)
        if barcode is None:
            zxing_decoded_data = ''
        else:
            zxing_decoded_data = barcode.data
    except Exception as e:
        print('decode_by_zxing encounter exception', e)

    return zxing_decoded_data


def decode_qr(img_qr, file_name):
    """
             分割二维码
             :param handle_image:原图片
             :param file_name:图片文件名
             :return: 分割出来的图片
    """
    if img_qr is None:
        return {
            "result": ERROR.err_crop,
            "qrcode": "",
        }
    if DEBUG_LEVEL:
        cv2.imshow('decode_qr', img_qr)
        cv2.waitKey(0)
    qr_image = Image.fromarray(img_qr)  # 输入图像
    height, width = qr_image.size
    decoder_data = decode_by_pyzbar(qr_image)
    decoderName = 'pyzbar'
    if decoder_data == '':
        tmp_file = os.path.join(TEMP_PATH, file_name + '_temp.jpg')
        cv2.imwrite(tmp_file, img_qr)
        decoder_data = decode_by_zxing(tmp_file)
        decoderName = 'zxing'
        os.remove(tmp_file)

    if decoder_data != '':  # 返回结果
        return {
            "result": ERROR.success,
            "qrcode": decoder_data,
            "decoderName": decoderName
        }
    min_len = min(height, width)
    max_len = max(height, width)
    if min_len < 80 or max_len > 1000 or max_len / min_len > 2:
        print('err_crop', file_name, height, width)
        return {
            "result": ERROR.err_crop,
            "qrcode": "",
        }
    else:
        return {
            "result": ERROR.err_other,
            "qrcode": "",
        }


def qr_main(_id, url):
    """
    主函数
    :param url:图片地址
    :return: 二维码的内容
    """
    file_name = get_md5_from_url(url)
    img_orig = download_image(url)
    img_remove_blank = remove_blank(img_orig)
    img_pre_crop = pre_crop_by_image_type(img_remove_blank)

    img_qr = crop_qr_box(img_pre_crop)
    qr_content = decode_qr(img_qr, file_name)
    dilate_erode_iterations = 1
    img_qr_opt = None
    if qr_content['result'] == ERROR.err_other:
        while qr_content['result'] != ERROR.success and dilate_erode_iterations <= 4:
            img_qr_opt = optimize_qr_img(img_qr, dilate_erode_iterations, dilate_erode_iterations)
            if DEBUG_LEVEL:
                cv2.imshow('img_qr_opt1', img_qr_opt)
                cv2.waitKey(0)
            qr_content = decode_qr(img_qr_opt, file_name)
            dilate_erode_iterations += 1

    if qr_content['result'] == ERROR.err_other:
        img_qr_opt = optimize_qr_img(img_qr, 1, 2)
        if DEBUG_LEVEL:
            cv2.imshow('img_qr_opt2', img_qr_opt)
            cv2.waitKey(0)
        qr_content = decode_qr(img_qr_opt, file_name)
    if qr_content['result'] == ERROR.err_other:
        img_qr_opt = optimize_qr_img(img_qr, 2, 1)
        if DEBUG_LEVEL:
            cv2.imshow('img_qr_opt3', img_qr_opt)
            cv2.waitKey(0)
        qr_content = decode_qr(img_qr_opt, file_name)
    # if qr_content['result'] == 2:
    #        image_template = cv2.imread('temo.jpg')
    #        image_merged = merged_image(segmentate_image, image_template)
    #        qr_content = image_distinguish(image_merged,file_name)
    if CONFIG.is_save_image:
        code = qr_content['result']
        if code == ERROR.success:  # 改动
            cv2.imwrite(os.path.join(TEMP_PATH, '%s_orig_%s.jpg' % (file_name, code)), img_orig)
            cv2.imwrite(os.path.join(TEMP_PATH, '%s_qr_%s.jpg' % (file_name, code)), img_qr)
            if img_qr_opt is not None:
                cv2.imwrite(os.path.join(TEMP_PATH, '%s_qr_opt_%s.jpg' % (file_name, code)), img_qr)
    print('qr_content=', qr_content)
    return qr_content


if __name__ == '__main__':
    count = 0
    url = open('resource.txt', 'rt', encoding='utf-8')
    # # url = open('src(1).txt','rt',encoding = 'utf-8')
    # # url = open('src.txt','rt',encoding = 'utf-8')
    url_image = url.readlines()
    m = len(url_image)
    for i in range(m):
        qr_content = qr_main('', url_image[i])
        print(qr_content)
        if qr_content['result'] == 0:
            count += 1
            print(count)

    # url='http://jfjun.img-cn-hangzhou.aliyuncs.com/resources/593a5468e2a853fb4660169f/2018-06/e220707d8161b9aa0b266de8142822c3.jpg'
    # url = 'http://jfjun.img-cn-hangzhou.aliyuncs.com/resources/5a7ea51419876d8b295ba958/2018-01/685549ca9dc9f4959fb5f94168226387.jpg?x-oss-process=image/rotate,90'
    # url='http://jfjun.img-cn-hangzhou.aliyuncs.com/resources/5a65572f420f0e9a26172538/2018-02/c7aec6db82b281c2df0f32a14ebb0f7d.jpg'
    # url = 'http://jfjun.img-cn-hangzhou.aliyuncs.com/resources/59f7ccc941e8779b4b3bb53f/2018-07/35b5692351ca2acbdf7018ce3ef651ec.jpg'
    # qr_main('aa', url)
