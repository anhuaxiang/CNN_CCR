# coding:utf-8
from __future__ import print_function
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import pickle
import os
import cv2
import random
import numpy as np
import shutil


# 查找字体的最小包含矩形
def FindImageBox(img):
    height = img.shape[0]
    width = img.shape[1]
    v_sum = np.sum(img, axis=0)
    h_sum = np.sum(img, axis=1)
    left = 0
    right = width - 1
    top = 0
    low = height - 1
    # 从左往右扫描，遇到非零像素点就以此为字体的左边界
    for i in range(width):
        if v_sum[i] < v_sum[0]:
            left = i
            break
    # 从右往左扫描，遇到非零像素点就以此为字体的右边界
    for i in range(width - 1, -1, -1):
        if v_sum[i] < v_sum[0]:
            right = i
            break
    # 从上往下扫描，遇到非零像素点就以此为字体的上边界
    for i in range(height):
        if h_sum[i] < h_sum[0]:
            top = i
            break
    # 从下往上扫描，遇到非零像素点就以此为字体的下边界
    for i in range(height - 1, -1, -1):
        if h_sum[i] < h_sum[0]:
            low = i
            break
    return left, top, right, low


# 生成字体图像
class Font2Image(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def do(self, font_path, each_char):
        # 黑色背景
        img = Image.new("RGB", (self.width, self.height), "white")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, int(self.width * 0.7))
        # 白色字体
        draw.text((5, 5), each_char, (0, 0, 0), font=font)
        data = list(img.getdata())
        np_img = np.asarray(data, dtype='uint8')
        np_img = np_img[:, 0]
        np_img = np_img.reshape((self.height, self.width))
        left, upper, right, lower = FindImageBox(np_img)
        np_img = np_img[upper-3: lower + 1+3, left-3: right + 1+3]

        return np_img


# 注意，chinese_labels里面的映射关系是：（ID：汉字）
def get_label_dict():
    f = open('./chinese_labels', 'rb')
    label = pickle.load(f)
    f.close()
    return label


if __name__ == "__main__":
    out_dir = os.path.expanduser('./dataset_auto')
    font_dir = os.path.expanduser('./chinese_fonts')
    test_ratio = 0.2
    width = 64
    height = 64
    train_image_dir_name = "train"
    test_image_dir_name = "test"

    # 将dataset分为train和test两个文件夹分别存储
    train_images_dir = os.path.join(out_dir, train_image_dir_name)
    test_images_dir = os.path.join(out_dir, test_image_dir_name)

    if os.path.isdir(train_images_dir):
        shutil.rmtree(train_images_dir)
    os.makedirs(train_images_dir)

    if os.path.isdir(test_images_dir):
        shutil.rmtree(test_images_dir)
    os.makedirs(test_images_dir)

    # 将汉字的label读入，得到（ID：汉字）的映射表label_dict
    label_dict = get_label_dict()

    char_list = []  # 汉字列表
    value_list = []  # label列表
    for (value, chars) in label_dict.items():
        print(value, chars)
        char_list.append(chars)
        value_list.append(value)

    # 合并成新的映射关系表：（汉字：ID）
    lang_chars = dict(zip(char_list, value_list))

    verified_font_paths = []
    # 找到字体文件
    for font_name in os.listdir(font_dir):
        path_font_file = os.path.join(font_dir, font_name)
        verified_font_paths.append(path_font_file)

    font2image = Font2Image(width, height)

    for (char, value) in lang_chars.items():  # 外层循环是字
        image_list = []
        print(char, value)
        for j, verified_font_path in enumerate(verified_font_paths):  # 内层循环是字体
            image = font2image.do(verified_font_path, char)
            image_list.append(image)

        test_num = len(image_list) * test_ratio
        random.shuffle(image_list)  # 图像列表打乱
        count = 0
        for i in range(len(image_list)):
            img = image_list[i]

            # 生成训练，测试文件的文件夹
            # if count < test_num:
            #     char_dir = os.path.join(test_images_dir, "%0.5d" % value)
            # else:
            #     char_dir = os.path.join(train_images_dir, "%0.5d" % value)

            char_dir = char_dir = os.path.join(test_images_dir, "%0.5d" % value)

            if not os.path.isdir(char_dir):
                os.makedirs(char_dir)

            path_image = os.path.join(char_dir, "%d.png" % count)
            cv2.imwrite(path_image, img)
            count += 1
