# coding: utf-8
import cv2
import numpy as np
import os


iterations = 0
border = 2
char = 8

name = 0
'''
    按行分割
'''
img = cv2.imread('./test.png', cv2.COLOR_BGR2GRAY)

# 二值化
(_, thresh) = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

# 扩大黑色面积，使效果更明显
closed = cv2.erode(thresh, None, iterations=iterations)

# 高度，宽度
height, width = closed.shape[:2]

# 每一行的投影
shadow_y = [0] * height

# 统计每一行的黑点数
black_point_num = 0
emptyImage_y = np.zeros((height, width, 3), np.uint8)
for y in range(0, height):
    for x in range(0, width):
        if closed[y, x][0] == 0:
            black_point_num += 1
        else:
            continue
    shadow_y[y] = black_point_num
    black_point_num = 0

# 裁剪图片
start_index = 0
end_index = 0
in_block = False
for i in range(len(shadow_y)):
    if not in_block and shadow_y[i] > 0:
        in_block = True
        start_index = i
    elif shadow_y[i] == 0 and in_block:
        end_index = i
        in_block = False
        new_image = img[start_index-border:end_index+border, 0:width - 1]
        name_d = ("%0.5d" % name)
        cv2.imwrite('./cut_image/' + str(name_d) + '.png', new_image)
        name += 1


'''
    按字分割
'''
name = 0
for file_name in os.listdir('./cut_image'):

    img = cv2.imread('./cut_image/' + file_name, cv2.COLOR_RGB2GRAY)

    # 二值化
    (_, thresh) = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)

    # 扩大黑色面积，使效果更明显
    closed = cv2.erode(thresh, None, iterations=iterations)

    # 高度，宽度
    height, width = closed.shape[:2]

    # 每一列的投影
    shadow_x = [0] * width

    # 统计每一列的黑点数
    black_point_num = 0
    emptyImage_x = np.zeros((height, width, 3), np.uint8)
    for x in range(0, width):
        for y in range(0, height):
            if closed[y, x][0] == 0:
                black_point_num += 1
            else:
                continue
        shadow_x[x] = black_point_num
        black_point_num = 0

    # 裁剪图片
    start_index = 0
    end_index = 0
    in_block = False
    for i in range(len(shadow_x)):
        if not in_block and shadow_x[i] > 0:
            in_block = True
            start_index = i
        elif shadow_x[i] == 0 and in_block and shadow_x[i+1] == 0:
            end_index = i
            in_block = False

            # 剔除标点符号
            if end_index - start_index > char:
                new_image = img[0:height - 1, start_index-border:end_index+border]
                name_d = ("%0.5d" % name)
                cv2.imwrite('./temp/' + str(name_d) + '.png', new_image)
                name += 1

for file_name in os.listdir('./cut_image'):
    os.remove('./cut_image/' + file_name)
