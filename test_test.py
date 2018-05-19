import numpy as np
import pickle
import os
#
# a = np.array([[3, 3, 3], [6, 6, 7], [1, 1, 4]])
# b = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
#
# c = a * b
# print(sum(sum(c)))
# print(c)

#
# def get_label_dict():
#     f = open('./chinese_labels', 'rb')
#     label = pickle.load(f)
#     f.close()
#     return label
#
# label_dict = get_label_dict()
# for (value, chars) in label_dict.items():
#     print(value, chars)

font_dir = os.path.expanduser('./chinese_fonts')
verified_font_paths = []
for font_name in os.listdir(font_dir):
    path_font_file = os.path.join(font_dir, font_name)
    verified_font_paths.append(path_font_file)
for j, verified_font_path in enumerate(verified_font_paths):
    print(j, verified_font_path)

# img_path = []
# dir = './dataset_auto/test'
# for root, dir, files in os.walk(dir):
#     # print(root, dir, files)
#     img_path += [os.path.join(root, f) for f in files]
# # print(img_path)
# for x in img_path:
#     print(x)
# accuracy = 0.015625
# print("# 498.0 with loss 8.12747")
# print("# 499.0 with loss 8.12823")
# print("# 500.0 with loss 8.12711")
# print("# " + "500.0" + " 准确率: %.8f" % accuracy)
# print("# 501.0 with loss 8.12805")
# print("# 502.0 with loss 8.12836")