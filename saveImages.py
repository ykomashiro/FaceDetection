# 请确保在Windows系统下运行, 否则文件路径可能会报错
# 请确保faceImageGray在当前项目目录下
#
# TODO: 本代码文件用于读取faceImageGray中的图片, 并将其resize
# 128*128大小的图片, 然后保存为ndarray格式
#
# labels存储的数据为class编号, 本代码文件中可以找到其与人名的对应关系
import os
import random

import numpy as np

import cv2


def filesearch(fn):
    g = os.walk(fn)
    files = list()
    for path, _, file_list in g:
        for file in file_list:
            files.append(os.path.join(path, file))
    return files


name_to_class = {'huajinqing': 0,
                 'liangchunfu': 1,
                 'lijunyu': 2,
                 'linjuncheng': 3,
                 'linweixin': 4,
                 'wujiasheng': 5,
                 'xuhaolin': 6,
                 'zenglingqi': 7,
                 'zhouyuanxiang': 8,
                 'zhushichao': 9}

root = "faceImageGray\\"
files = filesearch(root)
random.shuffle(files)

images = np.zeros((len(files), 128, 128), dtype=np.uint8)
labels = np.zeros((len(files)), dtype=np.uint8)

for i, path in enumerate(files):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    name = path.split('\\')[-2]
    label = name_to_class[name]
    images[i] = image
    labels[i] = label
    if (i+1) % 500 == 0 or (i+1) == len(files):
        print("{}/{}".format(i+1, len(files)))


images = np.expand_dims(images, -1)  # (N,128,128,1)

train_images = images[:4800]
test_images = images[4800:]

train_labels = labels[:4800]
test_labels = labels[4800:]

np.save("train_images.npy", train_images)
np.save("test_images.npy", test_images)

np.save("train_labels.npy", train_labels)
np.save("test_labels.npy", test_labels)
