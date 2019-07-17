# 请确保在Windows系统下运行, 否则文件路径可能会报错
# 请确保tensorflow的不要过低, 最好为tf.__version__=='1.13.1'
# 请确保程序运行时已存在model文件夹, 本程序不会自动创建
# 请确保当前目录有模型训练与测试时所需的数据文件, 最好为本人上传的数据文件
# 应确保有以下数据文件:
#   - train_images.npy
#   - test_images.npy
#   - train_labels.npy
#   - test_labels.npy

import numpy as np
import tensorflow as tf
import cv2
import os
import random
tf.enable_eager_execution()


def creat_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (5, 5), strides=(
            2, 2), padding='same', name='conv'),
        tf.keras.layers.Flatten(name='flatten'),
        tf.keras.layers.Dense(128, activation=tf.nn.relu, name='dense'),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax, name='output')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


"""
# NOTE: 模型训练代码段
# 导入数据
train_images = np.load('train_images.npy')
test_images = np.load('test_images.npy')
train_images = train_images/255.0
test_images = test_images/255.0
train_labels = np.load('train_labels.npy')
test_labels = np.load('test_labels.npy')
# 创建模型并进行训练
model = creat_model()
model.fit(train_images, train_labels, epochs=5)
# 评估模型
print("\ncalulate the test acc.\n")
test_loss, test_acc = model.evaluate(test_images, test_labels)

model.save_weights('model\weight', save_format='tf')

"""

"""
# NOTE: 模型测试代码段
# 导入数据
test_images = np.load('test_images.npy')
test_images = test_images/255.0
test_labels = np.load('test_labels.npy')
# 创建模型并预测
model = creat_model()
model.load_weights('model\weight')
pred = model.predict_classes(test_images)
acc = np.mean(pred == test_labels)
print("test acc: ", acc)
"""
