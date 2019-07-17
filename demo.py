# 请确保在Windows系统下运行, 否则文件路径可能会报错
# 请确保tensorflow的版本不要过低, 最好为tf.__version__=='1.13.1'
# NOTE: 请保持当前项目文件的完整性, 项目包含运行所必须的代码及权重文件
# 请确保程序运行时已存在model文件夹, 文件夹内应包含已训练好人脸识别模型的权重参数文件


from detect import *
from cnn import *
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
tf.enable_eager_execution()
print("open camera...")

# build face detected model
P = PNet()
R = RNet()
O = ONet()
P.restore()
R.restore()
O.restore()
detector = Detect(P, R, O)
# build face classified model
classifier = creat_model()
classifier.load_weights('model\weight')
# 类别与人名的一一对应关系
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
class_to_name = dict(zip(name_to_class.values(), name_to_class.keys()))

print('if you want to exit the camera, please enter q')
cap = cv2.VideoCapture(0)
# 用循环不断获取当前帧 处理后显示出来
while True:
    # 捕获当前帧
    ret, img = cap.read()
    temp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    loc = detector.detect_face(temp, minsize=100)
    if type(loc) != type(None):
        # 多人脸检测
        for p in loc:
            cv2.rectangle(img, (p[0], p[1]), (p[2], p[3]), (255, 0, 0), 1)
            input_ = temp[p[1]: p[3], p[0]: p[2]]
            # 处理图像, 使其符合人脸识别模型的输入
            input_ = cv2.cvtColor(input_.copy(), cv2.COLOR_RGB2GRAY)
            input_ = cv2.resize(input_, (128, 128))
            input_ = np.expand_dims(input_, 0)
            input_ = np.expand_dims(input_, -1)
            # 预测人脸类别
            pred = classifier.predict(input_/255.0)
            idx = np.argmax(pred, axis=-1)
            # 设定阈值, 若置信度低于0.8就将该人脸标记为Unknown
            if pred[0, idx[0]] > 0.5:
                text = class_to_name[idx[0]]
            else:
                text = "Unknow"

            cv2.putText(img, text, (p[0], p[3]),
                        cv2.FONT_ITALIC, 2, (0, 0, 255), 0)
    # 显示图像
    cv2.imshow('Camera', img)

    # 结束帧捕获的条件
    # 等待10ms 即帧频为100fps
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
# 释放资源
cap.release()
cv2.destroyAllWindows()
