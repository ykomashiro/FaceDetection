# 请确保在Windows系统下运行
# 请确保faceImages在当前项目目录下
#
# NOTE: 当前项目文件夹中包含本人实现mtcnn的代码及其权重参数, 故请保持当前项目文件的完整性
# TODO: 本代码文件用于读取faceImages中的图片, 使用MTCNN算法
# 检测人脸并将其转成灰度图储存在faceImageGray文件夹下
#
from detect import *
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cv2
tf.enable_eager_execution()


def filesearch(fn):
    g = os.walk(fn)
    files = list()
    for path, _, file_list in g:
        for file in file_list:
            files.append(os.path.join(path, file))
    return files


# 创建检测器并导入模型
P = PNet()
R = RNet()
O = ONet()
P.restore()
R.restore()
O.restore()
detector = Detect(P, R, O)

root = "faceImages\\"
grayroot = "faceImageGray\\"
dirs = os.listdir(root)
for dir_ in dirs:
    tmppath = grayroot+"\\"+dir_
    if not os.path.exists(tmppath):
        os.makedirs(tmppath)
filepaths = filesearch(root)
for path in tqdm(filepaths):
    dirname = path.split('\\')[-2]
    filename = path.split('\\')[-1]
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    loc = detector.detect_face(image, minsize=100)
    if type(loc) == type(None):
        continue
    img = image[loc[0][1]: loc[0][3], loc[0][0]: loc[0][2]]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imgpath = grayroot+"\\"+dirname+"\\"+filename
    cv2.imwrite(imgpath, img)
