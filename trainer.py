import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator


import utils

# 处理ASCII，返回的结果为{数字：0-9，}
def func(x): return x - 48 if x <= 57 else x - 87 if x <= 110 else x - 88  
# func(ord(x)) 0->0 a->10 z->35


def train(data, target, model_save):
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(40, 40)),         # 一维化 336
        keras.layers.Dense(40*40, activation=tf.nn.relu),   # 隐藏层
        keras.layers.Dropout(0.25),
        keras.layers.Dense(40*40, activation=tf.nn.relu),   # 隐藏层
        keras.layers.Dense(36, activation='softmax')        # 输出层 36个可能的值
    ])
    model.compile(optimizer='rmsprop',                      # 优化
                  loss='sparse_categorical_crossentropy',   # 损失函数
                  metrics=['accuracy'])                     # 用准确率衡量
    model.fit(data, target, batch_size=128, epochs=100)
    model.save(model_save)



# CUMT分法，长度为6（效果不好）
def split_pic(img):
    """img = Image.open('captcha.png')
    returns a dict using np.asarray()
    """
    img = img.convert('L').convert('1')
    x_size, y_size = img.size                      # 200 * 50
    piece = round((x_size-57)/12)                        # 12
    centers = [round(35+piece*(2*i+1)+2*i) for i in range(6)]

    ar = []
    for i, center in enumerate(centers):
        single_pic = img.crop((center-(piece+5), 10, center+(piece+5), y_size))   # 40 * 40 （8） ；34 * 40 （5）
        ar.append(np.asarray(single_pic, dtype='int8'))
    return ar



def _load_data(folder):
    "加载folder下的图片 返回图片numpy三维数组和其标记"
    count = 0
    imgs = os.listdir(folder)
    length = len(imgs)*6
    label = np.zeros(length, dtype="int8")
    # data = np.zeros((length, 40, 40,3), dtype="int8")
    data = np.zeros((length, 40, 40), dtype="int8")
    # * 分配三维空数组, data.shape = (length, 21, 16)

    img_count =0
    for img_name in imgs:
        img = cv2.imread('%s/%s' % (folder, img_name), cv2.COLOR_BGR2GRAY)
        (_, img) = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
        # 投影法分割
        single_imgs = utils.split_img_projection(img)
        if single_imgs == False:
            # print("【切割失败】",img_name)
            continue
        for i, single in enumerate(single_imgs):
            temp = single[:,:,0]
            for x in range(0, 40):               
                for y in range(0, 40):
                    temp[x,y] = single[x,y][0]
            data[count, :] = temp
            # ? 将(21*16)pixel的图片转成灰度图像数组,高40，宽40
            alpha = img_name.split('.')[0][i]  # 分割出单个元素的标签
            label[count] = func(ord(alpha))  #传入符号的ASCII
            count += 1
        img_count +=1
        # print(img_name,"加载成功！",img_count)
    print("共有图片：",len(imgs),"； 成功加载：",img_count)
    return data, label


if __name__ == "__main__":
    # model_file = './model/Model_tf.net'
    model_file = 'Model_tf.net'

    print('Training...')
    x_data, y_data = _load_data('./data/train/')
    train(x_data, y_data, model_file)

    print('\nTesting...')
    model = keras.models.load_model(model_file)
    x, y = _load_data('./data/test_sets/')
    model.evaluate(x, y) # 测试
