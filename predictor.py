from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2

import utils


def analyse(stream, model) -> dict:
    """stream = open(file, 'rb').read()
    先载入保存的model, 传入model对象来进行分析, 
    model = keras.models.load_model('../model/m.net')
    """
    def func(x): return x + 48 if x <= 9 else x + 87 if x <= 23 else x + 88
    # image = Image.open(BytesIO(stream))
    image = cv2.imdecode(np.frombuffer(stream, np.uint8), cv2.COLOR_BGR2GRAY)
    (_, image) = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
    # 先转化为灰度图像,  再转化为[0|1]值图像
    result = []
    data = np.zeros((1, 40, 40), dtype="int8")
    # 投影法分割
    single_imgs = utils.split_img_projection(np.array(image))
    if single_imgs == False:
        return False
    for single in single_imgs:
        temp = single[:,:,0]
        for x in range(0, 40):               
            for y in range(0, 40):
                temp[x,y] = single[x,y][0]
        data[0] = temp
        answer = model.predict(data)       # 此时answer->[36] 每个值都是介于0~1间 总和为1
        answer = np.argmax(answer)         # 找出预测最有信心的
        result.append(chr(func(answer)))   # 将这个值对应转换成character
    return result


if __name__ == '__main__':
    # model_file = './model/Model_tf.net'
    model_file = 'Model_tf.net'

    # Train and test see `trainer.py`
    print('Predicting...')
    model = keras.models.load_model(model_file)
    import os
    count_acc = 0
    count_split_fail = 0
    count_char_acc = 0
    for file_name in os.listdir('predict'):  # 遍历目录文件预测
        with open('./predict/'+ file_name, 'rb') as f:
            r = analyse(f.read(), model)
            if not r:
                print("【切割失败】",file_name)
                count_split_fail+=1
                continue
            count = 0
            for i in range(6):
                if list(file_name)[i] == r [i]:
                    count_char_acc+=1
                    count+=1
            mark = False
            if file_name[:6] == ''.join(r):
                mark = True
                count_acc +=1
            print(file_name, ':', r, round(count/6,2), mark)
    print("【共预测图片张数】：",len(os.listdir('predict')))
    print("【切割失败】：",count_split_fail)
    print("【切割成功率】：",1-count_split_fail/len(os.listdir('predict')))
    print("【图片成功率】：",count_acc/len(os.listdir('predict')))
    print("【图片准确率】：",count_acc/(len(os.listdir('predict'))-count_split_fail))
    print("【字符准确率】：",count_char_acc/((len(os.listdir('predict'))-count_split_fail)*6))
