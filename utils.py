
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2


"""
投影法划分
"""
def get_borders_projection(img):
    height, width = img.shape[:2]
    v = [0]*width
    a = 0
    #垂直投影
    for x in range(0, width):               
        for y in range(0, height):
    #         if closed[y,x][0] == 0:
            if img[y,x][0] == 0:
                a = a + 1
            else :
                continue
        v[x] = a
        a = 0

    x_borders = []
    current = width - 2
    # 默认干扰线宽度限
    default_line_limit = 8
    # 默认上一个切割点的干扰线宽度
    last_line_width = 4
    while(1):
        if current<5:
            break
        if v[current]==2 and v[current-3]>2:
            # print(current,v[current])
            x_borders.append(current)
            current-=20
        if current<width-5 and v[current+4]-v[current]>=5  and abs(v[current-7]-v[current])>=5 and v[current]<default_line_limit:
            # print(current,v[current-10],v[current],v[current+2],default_line_limit,last_line_width)
            x_borders.append(current)
            default_line_limit += (v[current] - last_line_width)
            last_line_width = v[current]
            # print(v[current-6:current+5])
    #         移动过两字符间的干扰线段
            while(v[current]<default_line_limit and abs(v[current-1]-v[current])<2):
                current-=1
    #             print(current)
            current-=20
    #     寻找最后一个分割线，①已经有6条线了；②此处刚经历下降；③
        if len(x_borders)==6 and v[current+4]-v[current]>5:
            # print(current,v[current-10],v[current],v[current+2])
            x_borders.append(current)
            break
        current-=1
    
    if len(x_borders)==7:
        return x_borders
    else:
        # print("borders",x_borders,len(x_borders))
        return False

def split_img_projection(img):
    borders = get_borders_projection(img)
    if borders:
        single_imgs = list()
        for i in range(6):
            single_img = img[9:49,borders[i+1]:borders[i]]
            w = single_img.shape[1]
            if w<=40:
                rest = 40 - w
                left = round(rest/2)
                right = rest - left
                single_img = cv2.copyMakeBorder(single_img, top=0, bottom=0, left=left, right=right, borderType= cv2.BORDER_CONSTANT, value=[255,255,255])
            else:
                single_img=single_img[:,0:40]
            single_imgs.append(single_img)
        single_imgs.reverse()
            
        return single_imgs
    else:
        return False
