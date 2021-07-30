#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@File   :find_plate.py
@IDE    :PyCharm
@Author :gpwang
@Date   :2021/7/30
@Desc   :传统算法的车牌检测项目
=================================================='''
import cv2
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import sys, os, json, random

# ---------读取图片-------
maxLength = 700  # 最大的长度

minArea = 2000  # 最小的面积

imgOri = cv2.imread("plate.jpg")
# assert imgOri is None, "please check your path"
img = np.copy(imgOri)
h, w = img.shape[:2]


# ---------缩放图片到指定尺寸-----------
def zoom(w, h, wMax, hMax):
    """
    由于加载图片大小的差异，缩放到固定大小的
    最重要原因是方便后面的模糊，开，闭操作，可以用一个统一的内核大小处理不同的图片了
    缩放到指定的图片尺寸
    :param w: 图片的宽
    :param h: 图片的高
    :param wMax: 最大宽度
    :param hMax: 最小高度
    :return:
    """
    withScale = 1.0 * wMax / w
    heightScale = 1.0 * hMax / h
    scale = min(withScale, heightScale)
    resizeWidth = int(w * scale)
    resizeHeight = int(h * scale)
    return resizeWidth, resizeHeight


imgWidth, imgHeight = zoom(w, h, maxLength, maxLength)

img = cv2.resize(img, (imgWidth, imgHeight), interpolation=cv2.INTER_AREA)

cv2.namedWindow("resizeWindow", 0)
cv2.imshow("resizeWindow", img)
# cv2.namedWindow("origin", 0)
# cv2.imshow("origin", imgOri)

"""
        2.图片预处理
        图片预处理是最重要的，这里主要作用是为了有效的寻找外围轮廓的主要方法有
            1.高斯模糊降低噪声
            2.开操作和加权来强化对比度
            3.二值化和Canny边缘检测来寻找物体的轮廓
            4.先闭后开操作找到整块整块的矩形
"""
# 加高斯模糊和灰度化处理
img = cv2.GaussianBlur(img, (3, 3), 0)
imgGary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.namedWindow("imgGary", 0)
cv2.imshow("imgGary", imgGary)
# 开操作
kernel = np.ones((20, 20), np.uint8)
imgOpen = cv2.morphologyEx(imgGary, cv2.MORPH_OPEN, kernel)
cv2.namedWindow("imgOpen", 0)
cv2.imshow("imgOpen", imgOpen)
# 加权操作 增加对比度
imgOpenWeight = cv2.addWeighted(imgGary, 1, imgOpen, -1, 0)
cv2.namedWindow("imgOpenWeight", 0)
cv2.imshow("imgOpenWeight", imgOpenWeight)
# 二值化和Canny边缘检测来找到物体的轮廓
# 二值化
ret, imgBin = cv2.threshold(imgOpenWeight, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
cv2.namedWindow("imgBin", 0)
cv2.imshow("imgBin", imgBin)
# canny边缘检测
imgEdge = cv2.Canny(imgBin, 100, 200)
cv2.namedWindow("imgEdge", 0)
cv2.imshow("imgEdge", imgEdge)
##先闭后开操作便于找到整块整块的矩形
kernel = np.ones((10, 19), np.uint8)
imgEdge = cv2.morphologyEx(imgEdge, cv2.MORPH_CLOSE, kernel)
imgEdge = cv2.morphologyEx(imgEdge, cv2.MORPH_OPEN, kernel)
cv2.namedWindow("imgEdgeProcessed", 0)
cv2.imshow("imgEdgeProcessed", imgEdge)
"""
    3 寻找轮廓
        这里只要通过findContour接口就可以
"""
image, contours, hierarchy = cv2.findContours(imgEdge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = [cnt for cnt in contours if cv2.contourArea(cnt) > minArea]

"""
    4 删除一些物理尺寸不满足的轮廓
        通过minAreaRect找到他们对应的最小矩形。先通过宽、高比来删除一些不符合条件的
"""
carPlateList = []
imgDark = np.zeros(img.shape, dtype=img.dtype)
for index, contour in enumerate(contours):
    rect = cv2.minAreaRect(contour)  # [中心(x,y),(宽，高) 旋转角度]
    w, h = rect[1]
    if w < h:
        w, h = h, w
    scale = w / h
    if scale > 2 and scale < 4:
        color = (255, 255, 255)
        carPlateList.append(rect)
        cv2.drawContours(imgDark, contours, index, color, 1, 8)
        box = cv2.boxPoints(rect)  # 将rect转换为点坐标
        """
        函数cv2.minAreaRect()返回一个Box2D结构rect:(最小外接矩形的中心(x,y),
        (宽度，高度),旋转角度)，但是要绘制这个矩形需要矩形的4个顶点坐标box，通过函数
        cv2.boxPoints()获得，返回形式为[[x0,y0],[x1,y1],[x2,y2],[x3,y3]]得到
        外接矩形4个顶点顺序
        """
        box = np.int0(box)
        cv2.drawContours(imgDark, [box], 0, (0, 0, 255), 1)
cv2.namedWindow("imgGray", 0)
cv2.imshow("imgGray", imgDark)
"""
  5 重映射
    这里主要是做仿射变换，将偏角摆正
"""


def pointLimit(point, maxWidth, maxHeight):
    if point[0] < 0:
        point[0] = 0
    if point[0] > maxWidth:
        point[0] = maxWidth
    if point[1] < 0:
        point[1] = 0
    if point[1] > maxHeight:
        point[1] = maxHeight


imgPlatList = []
for index, carPlat in enumerate(carPlateList):
    if carPlat[2] > -1 and carPlat[2] < 1:
        angle = 1
    else:
        angle = carPlat[2]

    carPlat = (carPlat[0], (carPlat[1][0] + 5, carPlat[1][1] + 5), angle)
    box = cv2.boxPoints(carPlat)

    # Which point is Left/Right/Top/Bottom
    w, h = carPlat[1][0], carPlat[1][1]
    if w > h:
        LT = box[1]
        LB = box[0]
        RT = box[2]
        RB = box[3]
    else:
        LT = box[2]
        LB = box[1]
        RT = box[3]
        RB = box[0]

    for point in [LT, LB, RT, RB]:
        pointLimit(point, imgWidth, imgHeight)

    # Do warpAffine
    newLB = [LT[0], LB[1]]
    newRB = [RB[0], LB[1]]
    oldTriangle = np.float32([LT, LB, RB])
    newTriangle = np.float32([LT, newLB, newRB])
    warpMat = cv2.getAffineTransform(oldTriangle, newTriangle)
    imgAffine = cv2.warpAffine(img, warpMat, (imgWidth, imgHeight))
    cv2.imshow("imgAffine" + str(index), imgAffine)
    print("Index: ", index)

    imgPlat = imgAffine[int(LT[1]):int(newLB[1]), int(newLB[0]):int(newRB[0])]
    imgPlatList.append(imgPlat)
    cv2.namedWindow("imgPlat", 0)
    cv2.imshow("imgPlat" + str(index), imgPlat)


def accurate_place(imgHsv, limit1, limit2, color):
    rows, cols = imgHsv.shape[:2]
    left = cols
    right = 0
    top = rows
    bottom = 0

    # rowsLimit = 21
    rowsLimit = rows * 0.8 if color != "green" else rows * 0.5  # 绿色有渐变
    colsLimit = cols * 0.8 if color != "green" else cols * 0.5  # 绿色有渐变
    for row in range(rows):
        count = 0
        for col in range(cols):
            H = imgHsv.item(row, col, 0)
            S = imgHsv.item(row, col, 1)
            V = imgHsv.item(row, col, 2)
            if limit1 < H <= limit2 and 34 < S:  # and 46 < V:
                count += 1
        if count > colsLimit:
            if top > row:
                top = row
            if bottom < row:
                bottom = row
    for col in range(cols):
        count = 0
        for row in range(rows):
            H = imgHsv.item(row, col, 0)
            S = imgHsv.item(row, col, 1)
            V = imgHsv.item(row, col, 2)
            if limit1 < H <= limit2 and 34 < S:  # and 46 < V:
                count += 1
        if count > rowsLimit:
            if left > col:
                left = col
            if right < col:
                right = col
    return left, right, top, bottom


"""
6 定位车牌颜色
  基本思路就是把经过透视变换的图转换到HSV空间，然后通过统计全部像素的个数
  以及单个颜色对应的个数，如果满足蓝色占了全部像素的1/3及以上的时候，就认为
  这是一个蓝色车牌
"""
colorList = []
for index, imgPlat in enumerate(imgPlatList):
    green = yellow = blue = 0
    imgHsv = cv2.cvtColor(imgPlat, cv2.COLOR_BGR2HSV)
    rows, cols = imgHsv.shape[:2]
    imgSize = cols * rows
    color = None

    for row in range(rows):
        for col in range(cols):
            H = imgHsv.item(row, col, 0)
            S = imgHsv.item(row, col, 1)
            V = imgHsv.item(row, col, 2)
            if 11 < H <= 34 and S > 34:
                yellow += 1
            elif 35 < H <= 99 and S > 34:
                green += 1
            elif 99 < H <= 124 and S > 34:
                blue += 1
    limit1 = limit2 = 0
    if yellow * 3 >= imgSize:
        color = "yellow"
        limit1 = 11
        limit2 = 34
    elif green * 3 >= imgSize:
        color = "green"
        limit1 = 35
        limit2 = 99
    elif blue * 3 >= imgSize:
        color = "blue"
        limit1 = 100
        limit2 = 124

    print("Image Index[", index, '], Color：', color)
    colorList.append(color)
    print(blue, green, yellow, imgSize)
    if color is None:
        continue
    left, right, top, bottom = accurate_place(imgHsv, limit1, limit2, color)
    w = right - left
    h = bottom - top
    if left == right or top == bottom:
        continue

    scale = w / h
    if scale < 2 or scale > 4:
        continue

    needAccurate = False
    if top >= bottom:
        top = 0
        bottom = rows
        needAccurate = True
    if left >= right:
        left = 0
        right = cols
        needAccurate = True
    # imgPlat[index] = imgPlat[top:bottom, left:right] \
    # if color != "green" or top < (bottom - top) // 4 \
    # else imgPlat[top - (bottom - top) // 4:bottom, left:right]
    imgPlatList[index] = imgPlat[top:bottom, left:right]
    # cv2.namedWindow("")
    cv2.imshow("Vehicle Image " + str(index), imgPlatList[index])
"""
7 根据颜色重新裁剪，筛选图片
    知道车牌颜色后，可以通过逐行，逐列扫描，把车牌精确到更小的范围
    还可以通过宽高比剔除一些不正确的矩形，还可以得到精确唯一的车牌图像
    内容
"""

# Step7: Resize vehicle img.



cv2.waitKey()
pass
