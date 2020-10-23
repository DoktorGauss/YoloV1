# data augementing script
import PIL
from PIL import Image
import cv2
import numpy as np
from sklearn.cluster import KMeans
import math
from PostSegmentation.bbox_utils import * 
from PostSegmentation.data_aug import *
import tensorflow as tf
from matplotlib import pyplot as plt

def clusterImage_kb(image,center):
    imageasarray = np.asarray(image)
    reshapedIamge = np.reshape(imageasarray,(imageasarray.shape[1]*imageasarray.shape[0],3))
    return KMeans(n_clusters=len(center),max_iter=1,init=center).fit(reshapedIamge)

def getCurrentRotationOfKassenbon(image):
    thresh = 75
    fn = lambda x : 255 if x > thresh else 0
    bw_img = image.convert('L').point(fn, mode='1')
    #bw_img = image.convert('P')
    #bw_img = clusterImage_kb(image,[(0,0,0),(255,255,255)])
    data = np.array(bw_img, dtype=int)
#    plt.imshow(data)
#    plt.show()

    first_point = None
    second_point = None

    offset_step = int(data.shape[1] * 0.1)
    #define custom loop
    #first point from mid to 0 row
    b_end_first = False
    offset_y_first = 1
    while(not b_end_first):
        y_index = int(data.shape[1] / 2) - (offset_y_first*offset_step) 
        if y_index < 0 : return None
        for x in range(data.shape[0]):
            value = int(data[y_index,x])
            if value == 1:
                first_point = (x,y_index)
                b_end_first = True
                break
        offset_y_first += 1
    #second point from mid to end row


    b_end_second = False
    offset_y_second = 1
    while(not b_end_second):
        y_index = int(data.shape[1] / 2) + (offset_y_second*offset_step) 
        if y_index > data.shape[1] : return None
        for x in range(data.shape[0]):
            value = int(data[y_index,x])
            if value == 1:
                second_point = (x,y_index)
                b_end_second = True
                break
        offset_y_second += 1
    if (second_point[0]-first_point[0]) == 0 : return 90

    m = (second_point[1]-first_point[1])/(second_point[0]-first_point[0])
    angle_in_radians = math.atan(m)
    angle_in_degrees = math.degrees(angle_in_radians)
    return angle_in_degrees
        




def rotateImageAndLabelsToOrdinalLine(arrayimage,bboxes):
    arrayimage = arrayimage.astype(np.uint8)
    image = Image.fromarray(arrayimage)
    rotation = getCurrentRotationOfKassenbon(image)

    lostRotation = 90-rotation

    angle = lostRotation
    if lostRotation > 90: angle = -(180-lostRotation)

    w,h = arrayimage.shape[1], arrayimage.shape[0]
    cx, cy = w//2, h//2

    img = rotate_im(arrayimage, -angle)
    print('rotate image in ', -angle)
    img = np.asarray(img)


    corners = get_corners(bboxes)

    corners = np.hstack((corners, bboxes[:,4:]))


    corners[:,:8] = rotate_box(corners[:,:8], angle, cx, cy, h, w)

    new_bbox = get_enclosing_box(corners)


    scale_factor_x = img.shape[1] / w

    scale_factor_y = img.shape[0] / h


    #img = np.true_divide(img, 255) 

    resizedimage = cv2.resize(img, (w,h))
    resizedimage = np.asarray(resizedimage)
    new_bbox[:,:4] = np.true_divide(new_bbox[:,:4],[scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]) 

    bboxes  = new_bbox

    bboxes = clip_box(bboxes, [0,0,w, h], 0.25)

    return resizedimage, bboxes