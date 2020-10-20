# data augementing script
import PIL
from PIL import Image
import cv2
import numpy as np
from sklearn.cluster import KMeans
import math

def clusterImage_kb(image,center):
    reshapedIamge = image.reshape((image.shape[1]*image.shape[0],3))
    return KMeans(n_clusters=len(center),max_iter=0,init=center).fit(reshapedIamge)

def getCurrentRotationOfKassenbon(image):
    thresh = 200
    fn = lambda x : 255 if x > thresh else 0
    bw_img = image.convert('L').point(fn, mode='1')
    data = np.asarray(bw_img)
    first_point = None
    second_point = None
    for x in range(bw_img.shape[0]):
        y = 0.2*bw_img.shape[1]
        if(data[y][x] == 255):
            first_point = (x,y)

    for x in range(bw_img.shape[0]):
        y = 0.8*bw_img.shape[1]
        if(data[y][x] == 255):
            second_point = (x,y)
    m = (second_point[1]-first_point[1])/(second_point[0]-first_point[0])
    angle_in_radians = math.atan(m)
    angle_in_degrees = math.degrees(angle_in_radians)
    return angle_in_degrees




def rotateImageAndLabelsToOrdinalLine(image,labels):
    img = Image.fromarray(image)
    rotation = getCurrentRotationOfKassenbon(img)
    lostRotation = 90-rotation
    img.rotate(lostRotation)
    return img


def rotateBoundingBoxes(labels,rotation):
    cos = math.cos(rotation)
    sin = math.sin(rotation)

    for label in labels:
        h = label['height']
        w = label['width']
        nW = int((h*sin)+(w*cos))
        nH = int((h*cos)+(w*sin))
        


