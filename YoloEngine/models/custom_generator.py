
from tensorflow import keras
import sys, getopt
import xml.etree.ElementTree as ET
import os
import cv2 as cv
import numpy as np

class My_Custom_Generator(keras.utils.Sequence) :
  
  def __init__(self, images, labels, batch_size, S=(50,1), B=2, C=4, normalize=True) :
    self.images = images
    self.labels = labels
    self.batch_size = batch_size
    self.S = S
    self.B = B
    self.C = C
    self.normalize = normalize
    
    
  def __len__(self) :
    return (np.ceil(len(self.images) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx) :
    batch_x = self.images[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

    train_image = []
    train_label = []

    for i in range(0, len(batch_x)):
      img_path = batch_x[i]
      label = batch_y[i]
      image, label_matrix = read(img_path, label, self.S, self.B, self.C, self.normalize)
      train_image.append(image)
      train_label.append(label_matrix)
    return np.array(train_image), np.array(train_label)

def resizexy(data, imageShape=(940,1280,3),yoloShape = (448,448,3)):
    xScale = yoloShape[0]/imageShape[1]
    yScale = yoloShape[1]/imageShape[0]
    # save the current values as old values
    xmin = int(xScale *data[0])
    ymin = int(yScale *data[1])
    xmax = int(xScale *data[2])
    ymax = int(yScale *data[3])
    return xmin, ymin, xmax,ymax
    
def read(image_path, label, S=(50,1), B=2, C=4,  normalize=True):
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    imageShape = image.shape[0:2]
    #image = cv.resize(image, (yoloShape[0], yoloShape[1]))
    image_h, image_w = image.shape[0:2]
    if normalize : image = image / 255.

    label_matrix = np.zeros([S[0], S[1], 5*B + C])
    for l in label:
        l = l.split(',')
        l = np.array(l, dtype=np.int)
        xmin = int(l[0])
        ymin = int(l[1])
        xmax = int(l[2])
        ymax = int(l[3])
        cls = l[4]
        x = (xmin + xmax) / 2 / image_w
        y = (ymin + ymax) / 2 / image_h
        w = (xmax - xmin) / image_w
        h = (ymax - ymin) / image_h
        loc = [S[1] * x, S[0] * y]
        loc_i = int(loc[1])
        loc_j = int(loc[0])
        y = loc[1] - loc_i
        x = loc[0] - loc_j

        if label_matrix[loc_i, loc_j, C+4] == 0:
          label_matrix[loc_i, loc_j, cls] = 1
          label_matrix[loc_i, loc_j, C:C+4] = [x, y, w, h]
          label_matrix[loc_i, loc_j, C+B*4] = 1  # response

    return image, label_matrix