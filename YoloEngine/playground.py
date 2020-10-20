
import numpy as np
from custom_loss import yolo_loss
import tensorflow as tf
import matplotlib.pyplot as plt    # for plotting the images
from yoloOutputFormat import convert_data_into_YOLO
from imageReader import readImageByPath
import sys, getopt
import xml.etree.ElementTree as ET
import os


S = (50,1)
C = 4
B = 2
inputShape = (448,448,3)


classes = ['UnternehmenHauptData','BonHauptData','BonPosition','Zahlung']

def readXML( image_folder_path='./data' ):
    train_path = image_folder_path + '/testLossFunction'
    # for each file in files 
    train_xml_data = []
    for filename in os.listdir(train_path):
        if not filename.endswith('.xml'): continue
        fullname = os.path.join(train_path, filename)
        tree = ET.parse(fullname)
        root = tree.getroot()
        xml_datas = []
        xml_metric = {}
        xml_metric['name'] = filename
        for child in root:
            if child.tag == 'size':
                for metric in child:
                    xml_metric[metric.tag] = metric.text
            if not child.tag == ('object'): continue
            child_xml_data = {}
            for childs in child:
                #if not myClasses(childs.tag): continue
                if not (childs.tag == 'name' or childs.tag == 'bndbox'): continue
                if childs.tag == 'name':
                    child_xml_data[childs.tag] = childs.text
                if childs.tag == 'bndbox':
                    for positions in childs:
                        child_xml_data[positions.tag] = positions.text
            xml_datas.append(child_xml_data)
        train_xml_data.append({'metric' : xml_metric, 'data' : xml_datas})
    return train_xml_data


loss = yolo_loss(lambda_c = 5, lambda_no=.5, S=S, B=B, C=C, inputShape = inputShape)

random_numbers_actual = tf.convert_to_tensor(np.random.rand(1,S[0],S[1],B*5+C))

data = readXML()
images = readImageByPath()

data, images = convert_data_into_YOLO(label_data = data, #nparray of label data
    image_data = images, # np array of image data
    classes = classes,
    inputShape=inputShape,  #inputshape of image
    S=S, # segmentation
    B=B, # number of prediction in each segment
    C=C, # number of classes we try to predict in each segment
    message='')

true_prediction = data[0]
false_prediction = data[1]
false_prediction_2 = data[2]
random_prediction = random_numbers_actual
    
print('true prediction', loss(true_prediction,true_prediction))
print('false prediction', loss(true_prediction,false_prediction))
print('random prediction', loss(true_prediction,random_prediction))
print('false prediction 2', loss(true_prediction,false_prediction_2))
