import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET
import os
from enum import Enum
from operator import itemgetter
from keras.optimizers import Adam
import tensorflow.keras.backend as kb
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.text as text

# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#   tf.config.experimental.set_memory_growth(gpu, True)

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")

yolo_input = (448, 448, 3) 
S = (100,100)
B = 2
C = 1
batch_size = 1
classes = ['Blase']
classes_dic = {'Blase':0}
data_path='/data/Heytex/train' #relative to this file
data_path_test='/data/Heytex/test' #relative to this file

print(tf.version.VERSION)
model = tf.keras.models.load_model('kassenbonModel_10102020',
                                   custom_objects={'loss': custom_loss(lambda_c = 5, lambda_no=.5, S=(50,1), B=1, C=4)})
model.summary()

X_Test = getImagesAsTensor()
img = np.array(X_Test)/255
print(img.shape)
Y_Test = model.predict(img[None,:,:,:])
BoundingBoxes = outputToBoundingBox(Y_Test[0], img)
showImageWithBoundingBoxes(X_Test,BoundingBoxes)