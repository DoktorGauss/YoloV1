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

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")


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