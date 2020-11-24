import numpy as np
import tensorflow as tf
import os
from custom_loss import yolo_loss
from loss_tensorflow import my_yolo_loss_tf
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

from my_yolo_loss import *


S = (50,50)
B = 2
C = 1
batch_size = 1

loss = my_yolo_loss(1.0,0.5,1.0,5.0,S,B,C,batch_size)
yol_loss = yolo_loss(5,0.5,S,B,C,(448,448,3))
my_yolo_loss_fct =  my_yolo_loss_tf(5, 0.5, S=S, B=B, C=C)


y_pred = tf.convert_to_tensor(np.random.rand(batch_size,S[0],S[1],5*B+C))
y_true = tf.convert_to_tensor(np.random.rand(batch_size,S[0],S[1],5*B+C))
#loss  = loss(y_true,y_pred)
#otherloss = yol_loss(y_true,y_pred)
my_yolo_loss = my_yolo_loss_fct(y_true,y_pred)
my_yolo_loss_1 = my_yolo_loss_fct(y_true,y_true)
my_yolo_loss_2 = my_yolo_loss_fct(y_pred,y_pred)


#print('loss',loss)
print('otherloss',my_yolo_loss)

