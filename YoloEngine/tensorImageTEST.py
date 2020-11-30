import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
from Preprocess.prepare_for_generator import create_train_txt, createAnnotationsTxt, create_dataset, createXYFromDataset
from models.custom_generator import My_Custom_Generator
from models.yolo_model import blasen_model
from models.yolov1_loss import my_yolo_loss_tf
from models.custom_image_callback import TensorBoardImage
from datetime import datetime
from adabelief_tf import AdaBeliefOptimizer
from models.custom_learningrate_scheduler import CustomLearningRateScheduler, lr_schedule
from models.lr_finder import MyLRFinder




yolo_input = (448, 448, 3) 
S = (50,50)
B = 2
C = 1
batch_size = 10


input1 = tf.convert_to_tensor(np.random.rand(batch_size,S[0],S[1],5*B + C))
input2 = tf.convert_to_tensor(np.random.rand(batch_size,S[0],S[1],5*B + C))

loss = my_yolo_loss_tf(5, 0.1, S,B,C)

myloss = loss(input1,input2)