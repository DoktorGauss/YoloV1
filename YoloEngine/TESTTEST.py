# # Only 2 lines will be added
# # Rest of the flow and code remains the same as default keras
# import plaidml.keras
# plaidml.keras.install_backend()
# import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

# # first lets import the useful stuff
# import tensorflow as tf
# import keras

# # import other stuff
# from keras import backend as K
# import numpy as np
# from models.custom_loss import my_yolo_loss_tf


# # yolo_input = (448, 448, 3) 
# # S = (50,50)
# # B = 2
# # C = 1
# # batch_size = 10


# # input1 = tf.convert_to_tensor(np.random.rand(batch_size,S[0],S[1],5*B + C))
# # input2 = tf.convert_to_tensor(np.random.rand(batch_size,S[0],S[1],5*B + C))



# # ones = tf.convert_to_tensor(np.ones((batch_size,S[0],S[1],5*B + C)))
# # zeros = tf.convert_to_tensor(np.zeros((batch_size,S[0],S[1],5*B + C)))



# # loss = my_yolo_loss_tf(5,0.5,S,B,C)
# # myloss = loss(input1, input2)
# # onesloss = loss(input1, ones)
# # zerosloss = loss(input1, zeros)




import plaidml.keras
import os
import tensorflow as tf
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
tf.keras.backend.set_floatx('float32')


os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
plaidml.keras.install_backend()


yolo_input = (448, 448, 3) 
S = (50,50)
B = 2
C = 1
batch_size = 2
classes = ['Blase']
classes_dic = {'Blase':0}
data_path='./data/train' #relative to this file
gdrive_data_path=""
data_path_save_model="./data" #relative to this file
data_path_test='./data/valid' #relative to this file
checkpointPath="./data/model/blasen_698_448_50_50_20201119-095001.hdf5"

name='blasen_'+ str(yolo_input[0]) + '_' + str(yolo_input[1]) + '_'+ str(S[0]) + '_' + str(S[1]) + '_' + datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = "logs/scalars/" + 'blasen' + datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1,profile_batch=5, write_graph=True, write_images=True,update_freq='batch')
mcp_save = ModelCheckpoint(data_path_save_model + '/model/' + name +'.hdf5', save_best_only=True,save_weights_only=False, monitor='loss', mode='min')


create_train_txt(data_path=data_path, dirname = '',b_aug_data=False)
createAnnotationsTxt(classes=classes_dic, data_path=data_path, dirname = '' )
data_set = create_dataset(data_path=data_path,dirname = '')
X_train, Y_train = createXYFromDataset(data_set)


create_train_txt(data_path=data_path_test, dirname = '',b_aug_data=False)
createAnnotationsTxt(classes=classes_dic, data_path=data_path_test, dirname = '')
data_set = create_dataset(data_path=data_path_test,dirname = '')
X_valid, Y_valid = createXYFromDataset(data_set)





my_training_batch_generator = My_Custom_Generator(X_train, Y_train, batch_size,S,B,C, yolo_input,True)
my_validation_batch_generator = My_Custom_Generator(X_valid, Y_valid, batch_size,S,B,C, yolo_input,True)

my_training_batch_generator.create_json()
my_validation_batch_generator.create_json()