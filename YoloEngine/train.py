# here we call the train loop
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime


from yolo_model import kassenbon_model_3
from xmlReader import readXML_Train_Test
from yoloOutputFormat import convert_data_into_YOLO
from imageReader import readImages_Train_Test
from custom_loss import custom_loss
from custom_metrics import true_positive_caller
from custom_callback import CustomCallback,LossAndErrorPrintingCallback
from custom_learningrate_scheduler import CustomLearningRateScheduler,lr_schedule


# yolo input shape (w,h,c)
## the image size of yolo input 
yolo_input = (448, 448, 3) 
S = (50,1)
B = 2
C = 4
classes = ['UnternehmenHauptData','BonHauptData','BonPosition','Zahlung']

name='kassenbon_'+ datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, write_graph=True, write_images=True,update_freq='batch')


model = kassenbon_model_3(name,yolo_shape=yolo_input, S=S, B=B, C=C)
model.compile(loss=custom_loss(lambda_c=5, 
                               lambda_no=.5, 
                               S=S, 
                               B=B, 
                               C=C),
    optimizer='adam',run_eagerly=True, metrics=["mae",true_positive_caller(S=(50,1), B=1, C=4)])
model.summary()
print('summary ended')



trainLabelData , testLabelData = readXML_Train_Test()
trainImageData, testImageData = readImages_Train_Test()

yolo_train_data_y, yolo_train_data_x = convert_data_into_YOLO(trainLabelData,trainImageData,classes,yolo_input,S,B,C,'train')
yolo_test_data_y, yolo_test_data_x = convert_data_into_YOLO(testLabelData,testImageData,classes,yolo_input,S,B,C, 'test')

# free memory
del trainLabelData
del testLabelData

print('start converting to tensorflow tensors')
tf_train_Y = tf.convert_to_tensor(yolo_train_data_y)
print('yolo_train_data_y -> tensor finished')
tf_train_X = tf.convert_to_tensor(yolo_train_data_x)
print('yolo_train_data_x -> tensor finished')
tf_test_Y =  tf.convert_to_tensor(yolo_test_data_y)
print('yolo_test_data_y -> tensor finished')
tf_test_X =  tf.convert_to_tensor(yolo_test_data_x)
print('yolo_test_data_x -> tensor finished')


# free memory
del yolo_train_data_y
del yolo_train_data_x
del yolo_test_data_y
del yolo_test_data_x


# defining a function to save the weights of best model
mcp_save = ModelCheckpoint('weights_' + name +'.hdf5', save_best_only=True, monitor='loss', mode='min')
print('FITTING STARTED')
model.fit(x=tf_train_X, y=tf_train_Y,
          batch_size=1,
          epochs = 20,
          verbose = 1,
          shuffle = True,
          workers= 4,
           callbacks=[
              CustomLearningRateScheduler(lr_schedule),
              mcp_save,
              tensorboard_callback,
              #CustomCallback(),
              #LossAndErrorPrintingCallback()
          ])
print('training finished')
model.save(name)
print('model saved as ' + name)
model.evaluate(tf_test_X,tf_test_Y)
print('evaluation finished')