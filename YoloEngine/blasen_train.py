# here we call the train loop
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime


from models.yolo_model import blasen_model
from Preprocess.xmlReader import readXML_Train_Test
from Preprocess.yoloOutputFormat import convert_data_into_YOLO
from Preprocess.imageReader import readImages_Train_Test
from models.custom_loss import *
from models.custom_metrics import true_positive_caller
from models.custom_callback import CustomCallback,LossAndErrorPrintingCallback
from models.custom_learningrate_scheduler import CustomLearningRateScheduler,lr_schedule
from Preprocess.prepare_for_generator import *
from models.custom_generator import *
from models.adabelief_optimizer import AdaBelief







# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#   tf.config.experimental.set_memory_growth(gpu, True)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

yolo_input = (448, 448, 3) 
S = (100,100)
B = 2
C = 1
batch_size = 1
classes = ['Blase']
classes_dic = {'Blase':0}
data_path='/data/Heytex/train' #relative to this file
data_path_test='/data/Heytex/test' #relative to this file

name='blasen'+ datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = "logs/scalars/" + 'blasen' + datetime.now().strftime("%Y%m%d-%H%M%S")




tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, write_graph=True, write_images=True,update_freq='batch')
mcp_save = ModelCheckpoint('weights_' + name +'.hdf5', save_best_only=True, monitor='loss', mode='min')


create_train_txt(data_path=data_path, dirname = os.path.dirname(__file__),b_aug_data=True)
createAnnotationsTxt(classes=classes_dic, data_path=data_path, dirname = os.path.dirname(__file__))
data_set = create_dataset(data_path=data_path,dirname = os.path.dirname(__file__))
X_train, Y_train = createXYFromDataset(data_set)


create_train_txt(data_path=data_path_test, dirname = os.path.dirname(__file__),b_aug_data=False)
createAnnotationsTxt(classes=classes_dic, data_path=data_path_test, dirname = os.path.dirname(__file__))
data_set = create_dataset(data_path=data_path_test,dirname = os.path.dirname(__file__))
X_valid, Y_valid = createXYFromDataset(data_set)

my_training_batch_generator = My_Custom_Generator(X_train, Y_train, batch_size,S,B,C,True)
my_validation_batch_generator = My_Custom_Generator(X_valid, Y_valid, batch_size,S,B,C,True)




model =  blasen_model(name, yolo_shape=yolo_input, S=(100,100), B=2, C=1 )
model.compile(
    loss=yolo_loss(lambda_c=5, 
                               lambda_no=.5, 
                               S=S, 
                               B=B, 
                               C=C),
    optimizer=AdaBelief(),
    metrics=[
        box_loss(lambda_c=5, 
                lambda_no=.5, 
                S=S, 
                B=B, 
                C=C),
        confidence_loss(lambda_c=5, 
                lambda_no=.5, 
                S=S, 
                B=B, 
                C=C),
        class_loss(lambda_c=5, 
                lambda_no=.5, 
                S=S, 
                B=B, 
                C=C)
    ])
model.summary()

# defining a function to save the weights of best model
print('FITTING STARTED')
model.fit(x=my_training_batch_generator,
          steps_per_epoch = int(len(X_train) // batch_size),
          epochs = 135,
          verbose = 1,
          shuffle = True,
          workers= 4,
          validation_data = my_validation_batch_generator,
          validation_steps = int(len(X_valid) // batch_size),
           callbacks=[
              CustomLearningRateScheduler(lr_schedule),
            #   mcp_save,
            #   tensorboard_callback
          ])
print('training finished')
model.save(name)
print('model saved as ' + name)
# model.evaluate(tf_test_X,tf_test_Y)
print('evaluation finished')