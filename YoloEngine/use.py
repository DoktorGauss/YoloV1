import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime


from models.yolo_model import blasen_model
from Preprocess.xmlReader import readXML_Train_Test
from Preprocess.yoloOutputFormat import convert_data_into_YOLO
from Preprocess.imageReader import readImages_Train_Test
# from models.custom_loss import *
from models.custom_metrics import true_positive_caller
from models.custom_callback import CustomCallback,LossAndErrorPrintingCallback
from models.custom_learningrate_scheduler import CustomLearningRateScheduler,lr_schedule
from Preprocess.prepare_for_generator import *
from models.custom_generator import *
from matplotlib import pyplot as plt


from Postprocess.yolo_v1_output_to_bndbox import yolo_v1_draw_calculation



os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

yolo_input = (1024, 768, 3) 
S = (50,50)
B = 2
C = 1
batch_size = 1
classes = ['Blase']
classes_dic = {'Blase':0}
data_path='/data/Heytex/train' #relative to this file
data_path_test='/data/Heytex/test' #relative to this file
weights_path = os.path.join(os.path.dirname(__file__),'blasen20201106-095120.hdf5')

name='blasen'+ datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = "logs/scalars/" + 'blasen' + datetime.now().strftime("%Y%m%d-%H%M%S")







create_train_txt(data_path=data_path_test, dirname = os.path.dirname(__file__),b_aug_data=False)
createAnnotationsTxt(classes=classes_dic, data_path=data_path_test, dirname = os.path.dirname(__file__))
data_set = create_dataset(data_path=data_path_test,dirname = os.path.dirname(__file__))
X_valid, Y_valid = createXYFromDataset(data_set)



my_validation_batch_generator = My_Custom_Generator(X_valid, Y_valid, batch_size,S,B,C,True,None)


model =  blasen_model(name, yolo_shape=yolo_input, S=S, B=B, C=C )

model.build(input_shape=(batch_size,yolo_input[0], yolo_input[1], yolo_input[2]))
model.load_weights(weights_path)

model.summary()

x,y = my_validation_batch_generator.__getitem__(0)
y_pred = model.predict(x)
yolo_v1_draw_calculation(x,y_pred,S,B,C,yolo_input)
