# here we call the train loop
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
from models.adabelief_optimizer import AdaBelief
from PostSegmentation.bbox_utils import CalculateAnchorsOfDataSet
from matplotlib import pyplot as plt

from models.loss_yolov1 import *
from Postprocess.yolo_output import postprocessing
from Postprocess.yolo_output_2 import yolo_net_out_to_car_boxes,draw_box
from adabelief_tf import AdaBeliefOptimizer
# from Loss.my_yolo_loss import my_yolo_loss

# calculated by KMEANS clustering of widht and height in all 








os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

yolo_input = (448, 448, 3) 
S = (25,25)
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
ANCHORS = CalculateAnchorsOfDataSet(Y_train,9)




create_train_txt(data_path=data_path_test, dirname = os.path.dirname(__file__),b_aug_data=False)
createAnnotationsTxt(classes=classes_dic, data_path=data_path_test, dirname = os.path.dirname(__file__))
data_set = create_dataset(data_path=data_path_test,dirname = os.path.dirname(__file__))
X_valid, Y_valid = createXYFromDataset(data_set)





# my_training_batch_generator = My_Custom_Generator(X_train, Y_train, batch_size,S,B,C,False,ANCHORS)
# my_validation_batch_generator = My_Custom_Generator(X_valid, Y_valid, batch_size,S,B,C,True,ANCHORS)


# model =  blasen_model(name, yolo_shape=yolo_input, S=S, B=B, C=C )
# model.compile(
#     loss=my_yolo_loss(1.0,0.5,1.0,5.0,S,B,C,yolo_input,batch_size),
#     optimizer=AdaBeliefOptimizer()
#     )
# model.build(input_shape=(batch_size,yolo_input[0], yolo_input[1], yolo_input[2]))
# model.summary()


# i = 0

# # X_train = X_train[:10]
# model.fit(x=my_training_batch_generator,
#           steps_per_epoch = int(len(X_train) // batch_size),
#           epochs = 10,
#           verbose = 1,
#           shuffle = True,
#           workers= 4,
#           validation_data = my_validation_batch_generator,
#           validation_steps = int(len(X_valid) // batch_size),
#            callbacks=[
#             #  CustomLearningRateScheduler(lr_schedule),
#             #   mcp_save,
#               #tensorboard_callback
#           ])
# print('training finished')

# for valid in X_valid:
#     image, y = my_validation_batch_generator.__getitem__(i)
#     y_strich = model.predict(image)
#     bbox =  yolo_net_out_to_car_boxes(y_strich[0], threshold = 0.17,C=C, B=2, S=S)
#     image = draw_box(bbox,image[0])
#     #image = postprocessing(y_strich,  valid,0.3, 0.5, yolo_input[0], yolo_input[1],classes,S,B,C,ANCHORS)
#     plt.imshow(image)
#     plt.show()
#     i += 1
