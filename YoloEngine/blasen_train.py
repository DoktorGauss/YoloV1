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
from models.lr_finder import MyLRFinder
tf.keras.backend.set_floatx('float32')


yolo_input = (448, 448, 3) 
S = (50,50)
B = 2
C = 1
batch_size = 24
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


create_train_txt(data_path=data_path, dirname = '',b_aug_data=True)
createAnnotationsTxt(classes=classes_dic, data_path=data_path, dirname = '' )
data_set = create_dataset(data_path=data_path,dirname = '')
X_train, Y_train = createXYFromDataset(data_set)


create_train_txt(data_path=data_path_test, dirname = '',b_aug_data=False)
createAnnotationsTxt(classes=classes_dic, data_path=data_path_test, dirname = '')
data_set = create_dataset(data_path=data_path_test,dirname = '')
X_valid, Y_valid = createXYFromDataset(data_set)

my_training_batch_generator = My_Custom_Generator(X_train, Y_train, batch_size,S,B,C, yolo_input,True)
my_validation_batch_generator = My_Custom_Generator(X_valid, Y_valid, batch_size,S,B,C, yolo_input,True)




model =  blasen_model(name, yolo_shape=yolo_input, S=S, B=B, C=C)
model.compile(
    loss=my_yolo_loss_tf(lambda_c = 5, lambda_no=.5, S=S, B=B, C=C),
    optimizer = AdaBeliefOptimizer(),
    #optimizer = 'adam',
    metrics = [
               tf.keras.metrics.AUC(),
    ]
    )
model.build(input_shape=(batch_size,yolo_input[0], yolo_input[1], yolo_input[2]))
# load best last model
#model.load_weights(checkpointPath)
#model = tf.keras.models.load_model(checkpointPath,custom_objects={'LeakyReLU':  tf.keras.layers.LeakyReLU(alpha=0.1)})
model.summary()


#load_ext tensorboard
#%tensorboard --logdir logs
tbi_callback = TensorBoardImage(tag=name,validationset=my_validation_batch_generator,classes=classes,S=S,B=B,C=C,imageshape=yolo_input)

print('FITTING STARTED')
model.fit(x=my_training_batch_generator,
          steps_per_epoch = int(len(X_train) // batch_size),
          epochs = 270,
          verbose = 1,
          shuffle = True,
          workers= 4,
          validation_data = my_validation_batch_generator,
          validation_steps = int(len(X_valid) // batch_size),
           callbacks=[
               CustomLearningRateScheduler(lr_schedule),
               mcp_save,
               tensorboard_callback,
               tbi_callback
          ])
print('training finished')
model.save(data_path_save_model+'/'+name)
print('model saved as ' + name)