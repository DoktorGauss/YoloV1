import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime


from models.yolo_model import kassenbon_model_3
from Preprocess.xmlReader import readXML_Train_Test
from Preprocess.yoloOutputFormat import convert_data_into_YOLO
from Preprocess.imageReader import readImages_Train_Test
from models.custom_loss import *
from models.custom_metrics import true_positive_caller
from models.custom_callback import CustomCallback,LossAndErrorPrintingCallback
from models.custom_learningrate_scheduler import CustomLearningRateScheduler,lr_schedule
from Preprocess.prepare_for_generator import *
from models.custom_generator import *
from PostSegmentation.data_aug import RandomHorizontalFlip,RandomScale,RandomRotate,RandomShear,RandomHSV,RandomTranslate, Sequence
from PostSegmentation.bbox_utils import draw_rect,create_labimg_xml
import matplotlib.pyplot as plt
from PIL import Image



# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")

yolo_input = (1024, 768, 3) 

S = (100,100)
B = 2
C = 1
batch_size = 1
classes = ['Blase']
classes_dic = {'Blase':0}
data_path='/data/train' #relative to this file
name='blasen'+ datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = "logs/scalars/" + 'blasen' + datetime.now().strftime("%Y%m%d-%H%M%S")


create_train_txt(data_path=data_path, dirname = os.path.dirname(__file__),b_aug_data=False)
createAnnotationsTxt(classes=classes_dic, data_path=data_path, dirname = os.path.dirname(__file__))
data_set = create_dataset(data_path=data_path,dirname = os.path.dirname(__file__))
X, Y = createXYFromDataset(data_set)

my_generator = My_Custom_Generator(X, Y, batch_size,S,B,C,yolo_input,False)


for i in range(len(X)):
    x_path_complete = X[i]
    x_path = x_path_complete.rpartition('/')[0]
    x_name = os.path.splitext(os.path.basename(x_path_complete))[0]
    x_type = os.path.splitext(os.path.basename(x_path_complete))[1]
    augementation_path = '/augementation/'
    #image  = cv.imread(x_path_complete)

    x, y = my_generator.__getitem__(i)
    
    y_bndbxo = scaleBndBoxes(Y[i], (1,1,1),(1,1,1),float)
    
    #start augementing sequence
    horizontalFlip = RandomHorizontalFlip(0.5)
    randomScale = RandomScale(0.4)
    randomRotate = RandomRotate(40)
    shear = RandomShear(0.2)
    randomTranslate =RandomTranslate(0.3)
    randomHSV = RandomHSV(10,30,20)
    sequence = Sequence([horizontalFlip,randomRotate,randomTranslate,randomHSV])
    for i in range(10):
        aug_x, aug_y = sequence(x[0].copy(), np.reshape(y_bndbxo.copy(),(-1,5)))
        perm_image_path = x_path + augementation_path + x_name+'_permuted_'+ str(i)+x_type
        perm_image_pil = Image.fromarray(aug_x)
        perm_image_pil.save(perm_image_path)
        create_labimg_xml(perm_image_path,aug_y,classes_dic,x_name+'_permuted_'+str(i))
        print(x_name + '_permuted_' + str(i) + '.jpg(.xml) wurde gespeichert')
        # if i == 0: 
        #     fig, (ax1, ax2) = plt.subplots(1,2)
        #     ax1.imshow(draw_rect(aug_x,aug_y))
        #     ax2.imshow(draw_rect(x[0],np.reshape(y_bndbxo.copy(),(-1,5))))            
        #     plt.show()

