

from Preprocess.batch_generator import *
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
from PostSegmentation.bbox_utils import CalculateAnchorsOfDataSet

yolo_input = (448, 448, 3) 
S = (100,100)
B = 2
C = 1
batch_size = 1
classes = ['Blase']
classes_dic = {'Blase':0}
data_path='/data/Heytex/train' #relative to this file
data_path_test='/data/Heytex/test' #relative to this file

IMAGE_H, IMAGE_W = yolo_input[1], yolo_input[0]
GRID_W = IMAGE_W / S[0]
GRID_H = IMAGE_H / S[1]
BATCH_SIZE       = 1
TRUE_BOX_BUFFER  = 50


create_train_txt(data_path=data_path, dirname = os.path.dirname(__file__),b_aug_data=True)
createAnnotationsTxt(classes=classes_dic, data_path=data_path, dirname = os.path.dirname(__file__))
data_set = create_dataset(data_path=data_path,dirname = os.path.dirname(__file__))
X_train, Y_train = createXYFromDataset(data_set)
ANCHORS = CalculateAnchorsOfDataSet(Y_train,4)






generator_config = {
    'IMAGE_H'         : IMAGE_H, 
    'IMAGE_W'         : IMAGE_W,
    'GRID_H'          : GRID_H,  
    'GRID_W'          : GRID_W,
    'LABELS'          : classes,
    'ANCHORS'         : ANCHORS[0],
    'BATCH_SIZE'      : BATCH_SIZE,
    'TRUE_BOX_BUFFER' : TRUE_BOX_BUFFER,
}


train_batch_generator = SimpleBatchGenerator(X_train, generator_config,
                                             norm=True, shuffle=True)


[x_batch,b_batch],y_batch = train_batch_generator.__getitem__(idx=3)