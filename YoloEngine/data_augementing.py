from Preprocess.prepare_for_generator import *
from models.custom_generator import *
from PostSegmentation.bbox_utils import *
from PostSegmentation.rotate_to_ordinal_line import rotateImageAndLabelsToOrdinalLine
from PostSegmentation.permute_bbndbox import *
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from PIL import Image



classes = {'UnternehmenHauptData':0,'BonHauptData':1,'BonPosition':2,'Zahlung':3}
S=(50,1) 
B=2 
C=4
yoloShape = (448,448,3)

create_train_txt(data_path='/data/train', dirname = os.path.dirname(__file__))
createAnnotationsTxt(classes=classes, data_path='/data/train', dirname = os.path.dirname(__file__))
data_set = create_dataset(data_path='/data/train',dirname = os.path.dirname(__file__))
X, Y = createXYFromDataset(data_set)
my_generator = My_Custom_Generator(X, Y, 1,S,B,C,yoloShape,False)


for i in range(len(X)):
    x_path_complete = X[i]
    x_path = x_path_complete.rpartition('/')[0]
    x_name = os.path.splitext(os.path.basename(x_path_complete))[0]
    x_type = os.path.splitext(os.path.basename(x_path_complete))[1]
    augementation_path = '/augementation/'
    y = np.asarray(Y[i])

    image  = cv.imread(x_path_complete)
    x_train, y_train = my_generator.__getitem__(i)
    y = scaleBndBoxes(y,image.shape,yoloShape)
    # STEP 1 rotate image to ordinale save image as x_name_ordinal.JPG or x_name_ordinal.xml
    ordinal_image, ordinal_bbox = rotateImageAndLabelsToOrdinalLine(np.reshape(x_train,yoloShape), y)
    ordinal_image = Image.fromarray(ordinal_image)
    ordinal_image_savepath = x_path + augementation_path + x_name+'_ordinal'+x_type
    ordinal_image.save(ordinal_image_savepath)
    ordinal_bbox[:,4:] = y[:,4:]
    create_labimg_xml(ordinal_image_savepath, ordinal_bbox,classes,x_name+'_ordinal')
    # STEP 2 real augementating save image as {x_name}_aug_{i}.JPG or {x_name}_aug_{i}.xml
    #permute bonposition=2
    number_of_possible_permutation, candidats, iou_matrix = calculate_possible_permutation(ordinal_image, ordinal_bbox)
    
    permutation_images = permute_image_by_permutation_matrix(np.array(ordinal_image), ordinal_bbox,iou_matrix)
    # for num_of_permutation in range(1,number_of_possible_permutation):
    #     perm = np.random.permutation(candidats)
    #     perm_image = permute_image(ordinal_image, candidats, perm)


 
