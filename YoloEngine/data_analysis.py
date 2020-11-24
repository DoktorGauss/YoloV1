from sklearn import datasets
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
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
from Postprocess.yolo_output import *
from PostSegmentation.bbox_utils import CalculateAnchorsOfDataSet


yolo_input = (1024, 768, 3) 
S = (100,100)
B = 2
C = 1
batch_size = 1
classes = ['Blase']
classes_dic = {'Blase':0}
data_path='/data/Heytex/train' #relative to this file
name='blasen'+ datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = "logs/scalars/" + 'blasen' + datetime.now().strftime("%Y%m%d-%H%M%S")


create_train_txt(data_path=data_path, dirname = os.path.dirname(__file__),b_aug_data=False)
createAnnotationsTxt(classes=classes_dic, data_path=data_path, dirname = os.path.dirname(__file__))
data_set = create_dataset(data_path=data_path,dirname = os.path.dirname(__file__))
X, Y = createXYFromDataset(data_set)
ANCHORS = CalculateAnchorsOfDataSet(Y,9)
print(ANCHORS)





# my_generator = My_Custom_Generator(X, Y, batch_size,S,B,C,False)

# strategies = ['uniform', 'quantile', 'kmeans']
# xdata = []
# ydata = []
# for i in range(len(X)):
#     x_path_complete = X[i]
#     x_path = x_path_complete.rpartition('/')[0]
#     x_name = os.path.splitext(os.path.basename(x_path_complete))[0]
#     x_type = os.path.splitext(os.path.basename(x_path_complete))[1]
#     augementation_path = '/augementation/'
#     #image  = cv.imread(x_path_complete)

#     x, y = my_generator.__getitem__(i)
#     y = np.random.rand(1,S[0],S[1],5*B+C)
#     bndboxes = decode_netout(y[0],1,yolo_input,S=S, B =B, C=C, obj_threshold=0.3, nms_threshold=0.3)
#     im = draw_boxes(x[0],bndboxes,classes)
#     plt.imshow(im)
#     plt.show()
    

# Y_array = []
# for image in Y:
#     for label in image:
#         Y_array.append(label)
# df = pd.DataFrame(Y_array,columns=['xmin', 'ymin', 'xmax', 'ymax','class'])
# df['width'] = df['xmax'] - df['xmin']
# df['height'] = df['ymax'] - df['ymin']
# df['ratio'] = df.apply(lambda row: row.height / row.width, axis=1)
# print(df.head())


# # ax1 = df.plot.scatter(x='height',

# #                       y='width',

# #                       c='DarkBlue')
# # ax2 = df.hist(column='ratio',bins=20)
# # plt.show()
# # df.to_csv('blasen.csv',index=False,sep = ';')

# # initialize KMeans object specifying the number of desired clusters
# n=4
# kmeans = KMeans(n_clusters=n)
# # learning the clustering from the input date

# data = np.array([df['width'].to_numpy(), df['height'].to_numpy()])
# data = np.reshape(data,(-1,2))
# kmeans = kmeans.fit(data)
# # output the labels for the input data
# print(kmeans.labels_)
# print(kmeans.cluster_centers_)
# df['label'] = kmeans.labels_
# print(df.head())
# LABEL_COLOR_MAP = {0 : 'r',
#                    1 : 'k',
#                    2 : 'b',
#                    3 : 'y',
#                    }

# label_color = [LABEL_COLOR_MAP[l] for l in df['label'] ]

# ax1 = df.plot.scatter(x='width',y='height',c=colorsr)
# # g = sns.clustermap(data)
# plt.show()