import sys, getopt
import xml.etree.ElementTree as ET
import os
import numpy as np

def create_train_txt(data_path='/data/train',  dirname = os.path.dirname(__file__),b_aug_data=False):
      absolute_path = dirname + data_path
      with open(os.path.join(absolute_path,'train.txt'), 'w') as f:
        for filename in os.listdir(absolute_path):
            if not filename.endswith('.xml'): continue
            fullname = os.path.join(absolute_path,filename)
            tree = ET.parse(fullname)
            root = tree.getroot()
            id = root.find('filename')
            f.write(id.text)
            f.write('\n')
            print(fullname, '  finished')
        f.close()
      
      if b_aug_data:
            aug_absolute_path = absolute_path + '/augementation'
            with open(os.path.join(absolute_path,'train.txt'), 'a') as f:
                  for filename in os.listdir(aug_absolute_path):
                        if not filename.endswith('.xml'): continue
                        fullname = os.path.join(aug_absolute_path,filename)
                        tree = ET.parse(fullname)
                        root = tree.getroot()
                        id = root.find('filename')
                        f.write('augementation/' + id.text)
                        f.write('\n')
                        print(fullname, ' finished')
                  f.close()


def createAnnotationsTxt(classes,data_path='/data/train',dirname = os.path.dirname(__file__)):
      absolute_path = dirname + data_path
      with open(os.path.join(absolute_path,'train.txt'), 'r') as f:
        image_ids = f.read().strip().split()
      with open(os.path.join(absolute_path,'train_image_annotation.txt'), 'w') as f:
        for image_id in image_ids:
              f.write(absolute_path + '/%s.JPG' % (image_id))
              convert_annotation(image_id,f,absolute_path,classes)
              f.write('\n')
              print(image_id, ' converted')

def convert_annotation(image_id, f,dirname,classes_num):
      in_file = dirname + '/%s.xml' % (image_id)
      tree = ET.parse(in_file)
      root = tree.getroot()
      for obj in root.iter('object'):
          difficult = obj.find('difficult').text
          cls = obj.find('name').text
          classes = list(classes_num.keys())
          if cls not in classes or int(difficult) == 1:
              continue
          cls_id = classes.index(cls)
          xmlbox = obj.find('bndbox')
          b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
                int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
          f.write(' ' + ','.join([str(a) for a in b]) + ',' + str(cls_id))

  



def create_dataset(data_path='/data/train',dirname = os.path.dirname(__file__)):
      dataset = []
      absolute_path = dirname + data_path
      with open(os.path.join(absolute_path,'train_image_annotation.txt'), 'r') as f:
        dataset = dataset + f.readlines()
      return dataset

def createXYFromDataset(data_set):
      X = []
      Y = []
      for item in data_set:
          item = item.replace("\n", "").split(" ")
          X.append(item[0])
          arr = []
          for i in range(1, len(item)):
                arr.append(item[i])
          Y.append(arr)
      return X,Y


def scaleBndBoxes(bbndBox, imageShape, yoloShape):
      bbndBox = np.array(bbndBox)
      bbndBox_int = []
      for box in bbndBox:
            box = box.split(',')
            box = np.array(box)
            box = box.astype(int)
            bbndBox_int.append(box)
      bbndBox_int = np.asarray(bbndBox_int)
      xScale = yoloShape[0]/imageShape[1]
      yScale = yoloShape[1]/imageShape[0]
      bbndBox_int[:,0] = np.int_(bbndBox_int[:,0] * xScale)
      bbndBox_int[:,1] = np.int_(bbndBox_int[:,1] * yScale)
      bbndBox_int[:,2] = np.int_(bbndBox_int[:,2] * xScale)
      bbndBox_int[:,3] = np.int_(bbndBox_int[:,3] * yScale)
      return bbndBox_int
            


# train_datasets = []
# val_datasets = []

# with open(os.path.join("data/VOCdevkit", '2007_train.txt'), 'r') as f:
#     train_datasets = train_datasets + f.readlines()
# with open(os.path.join("data/VOCdevkit", '2007_val.txt'), 'r') as f:
#     val_datasets = val_datasets + f.readlines()

# X_train = []
# Y_train = []

# X_val = []
# Y_val = []

# for item in train_datasets:
#   item = item.replace("\n", "").split(" ")
#   X_train.append(item[0])
#   arr = []
#   for i in range(1, len(item)):
#     arr.append(item[i])
#   Y_train.append(arr)

# for item in val_datasets:
#   item = item.replace("\n", "").split(" ")
#   X_val.append(item[0])
#   arr = []
#   for i in range(1, len(item)):
#     arr.append(item[i])
#   Y_val.append(arr)