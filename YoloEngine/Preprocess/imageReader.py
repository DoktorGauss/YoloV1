import os
import PIL
from PIL import Image
from numpy import asarray


def readImages_Train_Test(image_folder_path='./data'):
    train_path = image_folder_path + '/train'
    test_path = image_folder_path + '/test'
    train_image_data = readImage(train_path)
    test_image_data = readImage(test_path)
    return train_image_data, test_image_data

def readImage(path):
    imageData = []
    for filename in os.listdir(path):
        if not (filename.endswith('.jpg') or filename.endswith('.JPG')): continue
        fullname = os.path.join(path, filename)
        image = Image.open(fullname)
        data = asarray(image)
        imageData.append(data)
    print( path + 'data image readed')
    return imageData

def readImageByPath(path='./data/testLossFunction'):
    return readImage(path)