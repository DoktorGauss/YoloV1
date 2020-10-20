import tensorflow as tf
import numpy as np


def load_image( infilename, target_size ):
    data = tf.keras.preprocessing.image.load_img(
    infilename, grayscale=False, color_mode='rgb', target_size=target_size,
    interpolation='nearest')

    return data

def getImagesAsTensor( image_folder_path='./data' ):
    test_path = image_folder_path + '/train'    
    test_X = []
    for filename in os.listdir(test_path):
        if not filename.endswith('0977.JPG'): continue
        test_X.append(load_image(test_path + '/' + filename))   
    return test_X[0]