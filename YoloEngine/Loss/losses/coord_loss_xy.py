import tensorflow as tf
import numpy as np
from indicator.indicator import *

def coord_loss(lambda_coord,y_true, y_pred, S,B,C):
    def loss_coord():
        return lambda_coord*sum_loss_coord()

    def sum_loss_coord():
        SS = S[0] * S[1]
        sum = tf.convert_to_tensor(0.0,dtype='float64')
        for i in range (SS):
            for b in range(B):
                sum = tf.add(sum,loss_coord_obj(i,b))
        return sum

    def loss_coord_obj(i,b):
        indicator = my_indicator(y_true,y_pred,S,B,C)
        x_coord_true = get_x_true(i)
        x_coord_pred = get_x_pred(i,b)
        y_coord_true = get_y_true(i)
        y_coord_pred = get_y_pred(i,b)

        if indicator(1,i,b) == True:
            return loss_xy(x_coord_true,x_coord_pred,y_coord_true,y_coord_pred)
        else: 
            return tf.convert_to_tensor(0.0,dtype='float64')
    def get_x_true(i):
        i_th = int(i % S[0])
        j_th = int(int(i / S[0]) % S[1])
        return y_true[i_th,j_th,C]

    def get_x_pred(i,b):
        i_th = int(i % S[0])
        j_th = int(int(i / S[0]) % S[1])
        return y_pred[i_th,j_th,C+b*4]
        
    def get_y_pred(i,b):
        i_th = int(i % S[0])
        j_th = int(int(i / S[0]) % S[1])
        return y_pred[i_th,j_th,C+b*4+1]

    def get_y_true(i):
        i_th = int(i % S[0])
        j_th = int(int(i / S[0]) % S[1])
        return y_true[i_th,j_th,C+1]

    def loss_xy(x_coord_true,x_coord_pred,y_coord_true,y_coord_pred):
        x_diff = tf.pow(tf.subtract(x_coord_true,x_coord_pred),2) 
        y_diff = tf.pow(tf.subtract(y_coord_true,y_coord_pred),2) 
        return tf.add(x_diff, y_diff)

    return loss_coord