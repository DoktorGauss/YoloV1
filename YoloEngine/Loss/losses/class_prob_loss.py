import tensorflow as tf
import numpy as np
from indicator.indicator import *

def class_confidence_loss(lambda_class_conf,y_true, y_pred, S,B,C):
    def loss_class_confidence():
        return lambda_class_conf*sum_loss_class_confidence()

    def sum_loss_class_confidence():
        SS = S[0] * S[1]
        sum = tf.convert_to_tensor(0.0,dtype='float64')
        for i in range (SS):
            for c in range(C):
                sum = tf.add(sum,loss_confidence_class(i,c))
        return sum

    def loss_confidence_class(i,c):
        indicator = my_indicator(y_true,y_pred,S,B,C)
        c_true_class = get_c_true_class(i,c)
        c_pred_class = get_c_pred_class(i,c)
        if indicator(1,i,-1) == True:
            return loss_c_class(c_true_class,c_pred_class)
        else: 
            return tf.convert_to_tensor(0.0, dtype='float64')

    def get_c_true_class(i,c):
        i_th = int(i % S[0])
        j_th = int(int(i / S[0]) % S[1])
        return y_true[i_th,j_th,c]

    def get_c_pred_class(i,c):
        i_th = int(i % S[0])
        j_th = int(int(i / S[0]) % S[1])
        return y_pred[i_th,j_th,c]

    def loss_c_class(c_true_class,c_pred_class):
        diff = tf.subtract(c_true_class,c_pred_class)
        result = tf.pow(diff,2)
        return result

    return loss_class_confidence