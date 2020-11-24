import tensorflow as tf
import numpy as np
from indicator.indicator import *

def noobject_confidence_loss(lambda_noobj,y_true, y_pred, S,B,C):
    def loss_noobject_confidence():
        return lambda_noobj*sum_loss_noobject_confidence()

    def sum_loss_noobject_confidence():
        SS = S[0] * S[1]
        sum = tf.convert_to_tensor(0.0,dtype='float64')
        for i in range (SS):
            for b in range(B):
                sum = tf.add(sum, loss_confidence_noobj(i,b))
        return sum

    def loss_confidence_noobj(i,b):
        indicator = my_indicator(y_true,y_pred,S,B,C)
        c_true = get_c_true(i)
        c_pred = get_c_pred(i,b)

        if indicator(0,i,b) == True:
            return loss_c(c_true,c_pred)
        else: 
            return tf.convert_to_tensor(0.0,dtype='float64')

    def get_c_true(i):
        i_th = int(i % S[0])
        j_th = int(int(i / S[0]) % S[1])
        return y_true[i_th,j_th,C+B*4]

    def get_c_pred(i,b):
        i_th = int(i % S[0])
        j_th = int(int(i / S[0]) % S[1])
        return y_pred[i_th,j_th,C+4*B+b]

    def loss_c(c_true,c_pred):
        return tf.pow(tf.subtract(c_true,c_pred),2) 

    return loss_noobject_confidence