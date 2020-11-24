import tensorflow as tf
import numpy as np
from indicator.indicator import *


def wh_loss(lambda_wh,y_true, y_pred, S,B,C):
    def loss_wh():
        return lambda_wh*sum_loss_wh()

    def sum_loss_wh():
        SS = S[0] * S[1]
        sum = tf.convert_to_tensor(0.0,dtype='float64')
        for i in range (SS):
            for b in range(B):
                sum = tf.add(sum,loss_wh_obj(i,b))
        return sum

    def loss_wh_obj(i,b):
        indicator = my_indicator(y_true,y_pred,S,B,C)
        w_true = get_w_true(i)
        w_pred = get_w_pred(i,b)
        h_true = get_h_true(i)
        h_pred = get_h_pred(i,b)

        if indicator(1,i,b) == True:
            return calc_loss_wh(w_true,w_pred,h_true,h_pred)
        else: 
            return tf.convert_to_tensor(0.0,dtype='float64')


    def get_w_true(i):
        i_th = int(i % S[0])
        j_th = int(int(i / S[0]) % S[1])
        return y_true[i_th,j_th,C+2]

    def get_w_pred(i,b):
        i_th = int(i % S[0])
        j_th = int(int(i / S[0]) % S[1])
        return y_pred[i_th,j_th, C+b*4+2]
        
    def get_h_pred(i,b):
        i_th = int(i % S[0])
        j_th = int(int(i / S[0]) % S[1])
        return y_pred[i_th,j_th,C+b*4+3]

    def get_h_true(i):
        i_th = int(i % S[0])
        j_th = int(int(i / S[0]) % S[1])
        return y_true[i_th,j_th,C+3]

    def calc_loss_wh(w_true,w_pred,h_true,h_pred):
        w_diff = tf.pow(tf.subtract(tf.sqrt(w_true),tf.sqrt(w_pred)),2) 
        h_diff = tf.pow(tf.subtract(tf.sqrt(h_true),tf.sqrt(h_pred)),2) 
        return tf.add(w_diff, h_diff)

    return loss_wh