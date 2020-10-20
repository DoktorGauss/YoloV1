import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as kb


def true_positive_caller(S=(50,1), B=1, C=4):
    def true_positive(y_true, y_pred):
        true_class = y_true[..., : C]  # ? * 50 * 1 * (C*B)
        predict_class = y_pred[..., : C]  # ? * 50 * 1 * (C*B)
        metric = tf.keras.losses.categorical_crossentropy(true_class, predict_class)
        metric_list = kb.flatten(metric)
        return kb.mean(metric_list)
    return true_positive


