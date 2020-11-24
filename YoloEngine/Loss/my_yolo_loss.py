import keras
import tensorflow as tf
import numpy as np
from Loss.losses.class_prob_loss import class_confidence_loss
from Loss.losses.coord_loss_wh import wh_loss
from Loss.losses.coord_loss_xy import coord_loss
from Loss.losses.noobject_confidence_loss import noobject_confidence_loss
from Loss.losses.object_confidence_loss import object_confidence_loss


def my_yolo_loss(lambda_c = 1.0, lambda_no=.5,lambda_obj=1.0, lambda_bbx=5.0, S=(50,1), B=1, C=4, batch_size=2):
    def myLoss(y_true,y_pred):
        class_confidence_loss_val = 0.0
        loss_wh_val = 0.0
        loss_xy_val = 0.0
        noobject_confidence_loss_val =0.0
        object_confidence_loss_val = 0.0

        for batch in range(batch_size):
            y_true_batch = y_true[batch,:,:,:]
            y_pred_batch = y_pred[batch,:,:,:]

            class_confidence_loss_fct = class_confidence_loss(lambda_c,y_true_batch,y_pred_batch,S,B,C)
            loss_wh_fct = wh_loss(lambda_bbx, y_true_batch, y_pred_batch,S,B,C)
            loss_xy_fct = coord_loss(lambda_bbx, y_true_batch, y_pred_batch, S,B,C)
            noobject_confidence_fct = noobject_confidence_loss(lambda_no,y_true_batch,y_pred_batch,S,B,C)
            object_confidence_loss_fct = object_confidence_loss(lambda_obj, y_true_batch, y_pred_batch, S,B,C)

            class_confidence_loss_val += class_confidence_loss_fct()
            loss_wh_val += loss_wh_fct()
            loss_xy_val += loss_xy_fct()
            noobject_confidence_loss_val += noobject_confidence_fct()
            object_confidence_loss_val += object_confidence_loss_fct()

        tf.summary.scalar('class_confidence_loss', class_confidence_loss_val)
        tf.summary.scalar('loss_wh', loss_wh_val)
        tf.summary.scalar('loss_xy', loss_xy_val)
        tf.summary.scalar('noobject_confidence_loss', noobject_confidence_loss_val)
        tf.summary.scalar('object_confidence_loss', object_confidence_loss_val)

        loss = class_confidence_loss_val + loss_wh_val + loss_xy_val +  noobject_confidence_loss_val + object_confidence_loss_val
        return loss
    return myLoss