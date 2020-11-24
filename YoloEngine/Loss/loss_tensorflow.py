import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as kb


def xywh2minmax(xy, wh):
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2

    return xy_min, xy_max


def iou(pred_mins, pred_maxes, true_mins, true_maxes):
    intersect_mins = kb.maximum(pred_mins, true_mins)
    intersect_maxes = kb.minimum(pred_maxes, true_maxes)
    intersect_wh = kb.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_wh = pred_maxes - pred_mins
    true_wh = true_maxes - true_mins
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    return iou_scores


def yolo_head(feats):
    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = kb.shape(feats)[1:3]  # assuming channels last
    # In YOLO the height index is the inner most iteration.
    conv_height_index = kb.arange(0, stop=conv_dims[0])
    conv_width_index = kb.arange(0, stop=conv_dims[1])
    conv_height_index = kb.tile(conv_height_index, [conv_dims[1]])

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    # conv_width_index = kb.repeat_elements(conv_width_index, conv_dims[1], axis=0)
    conv_width_index = kb.tile(
        kb.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = kb.flatten(kb.transpose(conv_width_index))
    conv_index = kb.transpose(kb.stack([conv_height_index, conv_width_index]))
    conv_index = kb.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = kb.cast(conv_index, kb.dtype(feats))

    conv_dims = kb.cast(kb.reshape(conv_dims, [1, 1, 1, 1, 2]), kb.dtype(feats))
    #box_xy = (feats[..., :2] + conv_index) / conv_dims * 448
    #box_wh = feats[..., 2:4] * 448
    
    box_x = ((feats[..., :1] + conv_index[...,:1]) / conv_dims[...,:1]) 
    box_y = ((feats[..., 1:2] + conv_index[..., 1:2]) / conv_dims[..., 1:2]) 
    box_xy = kb.concatenate([box_x, box_y])

    box_w = feats[...,2:3]
    box_h = feats[...,3:4]
    box_wh = kb.concatenate([box_w, box_h])

    return box_xy, box_wh


def my_yolo_loss_tf(lambda_c = 5, lambda_no=.5, S=(50,1), B=1, C=4):
    def loss(y_true, y_pred):

        label_class = y_true[..., :C]  # ? * S0 * S1 * C
        response_mask = y_true[..., C+B*4]  # ? * S0 * S1 
        response_mask = kb.expand_dims(response_mask)  # ? * S0 * S1 * 1
        label_box = y_true[..., C: C+ 4]  # ? * S0 * S1 * 4

        predict_class = y_pred[..., :C]  # ? * 7 * 7 * 20
        predict_trust = y_pred[..., C+B*4:]  # ? * 7 * 7 * 2
        predict_box = y_pred[..., C:C+B*4]  # ? * 7 * 7 * 8

        _label_box = kb.reshape(label_box, [-1, S[0], S[1], 1, 4])
        _predict_box = kb.reshape(predict_box, [-1, S[0], S[1], B, 4])

        label_xy, label_wh = yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
        label_xy = kb.expand_dims(label_xy, 3)  # ? * 7 * 7 * 1 * 1 * 2
        label_wh = kb.expand_dims(label_wh, 3)  # ? * 7 * 7 * 1 * 1 * 2
        label_xy_min, label_xy_max = xywh2minmax(label_xy, label_wh)  # ? * 7 * 7 * 1 * 1 * 2, ? * 7 * 7 * 1 * 1 * 2

        predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2
        predict_xy = kb.expand_dims(predict_xy, 4)  # ? * 7 * 7 * 2 * 1 * 2
        predict_wh = kb.expand_dims(predict_wh, 4)  # ? * 7 * 7 * 2 * 1 * 2
        predict_xy_min, predict_xy_max = xywh2minmax(predict_xy, predict_wh)  # ? * 7 * 7 * 2 * 1 * 2, ? * 7 * 7 * 2 * 1 * 2

        iou_scores = iou(predict_xy_min, predict_xy_max, label_xy_min, label_xy_max)  # ? * 7 * 7 * 2 * 1
        best_ious = kb.max(iou_scores, axis=4)  # ? * 7 * 7 * 2
        best_box = kb.max(best_ious, axis=3, keepdims=True)  # ? * 7 * 7 * 1

        box_mask = kb.cast(best_ious >= best_box, kb.dtype(best_ious))  # ? * 7 * 7 * 2

        no_object_loss = lambda_no * (1 - box_mask * response_mask) * kb.square(0 - predict_trust)
        object_loss = box_mask * response_mask * kb.square(1 - predict_trust)
        no_object_loss = kb.sum(no_object_loss)
        object_loss = kb.sum(object_loss)

        class_loss = response_mask * kb.square(label_class - predict_class)
        class_loss = kb.sum(class_loss)

        _label_box = kb.reshape(label_box, [-1, S[0], S[1], 1, 4])
        _predict_box = kb.reshape(predict_box, [-1, S[0], S[1], B, 4])

        label_xy, label_wh = yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
        predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2

        box_mask = kb.expand_dims(box_mask)
        response_mask = kb.expand_dims(response_mask)

        label_x = label_xy[...,:1]
        label_y = label_xy[...,1:2] 
        label_xy = kb.concatenate([label_x, label_y])

        
        predict_x = predict_xy[...,:1]
        predict_y = predict_xy[...,1:2] 
        predict_xy = kb.concatenate([predict_x, predict_y])

        label_w = label_wh[...,:1]
        label_h = label_wh[...,1:2]
        label_wh = kb.concatenate([label_w, label_h])

        predict_w = predict_wh[...,:1]
        predict_h = predict_wh[...,1:2] 
        predict_wh = kb.concatenate([predict_w, predict_h])


        box_loss_xy = lambda_c * box_mask * response_mask * kb.square(label_xy - predict_xy)
        box_loss_wh = lambda_c * box_mask * response_mask * kb.square((kb.sqrt(label_wh) - kb.sqrt(predict_wh)))
        box_loss_xy = kb.sum(box_loss_xy)
        box_loss_wh = kb.sum(box_loss_wh)

        loss = (no_object_loss + object_loss + class_loss + box_loss_xy + box_loss_wh)

        tf.summary.scalar('no_object_loss', no_object_loss)
        tf.summary.scalar('object_loss', object_loss)
        tf.summary.scalar('class_loss', class_loss)
        tf.summary.scalar('box_loss_xy', box_loss_xy)
        tf.summary.scalar('box_loss_wh', box_loss_wh)
        return loss
    return loss