import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as kb
import unittest





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


def yolo_head(feats,inputShape):
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
    
    box_x = ((feats[..., :1] + conv_index[...,:1]) / conv_dims[...,:1]) * inputShape[0]
    box_y = ((feats[..., 1:2] + conv_index[..., 1:2]) / conv_dims[..., 1:2]) * inputShape[1]
    box_xy = kb.concatenate([box_x, box_y])

    box_w = feats[...,2:3] * inputShape[0]
    box_h = feats[...,3:4] * inputShape[1]
    box_wh = kb.concatenate([box_w, box_h])

    return box_xy, box_wh


def yolo_loss(lambda_c = 5, lambda_no=.5, S=(50,1), B=1, C=4, inputShape=(448,448,3)):
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

        label_xy, label_wh = yolo_head(_label_box,inputShape)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
        label_xy = kb.expand_dims(label_xy, 3)  # ? * 7 * 7 * 1 * 1 * 2
        label_wh = kb.expand_dims(label_wh, 3)  # ? * 7 * 7 * 1 * 1 * 2
        label_xy_min, label_xy_max = xywh2minmax(label_xy, label_wh)  # ? * 7 * 7 * 1 * 1 * 2, ? * 7 * 7 * 1 * 1 * 2

        predict_xy, predict_wh = yolo_head(_predict_box,inputShape)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2
        predict_xy = kb.expand_dims(predict_xy, 4)  # ? * 7 * 7 * 2 * 1 * 2
        predict_wh = kb.expand_dims(predict_wh, 4)  # ? * 7 * 7 * 2 * 1 * 2
        predict_xy_min, predict_xy_max = xywh2minmax(predict_xy, predict_wh)  # ? * 7 * 7 * 2 * 1 * 2, ? * 7 * 7 * 2 * 1 * 2

         iou_scores = iou(predict_xy_min, predict_xy_max, label_xy_min, label_xy_max)  # ? * 7 * 7 * 2 * 1
        best_ious = kb.max(iou_scores, axis=4)  # ? * 7 * 7 * 2
        best_box = kb.max(best_ious, axis=3, keepdims=True)  # ? * 7 * 7 * 1

        box_mask = kb.cast(best_ious >= best_box, kb.dtype(best_ious))  # ? * 7 * 7 * 2

        no_object_loss = lambda_no * (1 - box_mask * response_mask) * kb.square(0 - predict_trust)
        object_loss = box_mask * response_mask * kb.square(1 - predict_trust)
        confidence_loss = no_object_loss + object_loss
        confidence_loss = kb.sum(confidence_loss)

        class_loss = response_mask * kb.square(label_class - predict_class)
        class_loss = kb.sum(class_loss)

        _label_box = kb.reshape(label_box, [-1, S[0], S[1], 1, 4])
        _predict_box = kb.reshape(predict_box, [-1, S[0], S[1], B, 4])

        label_xy, label_wh = yolo_head(_label_box,inputShape)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
        predict_xy, predict_wh = yolo_head(_predict_box,inputShape)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2

        box_mask = kb.expand_dims(box_mask)
        response_mask = kb.expand_dims(response_mask)

        label_x = label_xy[...,:1] / inputShape[0]
        label_y = label_xy[...,1:2] / inputShape[1]
        label_xy = kb.concatenate([label_x, label_y])

        
        predict_x = predict_xy[...,:1] / inputShape[0]
        predict_y = predict_xy[...,1:2] / inputShape[1]
        predict_xy = kb.concatenate([predict_x, predict_y])

        label_w = label_wh[...,:1] / inputShape[0]
        label_h = label_wh[...,1:2] / inputShape[1]
        label_wh = kb.concatenate([label_w, label_h])

        predict_w = predict_wh[...,:1] / inputShape[0]
        predict_h = predict_wh[...,1:2] / inputShape[1]
        predict_wh = kb.concatenate([predict_w, predict_h])


        box_loss = lambda_c * box_mask * response_mask * kb.square(label_xy - predict_xy)
        box_loss += lambda_c * box_mask * response_mask * kb.square((kb.sqrt(label_wh) - kb.sqrt(predict_wh)))
        box_loss = kb.sum(box_loss)

        loss = confidence_loss + class_loss + box_loss
        return loss
    return loss




def myLoss(lambda_c = 5, lambda_no=.5,lambda_obj=1.0, lambda_bbx=5.0, S=(50,1), B=1, C=4, inputShape=(448,448,3), batch_size=2):
    def myLoss(y_true,y_pred):
        labels = y_true
        predicts = y_pred
        print(y_true.shape)
        print(y_pred.shape)
        cell_size = S
        boxes_per_cell = B
        num_class = C
        class_scale = lambda_c
        object_scale = lambda_obj
        noobject_scale = lambda_no
        coord_scale=lambda_bbx
        boundary1 = S[0] * S[1] * C
        boundary2= boundary1 + cell_size[0] * cell_size[1] * boxes_per_cell
        
        offset = np.transpose(np.reshape(np.array(
            [np.arange(cell_size[0])] * cell_size[1] * boxes_per_cell),
            (boxes_per_cell, cell_size[0], cell_size[1])), (1, 2, 0))

        predict_classes = tf.reshape(predicts[:, :boundary1], [batch_size, cell_size[0], cell_size[1], num_class])
        predict_scales = tf.reshape(predicts[:, boundary1:boundary2], [batch_size, cell_size[0], cell_size[1], boxes_per_cell])
        predict_boxes = tf.reshape(predicts[:, boundary2:], [batch_size, cell_size[0], cell_size[1], boxes_per_cell, 4])

        response = tf.reshape(labels[:, :, :, 0], [batch_size, cell_size[0], cell_size[1], 1])
        boxes = tf.reshape(labels[:, :, :, 1:5], [batch_size, cell_size[0], cell_size[1], 1, 4])
        # boxes = tf.tile(boxes, [1, 1, 1, boxes_per_cell, 1]) / self.image_size[0]
        # boxes = tf.tile(boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size[0]

        classes = labels[:, :, :, 5:]

        offset = tf.constant(offset, dtype=tf.float32)
        offset = tf.reshape(offset, [1, cell_size[0], cell_size[1], boxes_per_cell])
        offset = tf.tile(offset, [batch_size, 1, 1, 1])
        predict_boxes_tran = tf.stack([1. * (predict_boxes[:, :, :, :, 0] + offset) / cell_size[0],
                                        1. * (predict_boxes[:, :, :, :, 1] + tf.transpose(offset, (0, 2, 1, 3))) / cell_size[1],
                                        tf.square(predict_boxes[:, :, :, :, 2]),
                                        tf.square(predict_boxes[:, :, :, :, 3])])
        predict_boxes_tran = tf.transpose(predict_boxes_tran, [1, 2, 3, 4, 0])

        iou_predict_truth = calc_iou(predict_boxes_tran, boxes)

        object_mask = tf.math.reduce_max(iou_predict_truth, 3, keepdims=True)
        object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response

        noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

        boxes_tran = tf.stack([1. * boxes[:, :, :, :, 0] * cell_size[0] - offset,
                                1. * boxes[:, :, :, :, 1] * cell_size[1] - tf.transpose(offset, (0, 2, 1, 3)),
                                tf.sqrt(boxes[:, :, :, :, 2]),
                                tf.sqrt(boxes[:, :, :, :, 3])])
        boxes_tran = tf.transpose(boxes_tran, [1, 2, 3, 4, 0])

        # class_loss
        class_delta = response * (predict_classes - classes)
        class_loss = tf.math.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]), name='class_loss') * class_scale

        # object_loss
        object_delta = object_mask * (predict_scales - iou_predict_truth)
        object_loss = tf.math.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]), name='object_loss') * object_scale

        # noobject_loss
        noobject_delta = noobject_mask * predict_scales
        noobject_loss = tf.math.reduce_mean(tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]), name='noobject_loss') * noobject_scale

        # coord_loss
        coord_mask = tf.expand_dims(object_mask, 4)
        boxes_delta = coord_mask * (predict_boxes - boxes_tran)
        coord_loss = tf.math.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]), name='coord_loss') * coord_scale

        tf.summary.scalar('class_loss', class_loss)
        tf.summary.scalar('object_loss', object_loss)
        tf.summary.scalar('noobject_loss', noobject_loss)
        tf.summary.scalar('coord_loss', coord_loss)

        tf.summary.histogram('boxes_delta_x', boxes_delta[:, :, :, :, 0])
        tf.summary.histogram('boxes_delta_y', boxes_delta[:, :, :, :, 1])
        tf.summary.histogram('boxes_delta_w', boxes_delta[:, :, :, :, 2])
        tf.summary.histogram('boxes_delta_h', boxes_delta[:, :, :, :, 3])
        tf.summary.histogram('iou', iou_predict_truth)

        return class_loss+object_loss+noobject_loss+coord_loss
    return myLoss

def calc_iou(boxes1, boxes2):
    """calculate ious
    Args:
        boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
        boxes2: 1-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
    Return:
        iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    """
    boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,
                        boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,
                        boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,
                        boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0])
    boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])

    boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
                        boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
                        boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
                        boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2.0])
    boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])

    lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
    rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

    intersection = tf.maximum(0.0, rd - lu)
    inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

    square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
        (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
    square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
        (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

    union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

    return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)





class TestStringMethods(unittest.TestCase):
    def test_loss(self):
        
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)
# TODO : generic division by input Size 
# is 448 fix for now 



tensor = tf.convert_to_tensor(np.random.rand(5,5,6))
tensorb = tf.convert_to_tensor(np.random.rand(5,5,6))

loss = yolo_loss(lambda_c = 5, lambda_no=.5, S=(5,5), B=1, C=1, inputShape=(1024,768,3))
value = loss(tensor,tensor)