import keras
import tensorflow as tf
import numpy as np





def MYLOSS (lambda_c = 5, lambda_no=.5,lambda_obj=1.0, lambda_bbx=5.0, S=(50,1), B=1, C=4, inputShape=(448,448,3), batch_size=2):
    def myLoss(y_true,y_pred):
        predicts = y_pred
        labels = y_true
        image_shape = inputShape

        cell_size = S
        boxes_per_cell = B
        num_classes = C
        class_scale = lambda_c
        object_scale = lambda_obj
        noobject_scale = lambda_no
        coord_scale=lambda_bbx
        boundary1 = S[0] * S[1] * C
        boundary2= boundary1 + cell_size[0] * cell_size[1] * boxes_per_cell



        index_classification = tf.multiply(tf.multiply(cell_size[0],cell_size[1]), num_classes)
        index_confidence = tf.multiply(tf.multiply(cell_size[0],cell_size[1]), num_classes + boxes_per_cell)


        predict_classes = tf.reshape(predicts[:, :index_classification], [-1, cell_size[0], cell_size[1], num_classes])
        predict_scales = tf.reshape(predicts[:, index_classification:index_confidence], [-1,cell_size[0], cell_size[1], boxes_per_cell])
        predict_boxes = tf.reshape(predicts[:, index_confidence:], [-1, cell_size[0], cell_size[1], boxes_per_cell, 4])

        response = tf.reshape(labels[:, :, :, 0], [-1, cell_size[0], cell_size[1], 1])
        regression_labels = tf.reshape(labels[:, :, :, 1:5], [-1, cell_size[0], cell_size[1], 1, 4])
        regression_labels =tf.math.divide(tf.tile(regression_labels, [1, 1, 1, boxes_per_cell, 1]), tf.cast(image_shape[0],dtype='float'))
        classification_labels = labels[:, :, :, 5:]

        offset = np.transpose(np.reshape(np.array(
            [np.arange(cell_size[0])] * cell_size[1] * boxes_per_cell),
            (boxes_per_cell, cell_size[0], cell_size[1])), (1, 2, 0))
        offset = tf.constant(offset, dtype=tf.float32)
        offset = tf.reshape(offset, [1, cell_size[0], cell_size[1], boxes_per_cell])

        regression = tf.stack([(predict_boxes[:, :, :, :, 0] + offset) / cell_size[0],
                                       (predict_boxes[:, :, :, :, 1] + tf.transpose(offset, (0, 2, 1, 3))) / cell_size[1],
                                       tf.square(predict_boxes[:, :, :, :, 2]),
                                       tf.square(predict_boxes[:, :, :, :, 3])])
        regression = tf.transpose(regression, [1, 2, 3, 4, 0])

        iou_predict_truth = calc_iou(regression, regression_labels)

        # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        object_mask = tf.reduce_max(iou_predict_truth, 3, keepdims=True)
        object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response

        regression_target = tf.stack([regression_labels[:, :, :, :, 0] * cell_size[0] - offset,
                               regression_labels[:, :, :, :, 1] * cell_size[1] - tf.transpose(offset, (0, 2, 1, 3)),
                               tf.sqrt(regression_labels[:, :, :, :, 2]),
                               tf.sqrt(regression_labels[:, :, :, :, 3])])
        regression_target = tf.transpose(regression_target, [1, 2, 3, 4, 0])

        # regression loss (localization loss) coord_loss
        coord_loss,boxes_delta = regression_loss(regression_target, predict_boxes, object_mask,coord_scale)

        # confidence loss
        object_loss, noobject_loss = confidence_loss(predict_scales, iou_predict_truth, object_mask,object_scale,noobject_scale)

        # classification loss
        cls_loss = classification_loss(classification_labels, predict_classes, response,class_scale)

        tf.summary.scalar('class_loss', cls_loss)
        tf.summary.scalar('object_loss', object_loss)
        tf.summary.scalar('noobject_loss', noobject_loss)
        tf.summary.scalar('coord_loss', coord_loss)

        tf.summary.histogram('boxes_delta_x', boxes_delta[:, :, :, :, 0])
        tf.summary.histogram('boxes_delta_y', boxes_delta[:, :, :, :, 1])
        tf.summary.histogram('boxes_delta_w', boxes_delta[:, :, :, :, 2])
        tf.summary.histogram('boxes_delta_h', boxes_delta[:, :, :, :, 3])
        tf.summary.histogram('iou', iou_predict_truth)

        return coord_loss+object_loss+noobject_loss+cls_loss
    return myLoss


def classification_loss(labels,  classification, response,class_scale):
        class_delta = response * (labels - classification)
        cls_loss = tf.math.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]), name='cls_loss') * class_scale

        return cls_loss

def regression_loss(regression_target, regression,object_mask,coord_scale):
    coord_mask = tf.expand_dims(object_mask, 4)
    boxes_delta = coord_mask * (regression_target - regression)
    coord_loss = tf.math.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                                name='coord_loss') * coord_scale
    reg_loss = tf.constant(0,dtype=tf.float32)
    return reg_loss,boxes_delta

def confidence_loss(predict_scales, iou_predict_truth, object_mask,object_scale,noobject_scale):
    # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

    # object_loss
    object_delta = object_mask * (iou_predict_truth - predict_scales)
    object_loss = tf.math.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),  name='object_loss') * object_scale

    # noobject_loss
    noobject_delta = noobject_mask * predict_scales
    noobject_loss = tf.math.reduce_mean(tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]), name='noobject_loss') * noobject_scale

    return [object_loss, noobject_loss]


def calc_iou(boxes1, boxes2, scope='iou'):
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

        # calculate the left up point & right down point
        lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
        rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

        # intersection
        intersection = tf.maximum(0.0, rd - lu)
        inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

        # calculate the boxs1 square and boxs2 square
        square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
                    (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
        square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
                    (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

        union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

def compute_output_shape(input_shape):
        return [(1,), (1,), (1,), (1,)]

def compute_mask(inputs, mask=None):
    return [None, None, None, None]