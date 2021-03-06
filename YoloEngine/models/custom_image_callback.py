import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.backend as kb
import tensorflow.keras as keras
import numpy as np
import os
import xml.etree.ElementTree as ET
import tensorflow as tf
import copy
import cv2


def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 100, iou_threshold = 0.1):
    """
    Applies Non-max suppression (NMS) to set of boxes
    
    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box
    
    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """
    
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')     # tensor to be used in tf.image.non_max_suppression()
    #K.get_session().run(tf.variables_initializer([max_boxes_tensor])) # initialize variable max_boxes_tensor
    
    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    boxes = K.reshape(boxes, shape=(-1,4))
    nms_indices = tf.image.non_max_suppression( boxes, scores, max_boxes_tensor, iou_threshold)
    
    # Use K.gather() to select only nms_indices from scores, boxes and classes
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)
    
    return scores, boxes, classes

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    """Filters YOLO boxes by thresholding on object and class confidence.
    
    Arguments:
    box_confidence -- tensor of shape (19, 19, 5, 1)
    boxes -- tensor of shape (19, 19, 5, 4)
    box_class_probs -- tensor of shape (19, 19, 5, 80)
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    
    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes
    
    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold. 
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """
    
    # Step 1: Compute box scores
    box_scores = box_confidence * box_class_probs
    
    # Step 2: Find the box_classes using the max box_scores, keep track of the corresponding score
    box_classes = K.argmax(box_scores, axis = -1)
    box_class_scores = K.max(box_scores, axis = -1, keepdims = None)
    
    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    filtering_mask = box_class_scores >= threshold
    
    # Step 4: Apply the mask to box_class_scores, boxes and box_classes
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    
    return scores, boxes, classes

def my_yolo_head(feats,inputShape):
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
  
  box_x = (feats[..., :1] + conv_index[...,:1]) / conv_dims[...,:1]
  box_y = (feats[..., 1:2] + conv_index[..., 1:2]) / conv_dims[..., 1:2]
  box_xy = kb.concatenate([box_x, box_y])

  box_w = feats[...,2:3]
  box_h = feats[...,3:4]
  box_wh = kb.concatenate([box_w, box_h])

  return box_xy, box_wh   

def yolo_outputs(S,B,C,inputShape):
  def yolo_output(output):
      output_class = output[..., :C]  # ? * 7 * 7 * 20
      output_trust = output[..., C+B*4:]  # ? * 7 * 7 * 2
      output_box = output[..., C:C+B*4]  # ? * 7 * 7 * 8
      _output_box = kb.reshape(output_box, [-1, S[0], S[1], B, 4])
      output_xy, output_wh = my_yolo_head(_output_box,inputShape)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2
      #output_xy = kb.expand_dims(predict_xy, 4)  # ? * 7 * 7 * 2 * 1 * 2
      #output_wh = kb.expand_dims(predict_wh, 4)  # ? * 7 * 7 * 2 * 1 * 2

      return output_trust, output_xy, output_wh, output_class
  return yolo_output
def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return K.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])

def get_boxes(yolo_output,S,B,C,inputShape,score_threshold):
  _yolo_outputs = yolo_outputs(S,B,C,inputShape)

  box_confidence, box_xy, box_wh, box_class_probs = _yolo_outputs(yolo_output)

  # Convert boxes to be ready for filtering functions 
  boxes = yolo_boxes_to_corners(box_xy, box_wh)

  # Perform Score-filtering with a threshold of score_threshold
  scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = score_threshold)

  # Scales the predicted boxes in order to be drawable on the image.
  # boxes = scale_boxes(boxes, image_shape)
  boxes = kb.reshape(boxes,(-1,B,4))
  boxes, scores, classes = prepare(boxes,scores,classes,B)
  #kb.print_tensor('scores',scores.shape)
  #kb.print_tensor('boxes',boxes.shape)
  #kb.print_tensor('classes',classes.shape)
  # Perform Non-max suppression with a threshold of iou_threshold
  scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes = S[0]*S[1]*B, iou_threshold = 0.1)
  return scores, boxes, classes

def prepare(boxes,scores,classes,B):
  rboxes = boxes[:,0,:]
  rscores = scores
  rclasses = classes
  for b in range(B-1):
    rboxes = kb.concatenate(( kb.reshape(rboxes,(-1,4)), kb.reshape(boxes[:,b+1,:],(-1,4))), axis=0)
    rscores = kb.concatenate((rscores,scores), axis=0)
    rclasses = kb.concatenate((rclasses,classes), axis=0)

  return rboxes, rscores, rclasses



def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    from PIL import Image
    height, width, channel = tensor.shape
    image = Image.fromarray((tensor * 255).astype(np.uint8))
    return np.asarray(image,dtype='uint8')

class Box:
    def __init__(self):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.c = float()
        self.prob = float()
        self.xmin = int()
        self.xmax = int()
        self.ymin = int()
        self.ymax = int()
        self.classIndx=int()

def overlap(x1, w1, x2, w2):
    l1 = x1 - w1 / 2.
    l2 = x2 - w2 / 2.
    left = max(l1, l2)
    r1 = x1 + w1 / 2.
    r2 = x2 + w2 / 2.
    right = min(r1, r2)
    return right - left


def box_intersection(a, b):
    """

    :param a: Box 1
    :param b: Box 2
    :return: Intersection area of the 2 boxes
    """
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0
    area = w * h
    return area


def box_union(a, b):
    """

    :param a: Box 1
    :param b: Box 2
    :return: Area under the union of the 2 boxes
    """
    i = box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u


def box_iou(a, b):
    """

    :param a: Box 1
    :param b: Box 2
    :return: Intersection over union, which is ratio of intersection area to union area of the 2 boxes
    """
    return box_intersection(a, b) / box_union(a, b)



def yolo_output_to_box(yolo_output, conf_threshold = 0.05, iou_threshold=0.1, sqrt=1.8,C=20, B=2, S=(50,1), imageShape = (1024,768,3)):
    boxes = []
    SS = S[0]*S[1]
    confidence_scores = yolo_output[:,:, C+B*4:]
    cords = yolo_output[:,:, C:C+B*4]
    probabilities = yolo_output[:,:,:C]
    # Reshape the arrays so that its easier to loop over them
    probabilities = tf.reshape(probabilities, (SS,C))
    confs = tf.reshape(confidence_scores, (SS,B))
    cords = tf.reshape(cords, (SS,B,4))

    #probabilities = probabilities.reshape((SS, C))
    #confs = confidence_scores.reshape((SS, B))
    #cords = cords.reshape((SS, B, 4))

    for grid in range(SS):
        for b in range(B):
            bx = Box()
            bx.c = confs[grid, b]
            p = probabilities[grid, :] * bx.c
            for i in range(C):
              if p[i] >= conf_threshold:
                # bounding box xand y coordinates are offsets of a particular grid cell location,
                # so they are also bounded between 0 and 1.
                # convert them absolute locations relative to the image size
                gx = int(grid % S[1])
                gy = int(int(grid // S[1]) % S[0])
                gw = imageShape[0] / S[0]
                gh = imageShape[1] / S[1]

                bx.x =  (float(gx*gw) + cords[grid,b,0] * float(gw))/imageShape[0]
                bx.y = (float(gy*gh) + cords[grid,b,1] * float(gh))/imageShape[1]
                #bx.x = (cords[grid, b, 0] + grid % S[1]) / S[1]
                #bx.y = (cords[grid, b, 1] + grid // S[1]) / S[0]
                bx.w = cords[grid, b, 2] ** sqrt
                bx.h = cords[grid, b, 3] ** sqrt

                bx.xmin = bx.x - bx.w/2
                bx.xmax = bx.x + bx.w/2
                bx.ymin = bx.y - bx.h/2
                bx.ymax = bx.y + bx.h/2
                bx.classIndx = i
                bx.prob = p[i]

                boxes.append(bx)
           
            
            # multiply confidence scores with class probabilities to get class sepcific confidence scores
            

    # sort the boxes by confidence score, in the descending order
    boxes.sort(key=lambda b: b.prob, reverse=True)


    for i in range(len(boxes)):
        boxi = boxes[i]
        if boxi.prob == 0:
            continue

        for j in range(i + 1, len(boxes)):
            boxj = boxes[j]

            # If boxes have more than 40% overlap then retain the box with the highest confidence score
            if box_iou(boxi, boxj) >= iou_threshold:
                boxes[j].prob = 0

    boxes = [b for b in boxes if b.prob > 0]

    boxarray =np.asarray([[b.ymin, b.xmin, b.ymax, b.xmax] for b in boxes])
    boxarray = np.asarray(boxarray)
    return tf.convert_to_tensor(boxarray,dtype='float64')


def draw_boxes(boxes,im):
    imgcv1 = im.copy()
    [xmin, xmax] = crop_dim[0]
    [ymin, ymax] = crop_dim[1]
    
    height, width, _ = imgcv1.shape
    for b in boxes:
        w = xmax - xmin
        h = ymax - ymin

        left  = int ((b.x - b.w/2.) * w) + xmin
        right = int ((b.x + b.w/2.) * w) + xmin
        top   = int ((b.y - b.h/2.) * h) + ymin
        bot   = int ((b.y + b.h/2.) * h) + ymin

        if left  < 0:
            left = 0
        if right > width - 1:
            right = width - 1
        if top < 0:
            top = 0
        if bot>height - 1: 
            bot = height - 1
        
        thick = 5 #int((height + width // 150))
        
        cv2.rectangle(imgcv1, (left, top), (right, bot), (255,0,0), thick)

    return imgcv1
class TensorBoardImage(keras.callbacks.Callback):
    def __init__(self, tag,validationset,classes,S,B,C,imageshape):
        super(TensorBoardImage, self).__init__()
        self.tag = tag
        self.validationset = validationset
        self.classes = classes
        self.S = S
        self.B = B
        self.C = C
        self.imageShape = imageshape
        self.bTrueDrawed = False
      

    def on_epoch_begin(self, epoch, logs={}):
        count = 0
        index = 0
        images, y_val = self.validationset.__getitem__(0)
        image = images[index]
        normimage = image.copy()

        y_pred = self.model.predict(kb.reshape(image,(-1,self.imageShape[0],self.imageShape[1],self.imageShape[2])))
        scores, boxes, classes = get_boxes(y_pred,self.S,self.B,self.C,self.imageShape,0.5)
        boxes = kb.reshape(boxes, (1,-1,4))
        colors = np.array([[1.0, 0.0, 0.0]])
        K.print_tensor(boxes.shape)
        #K.print_tensor(boxes[0,:,:])
        if tf.rank(boxes) == 3 and len(boxes)>0:
            myBnDBoxImage = tf.image.draw_bounding_boxes([image], boxes,colors=colors)
            #if not self.bTrueDrawed:
            if True:
              y_val = kb.reshape(y_val[index], (-1, self.S[0],self.S[1],5*self.B+self.C))
              val_scores, val_boxes, val_classes = get_boxes(tf.cast(y_val, dtype='float'), self.S, self.B,self.C, self.imageShape, 0.3)
              val_boxes = kb.reshape(val_boxes, (1,-1,4))
              val_colors = np.array([[0.0, 1.0, 0.0]])
              myTrueBoxImage = tf.image.draw_bounding_boxes(myBnDBoxImage, val_boxes,colors=val_colors)
              #tf.summary.image(self.tag+str(count)+'_true', myTrueBoxImage, step=epoch)
              self.bTrueDrawed = True
            tf.summary.image(self.tag+str(count), myTrueBoxImage, step=epoch)
        count += 1

   

