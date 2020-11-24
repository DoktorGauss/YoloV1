import cv2
import numpy as np
import tensorflow.keras.backend as kb

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        
        self.c     = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score



def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union

def draw_boxes(image, boxes, labels):
    image_h, image_w, _ = image.shape

    for box in boxes:
        xmin = int(box.xmin)
        ymin = int(box.ymin)
        xmax = int(box.xmax)
        ymax = int(box.ymax)

        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,255,0), 3)
        cv2.putText(image, 
                    labels[box.get_label()] + ' ' + str(box.get_score()), 
                    (xmin, ymin - 13), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1e-3 * image_h, 
                    (0,255,0), 2)
        
    return image     



def decode_netout_keras_backend(netout,nb_class,imageShape, S=(7,7), B = 2, C=20, obj_threshold=0.3, nms_threshold=0.3):
    grid_h, grid_w = netout.shape[:2]
    nb_box = int((netout.shape[2]-C)/5)
    boxes = []
    # decode the output by the network
    # netout[...,  C+B*4]  = _sigmoid(netout[...,  C+B*4])
    # confidenceMatrix = np.max(netout[...,  C+B*4:], axis=1,keepdims=True)[..., np.newaxis]
    confidenceMatrix = kb.zeros((S[0],S[1],1))
    for row in range(len(netout)):
        for column in range(len(netout[row])):
            mymax = kb.max(netout[row,column,C+B*4:])
            confidenceMatrix[row][column] = mymax

    netout[..., :C] =  confidenceMatrix * _softmax(netout[..., :C])
    netout[..., :C] *= netout[..., :C] > obj_threshold
    


    predict_class = netout[..., :C]  # ? * 7 * 7 * 20
    predict_trust = netout[..., C+B*4:]  # ? * 7 * 7 * 2
    predict_box = netout[..., C:C+B*4]  # ? * 7 * 7 * 8
    _predict_box = np.reshape(predict_box, (S[0], S[1], B, 4))


    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                class_confidence = netout[row,col,:C]
                confidence = netout[row,col,C+B*4+b]
                
                if confidence > 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = yolo_head(_predict_box,imageShape,row,col,b)
                    box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, confidence, class_confidence)
                    
                    boxes.append(box)

    # suppress non-maximal boxes
    for c in range(nb_class):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            
            if boxes[index_i].classes[c] == 0: 
                continue
            else:
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    
                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0
                        
    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]
    
    return boxes    

def decode_netout(netout,nb_class,imageShape, S=(7,7), B = 2, C=20, obj_threshold=0.3, nms_threshold=0.3):
    grid_h, grid_w = netout.shape[:2]
    nb_box = int((netout.shape[2]-C)/5)
    boxes = []
    # decode the output by the network
    # netout[...,  C+B*4]  = _sigmoid(netout[...,  C+B*4])
    # confidenceMatrix = np.max(netout[...,  C+B*4:], axis=1,keepdims=True)[..., np.newaxis]
    confidenceMatrix = np.zeros((S[0],S[1],1))
    for row in range(len(netout)):
        for column in range(len(netout[row])):
            mymax = np.max(netout[row,column,C+B*4:])
            confidenceMatrix[row][column] = mymax

    netout[..., :C] =  confidenceMatrix * _softmax(netout[..., :C])
    netout[..., :C] *= netout[..., :C] > obj_threshold
    


    predict_class = netout[..., :C]  # ? * 7 * 7 * 20
    predict_trust = netout[..., C+B*4:]  # ? * 7 * 7 * 2
    predict_box = netout[..., C:C+B*4]  # ? * 7 * 7 * 8
    _predict_box = np.reshape(predict_box, (S[0], S[1], B, 4))


    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                class_confidence = netout[row,col,:C]
                confidence = netout[row,col,C+B*4+b]
                
                if confidence > 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = yolo_head(_predict_box,imageShape,row,col,b)
                    box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, confidence, class_confidence)
                    
                    boxes.append(box)

    # suppress non-maximal boxes
    for c in range(nb_class):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            
            if boxes[index_i].classes[c] == 0: 
                continue
            else:
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    
                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0
                        
    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]
    
    return boxes    

def yolo_head(feats,inputShape,row,col,b):
    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = np.shape(feats)[0:2]  # assuming channels last
    # In YOLO the height index is the inner most iteration.
    conv_height_index = np.arange(0, stop=conv_dims[0])
    conv_width_index = np.arange(0, stop=conv_dims[1])
    conv_height_index = np.tile(conv_height_index, [conv_dims[1]])

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    # conv_width_index = kb.repeat_elements(conv_width_index, conv_dims[1], axis=0)
    conv_width_index = np.tile(
        np.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = np.transpose(conv_width_index).flatten()
    conv_index = np.transpose(np.stack([conv_height_index, conv_width_index]))
    conv_index = np.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    # conv_index = np.cast(conv_index, np.dtype(feats.dtype))
    conv_dims = np.reshape(conv_dims, [1, 1, 1, 1, 2])

    # conv_dims = np.cast(np.reshape(conv_dims, [1, 1, 1, 1, 2]), np.dtype(feats.dtype))
    #box_xy = (feats[..., :2] + conv_index) / conv_dims * 448
    #box_wh = feats[..., 2:4] * 448
    
    box_x = ((feats[..., :1] + conv_index[...,:1]) / conv_dims[...,:1]) * inputShape[0]
    box_y = ((feats[..., 1:2] + conv_index[..., 1:2]) / conv_dims[..., 1:2]) * inputShape[1]
    box_xy = np.concatenate([box_x, box_y])

    box_w = feats[...,2:3] * inputShape[0]
    box_h = feats[...,3:4] * inputShape[1]
    box_wh = np.concatenate([box_w, box_h])



    box_x = box_x[0][row][col][b][0]
    box_y = box_y[0][row][col][b][0]
    box_w = box_w[row][col][b][0]
    box_h = box_h[row][col][b][0]


    return box_x,box_y,box_w,box_h

def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua  
    
def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap      
        
def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3          

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def _softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)
    
    if np.min(x) < t:
        x = x/np.min(x)*t
        
    e_x = np.exp(x)
    
    return e_x / e_x.sum(axis, keepdims=True)

def _softmax_keras_backend(x, axis=-1, t=-100.):
    x = x - kb.max(x)
    
    if kb.min(x) < t:
        x = x/kb.min(x)*t
        
    e_x = KeyboardInterrupt.exp(x)
    
    return e_x / e_x.sum(axis, keepdims=True)
    
def iou(boxA,boxB):
      # boxA = boxB = [x1,y1,x2,y2]

  # Determine the coordinates of the intersection rectangle
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])
 
  # Compute the area of intersection
  intersection_area = (xB - xA + 1) * (yB - yA + 1)
 
  # Compute the area of both rectangles
  boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
  boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
  # Compute the IOU
  iou = intersection_area / float(boxA_area + boxB_area - intersection_area)

  return iou



def non_maximal_suppression(thresholded_predictions,iou_threshold):

  nms_predictions = []

  # Add the best B-Box because it will never be deleted
  nms_predictions.append(thresholded_predictions[0])

  # For each B-Box (starting from the 2nd) check its iou with the higher score B-Boxes
  # thresholded_predictions[i][0] = [x1,y1,x2,y2]
  i = 1
  while i < len(thresholded_predictions):
    n_boxes_to_check = len(nms_predictions)
    #print('N boxes to check = {}'.format(n_boxes_to_check))
    to_delete = False

    j = 0
    while j < n_boxes_to_check:
        curr_iou = iou(thresholded_predictions[i][0],nms_predictions[j][0])
        if(curr_iou > iou_threshold ):
            to_delete = True
        #print('Checking box {} vs {}: IOU = {} , To delete = {}'.format(thresholded_predictions[i][0],nms_predictions[j][0],curr_iou,to_delete))
        j = j+1

    if to_delete == False:
        nms_predictions.append(thresholded_predictions[i])
    i = i+1

  return nms_predictions

def postprocessing(predictions,input_img_path,score_threshold,iou_threshold,input_height,input_width,classes,S,B,C,anchors):
    
    input_image = cv2.imread(input_img_path)
    input_image = cv2.resize(input_image,(input_height, input_width), interpolation = cv2.INTER_CUBIC)

    n_classes = C
    n_grid_cells_n = S[0]
    n_grid_cells_m = S[1]

    grid_width = input_width/n_grid_cells_m
    grid_height = input_height/n_grid_cells_n

    n_b_boxes = B
    n_b_box_coord = 4

    # Names and colors for each class
    #classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    classes = classes
    colors = [(254.0, 254.0, 254), (239.88888888888889, 211.66666666666669, 127), 
                (225.77777777777777, 169.33333333333334, 0), (211.66666666666669, 127.0, 254),
                (197.55555555555557, 84.66666666666667, 127), (183.44444444444443, 42.33333333333332, 0),
                (169.33333333333334, 0.0, 254), (155.22222222222223, -42.33333333333335, 127),
                (141.11111111111111, -84.66666666666664, 0), (127.0, 254.0, 254), 
                (112.88888888888889, 211.66666666666669, 127), (98.77777777777777, 169.33333333333334, 0),
                (84.66666666666667, 127.0, 254), (70.55555555555556, 84.66666666666667, 127),
                (56.44444444444444, 42.33333333333332, 0), (42.33333333333332, 0.0, 254), 
                (28.222222222222236, -42.33333333333335, 127), (14.111111111111118, -84.66666666666664, 0),
                (0.0, 254.0, 254), (-14.111111111111118, 211.66666666666669, 127)]

    # Pre-computed YOLOv2 shapes of the k=5 B-Boxes
    #anchors = [1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52]

    thresholded_predictions = []
    print('Thresholding on (Objectness score)*(Best class score) with threshold = {}'.format(score_threshold))

    # IMPORTANT: reshape to have shape = [ 13 x 13 x (5 B-Boxes) x (4 Coords + 1 Obj score + 20 Class scores ) ]
    # From now on the predictions are ORDERED and can be extracted in a simple way!
    # We have 13x13 grid cells, each cell has 5 B-Boxes, each B-Box have 25 channels with 4 coords, 1 Obj score , 20 Class scores
    # E.g. predictions[row, col, b, :4] will return the 4 coords of the "b" B-Box which is in the [row,col] grid cell
    #predictions = np.reshape(predictions,(13,13,5,25))
    #[class_probs, boxes,confs

    #index_coords = lambda x :  [C:C+x*4]
    #index_objScore = lambda x: [C+B*4+x]
    #index_classScore = [:C]
    predictions = np.reshape(predictions,(S[0],S[1],5*B+C))

    reordered_prediction = np.zeros((S[0],S[1],B*(5+C)))
    offset = 5+C
    for b in range(B):
        currentOffset = b*offset
        pred_b_classScore = predictions[:,:,:C]
        pred_b_objScore = predictions[:,:,C+B*4+b]
        pred_b_coords = predictions[:,:,C+b*4:C+(b+1)*4]

        reordered_prediction[:,:,currentOffset:currentOffset+4] = pred_b_coords
        reordered_prediction[:,:,currentOffset+4] = pred_b_objScore
        reordered_prediction[:,:,currentOffset+5:currentOffset+5+C] = pred_b_classScore 


    #reorder = lambda x: predictions[:,:,index_coords(x)],predictions[:,:,index_objScore(x)],predictions[:,:,index_classScore(x)]
#    predictions = [:,:, reorder(b) for b in range(B)]
    predictions = np.reshape(reordered_prediction,(S[0],S[1],B,5+C))


    # IMPORTANT: Compute the coordinates and score of the B-Boxes by considering the parametrization of YOLOv2
    for row in range(n_grid_cells_n):
        for col in range(n_grid_cells_m):
            for b in range(n_b_boxes):

                tx, ty, tw, th, tc = predictions[row, col, b, :5]

                # IMPORTANT: (416 img size) / (13 grid cells) = 32!
                # YOLOv2 predicts parametrized coordinates that must be converted to full size
                # final_coordinates = parametrized_coordinates * 32.0 ( You can see other EQUIVALENT ways to do this...)
                center_x = (float(col) + _sigmoid(tx)) * grid_width
                center_y = (float(row) + _sigmoid(ty)) * grid_height

                # roi_w = np.exp(tw) * anchors[2*b + 0] * grid_width
                # roi_h = np.exp(th) * anchors[2*b + 1] * grid_height
                roi_w = np.exp(tw)* grid_width
                roi_h = np.exp(th) * grid_height

                final_confidence = _sigmoid(tc)

                # Find best class
                class_predictions = predictions[row, col, b, 5:]
                class_predictions = _softmax(class_predictions)

                class_predictions = tuple(class_predictions)
                best_class = class_predictions.index(np.max(class_predictions))
                best_class_score = class_predictions[best_class]

                # Compute the final coordinates on both axes
                left   = int(center_x - (roi_w/2.))
                right  = int(center_x + (roi_w/2.))
                top    = int(center_y - (roi_h/2.))
                bottom = int(center_y + (roi_h/2.))
                
                if( (final_confidence * best_class_score) > score_threshold):
                    thresholded_predictions.append([[left,top,right,bottom],final_confidence * best_class_score,classes[best_class]])

    # Sort the B-boxes by their final score
    thresholded_predictions.sort(key=lambda tup: tup[1],reverse=True)

    print('Printing {} B-boxes survived after score thresholding:'.format(len(thresholded_predictions)))
    for i in range(len(thresholded_predictions)):
        print('B-Box {} : {}'.format(i+1,thresholded_predictions[i]))

    # Non maximal suppression
    print('Non maximal suppression with iou threshold = {}'.format(iou_threshold))
    nms_predictions = non_maximal_suppression(thresholded_predictions,iou_threshold)

    # Print survived b-boxes
    print('Printing the {} B-Boxes survived after non maximal suppression:'.format(len(nms_predictions)))
    for i in range(len(nms_predictions)):
        print('B-Box {} : {}'.format(i+1,nms_predictions[i]))

    # Draw final B-Boxes and label on input image
    for i in range(len(nms_predictions)):

        color = colors[classes.index(nms_predictions[i][2])]
        best_class_name = nms_predictions[i][2]

        # Put a class rectangle with B-Box coordinates and a class label on the image
        input_image = cv2.rectangle(input_image,(nms_predictions[i][0][0],nms_predictions[i][0][1]),(nms_predictions[i][0][2],nms_predictions[i][0][3]),color)
        cv2.putText(input_image,best_class_name,(int((nms_predictions[i][0][0]+nms_predictions[i][0][2])/2),int((nms_predictions[i][0][1]+nms_predictions[i][0][3])/2)),cv2.FONT_HERSHEY_SIMPLEX,1,color,3)
    
    return input_image