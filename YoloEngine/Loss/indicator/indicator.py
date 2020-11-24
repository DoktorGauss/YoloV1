import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as kb

def overlap(x1,w1,x2,w2):
    l1 = x1 - w1 / 2.;
    l2 = x2 - w2 / 2.;
    left = np.max((l1, l2))
    r1 = x1 + w1 / 2.;
    r2 = x2 + w2 / 2.;
    right = np.min((r1, r2))
    return right - left;

def box_intersection(a, b):
    w = overlap(a[0], a[2], b[0], b[2]);
    h = overlap(a[1], a[3], b[1], b[3]);
    if w < 0 or h < 0: return 0;
    area = w * h;
    return area;

def box_union(a, b):
    i = box_intersection(a, b);
    u = a[2] * a[3] + b[2] * b[3] - i;
    return u;

def calc_iou(a, b,S, ith, jth):
    x1 = (jth + a[0])/S[0]
    y1 = (ith + a[1])/S[1]
    w1 = a[2]**2
    h1 = a[3]**2
    x2 = (jth + b[0])/S[0]
    y2 = (ith + b[1])/S[1]
    w2 = b[2]**2
    h2 = b[3]**2
    value = box_intersection(np.array([x1,y1,w1,h1]), np.array([x2,y2,w2,h2])) / box_union(np.array([x1,y1,w1,h1]), np.array([x2,y2,w2,h2]))
    if np.abs(value - 1) < 0.01: value = 1.0
    return value;

def my_indicator(y_true,y_pred, S,B,C):
    def indicator(obj,i,b):
        # case 1_obj_i_j
        i_th = int(i % S[0])
        j_th = int(int(i / S[0]) % S[1])

        true_confidence = y_true[i_th, j_th, C+B*4]
        true_box = y_true[i_th, j_th, C:C+4]
            # object indicator
        if b == -1 :
            return (tf.greater(true_confidence,0.5)) == (int(obj)==1)
        else:
            if (tf.greater(true_confidence,0.5)) == (int(obj)==1):
                if int(obj) == 1:

                    
                    iou_array = np.array([ calc_iou( true_box, y_pred[i_th,j_th, C+box_num*4: C+(box_num+1)*4] , S, i_th , j_th) for box_num in range(B)])
                    highest = np.max(iou_array,axis=0)
                    index = np.where(iou_array == highest)[0]
                    if index[0] == b:
                        return True
                    else:
                        return False
                else:
                    return True
            else:
                return False
    return indicator