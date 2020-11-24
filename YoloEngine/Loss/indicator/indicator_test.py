import unittest
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
        idx1 = S[0] * S[1] * C
        idx2 = idx1 + S[0] * S[1] * B
        SS = S[0]*S[1]
        class_probs_pred = np.reshape(y_pred[:idx1], ([S[0], S[1], C]))
        confs_pred = np.reshape(y_pred[idx1:idx2], ([S[0], S[1], B]))
        boxes_pred = np.reshape(y_pred[idx2:], ([S[0], S[1], B * 4]))

        # case 1_obj_i_j
        
        i_th = int(i % S[0])
        j_th = int(int(i / S[0]) % S[1])

        true_confidence = y_true[i_th, j_th, C+B*4]
        true_box = y_true[i_th, j_th, C:C+4]
            # object indicator
        if b == -1 :
            return int(true_confidence) == int(obj)
        else:
            if int(true_confidence) == int(obj):
                if int(obj) == 1:
                    iou_array = np.array([ calc_iou(true_box, boxes_pred[i_th,j_th, box_num*4: (box_num+1)*4],S,i_th,j_th) for box_num in range(B)])
                    highest = np.max(iou_array,axis=0)
                    b_th_prediction = boxes_pred[i_th,j_th, b*4: (b+1)*4]
                    b_th_iou = calc_iou(true_box, b_th_prediction,S,i_th,j_th)
                    if b_th_iou == highest:
                        return True
                    else:
                        return False
                else:
                    return True
            else:
                return False
    return indicator

class Indicator_Test(unittest.TestCase):
    def __init__(self):
        self.S=(50,50)
        self.B=2
        self.C=1
        self.batch_size = 1
        self.Shape=(1024,748,3)
       
    # def test_upper(self):
    #     self.assertEqual('foo'.upper(), 'FOO')

    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())

    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)

    

    def basic_test_noObj(self):
        self.S=(1,1)
        self.B = 2
        self.C = 1
        
        y_true = np.zeros((self.S[0],self.S[1],5*self.B + self.C))

        y_pred = np.zeros(self.S[0]*self.S[1]*(5*self.B + self.C))
        y_pred[0] = 1.0  # P(class_i|obj)
        y_pred[1] = 0.0  # P(obj_1)
        y_pred[2] = 1.0  # P(obj_2)
        y_pred[3] = 0.5
        y_pred[4] = 0.5
        y_pred[5] = 0.2
        y_pred[6] = 0.3
        y_pred[7] = 0.1
        y_pred[8] = 0.2
        y_pred[9] = 0.1
        y_pred[10] = 0.7

        my_indicator_fct = my_indicator(y_true,y_pred, self.S, self.B, self.C)

        b_0 = my_indicator_fct(1,0,0)
        b_1 = my_indicator_fct(1,0,1)
        b_2 = my_indicator_fct(1,0,-1)
        b_3 = my_indicator_fct(0,0,-1)
        b_4 = my_indicator_fct(0,0,0)
        b_5 = my_indicator_fct(0,0,1)
        print(b_0,'b_0: object in cell 0 predictor 0')
        print(b_1,'b_1: object in cell 0 predictor 1')
        print(b_2,'b_2: object in cell 0')
        print(b_3,'b_3: no object in cell 0')
        print(b_4,'b_4: no object in cell 0 predictor 0')
        print(b_5,'b_5: no object in cell 0 predictor 1')


    def basic_test(self):
        self.S=(1,1)
        self.B = 2
        self.C = 1
        
        y_true = np.zeros((self.S[0],self.S[1],5*self.B + self.C))
        y_true[0,0,0] = 1.0 #P(class_i | obj)
        y_true[0,0,1] = 0.5
        y_true[0,0,2] = 0.5
        y_true[0,0,3] = 0.2
        y_true[0,0,4] = 0.3
        y_true[0,0,9] = 1.0  # P(obj)
        
        y_pred = np.zeros(self.S[0]*self.S[1]*(5*self.B + self.C))
        y_pred[0] = 1.0  # P(class_i|obj)
        y_pred[1] = 0.0  # P(obj_1)
        y_pred[2] = 1.0  # P(obj_2)
        y_pred[3] = 0.5
        y_pred[4] = 0.5
        y_pred[5] = 0.2
        y_pred[6] = 0.3
        y_pred[7] = 0.1
        y_pred[8] = 0.2
        y_pred[9] = 0.1
        y_pred[10] = 0.7

        my_indicator_fct = my_indicator(y_true,y_pred, self.S, self.B, self.C)

        b_0 = my_indicator_fct(1,0,0)
        b_1 = my_indicator_fct(1,0,1)
        b_2 = my_indicator_fct(1,0,-1)
        b_3 = my_indicator_fct(0,0,-1)
        b_4 = my_indicator_fct(0,0,0)
        b_5 = my_indicator_fct(0,0,1)
        print(b_0,'b_0: object in cell 0 predictor 0')
        print(b_1,'b_1: object in cell 0 predictor 1')
        print(b_2,'b_2: object in cell 0')
        print(b_3,'b_3: no object in cell 0')
        print(b_4,'b_4: no object in cell 0 predictor 0')
        print(b_5,'b_5: no object in cell 0 predictor 1')




        




unit_test = Indicator_Test()
test = unit_test.basic_test()
test2 = unit_test.basic_test_noObj()