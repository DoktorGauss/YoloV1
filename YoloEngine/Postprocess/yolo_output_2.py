
import numpy as np
import cv2

class Box:
    def __init__(self):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.c = float()
        self.prob = float()
        self.s1 = int()
        self.s2 = int()

def overlap(x1,w1,x2,w2):
    l1 = x1 - w1 / 2.;
    l2 = x2 - w2 / 2.;
    left = max(l1, l2)
    r1 = x1 + w1 / 2.;
    r2 = x2 + w2 / 2.;
    right = min(r1, r2)
    return right - left;

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w);
    h = overlap(a.y, a.h, b.y, b.h);
    if w < 0 or h < 0: return 0;
    area = w * h;
    return area;

def box_union(a, b):
    i = box_intersection(a, b);
    u = a.w * a.h + b.w * b.h - i;
    return u;

def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b);


def yolo_net_out_to_car_boxes(net_out, threshold = 0.2, sqrt=1.8,C=20, B=2, S=(50,1)):
    class_num = 0
    boxes = []
    SS        =  S[0] * S[1] # number of grid cells
    prob_size = SS * C # class probabilities
    conf_size = SS * B # confidences for each grid cell
    
    probs = net_out[0 : prob_size]
    confs = net_out[prob_size : (prob_size + conf_size)]
    cords = net_out[(prob_size + conf_size) : ]
    probs = probs.reshape([SS, C])
    confs = confs.reshape([SS, B])
    cords = cords.reshape([SS, B, 4])
    
    for grid in range(SS):
        for b in range(B):
            bx   = Box()
            bx.c =  confs[grid, b]
            bx.x = (cords[grid, b, 0] + grid %  S[1]) / S[1]
            bx.y = (cords[grid, b, 1] + grid // S[0]) / S[0]
            bx.w =  cords[grid, b, 2] ** sqrt 
            bx.h =  cords[grid, b, 3] ** sqrt
            bx.s1 = grid %  S[0]
            bx.s2 = int(grid / S[0]) % S[1]
            p = probs[grid, :] * bx.c
            
            if p[class_num] >= threshold:
                bx.prob = p[class_num]
                boxes.append(bx)
                
    # combine boxes that are overlap
    boxes.sort(key=lambda b:b.prob,reverse=True)
    for i in range(len(boxes)):
        boxi = boxes[i]
        if boxi.prob == 0: continue
        for j in range(i + 1, len(boxes)):
            boxj = boxes[j]
            if box_iou(boxi, boxj) >= .1:
                boxes[j].prob = 0.
    boxes = [b for b in boxes if b.prob > 0.]
    
    return boxes

def draw_box(boxes,im):
    imgcv = im
    for b in boxes:
        h, w, _ = imgcv.shape
        
        left  = int ((b.x - b.w/2.) * w)
        right = int ((b.x + b.w/2.) * w)
        top   = int ((b.y - b.h/2.) * h)
        bot   = int ((b.y + b.h/2.) * h)

        # left = int(left*(xmax-xmin)/w + xmin)
        # right = int(right*(xmax-xmin)/w + xmin)
        # top = int(top*(ymax-ymin)/h + ymin)
        # bot = int(bot*(ymax-ymin)/h + ymin)

        if left  < 0    :  left = 0
        if right > w - 1: right = w - 1
        if top   < 0    :   top = 0
        if bot   > h - 1:   bot = h - 1
        thick = int((h + w) // 150)
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
        random = int(np.floor(np.random.random() * len(colors)))
        cv2.rectangle(imgcv, (left, top), (right, bot), colors[random], 1)

    return imgcv