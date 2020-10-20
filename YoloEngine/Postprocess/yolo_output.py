import numpy as np
import tensorflow as tf

def getTypeAsTuple(name):
    if name == 'UnternehmenHauptData': return [1,0,0,0]
    elif name == 'BonHauptData':       return [0,1,0,0]
    elif name == 'BonPosition':        return [0,0,1,0]
    elif name == 'Zahlung':            return [0,0,0,1]
    #elif name == 'Umsatzsteuer':       return [0,0,0,0,1,0,0,0]
    #elif name == 'Transaktionsdaten':  return [0,0,0,0,0,1,0,0]
    #elif name == 'Werbung':            return [0,0,0,0,0,0,1,0]
    #elif name == 'Rabatt':             return [0,0,0,0,0,0,0,1]
    else:                              return [0,0,0,0]

def getTupleAsType(T):
    if T ==   [1,0,0,0]:    return 'UnternehmenHauptData'
    elif T == [0,1,0,0]:    return 'BonHauptData'
    elif T == [0,0,1,0]:    return 'BonPosition'
    elif T == [0,0,0,1]:    return 'Zahlung'
    #elif T == [0,0,0,0,1,0,0,0]:    return 'Umsatzsteuer'
    #elif T == [0,0,0,0,0,1,0,0]:    return 'Transaktionsdaten'
    #elif T == [0,0,0,0,0,0,1,0]:    return 'Werbung'
    #elif T == [0,0,0,0,0,0,0,1]:    return 'Rabatt'
    else:                           return ''
    

def calculateClusterFromId(input, cluster_id,S):
    height = len(input)
    width = len(input[len(input)-1])
    cluster_height = height / S[0]
    cluster_width = width / S[1]
    x = cluster_id[1]*cluster_width + cluster_width/2
    y = cluster_id[0]*cluster_height + cluster_height/2
    return {'x': x, 'y':y , 'width': cluster_width, 'height': cluster_height}
    
    
    
def getNameById(id):
    if   id ==   0:    return 'UnternehmenHauptData'
    elif id == 1:    return 'BonHauptData'
    elif id== 2:    return 'BonPosition'
    elif id== 3:    return 'Zahlung'
    #elif id== 4:    return 'Umsatzsteuer'
    #elif id== 5:    return 'Transaktionsdaten'
    #elif id== 6:    return 'Werbung'
    #elif id== 7:    return 'Rabatt'
    else:                           return ''


def outputToBoundingBox(output, input, S=(50,1), C=4, B=1, threshold_object= 0.7):
    image_height = len(input)
    image_width = len(input[len(input)-1])
    outputToBoundingBox = []
    for s1 in range(S[0]):
        for s2 in range(S[1]):
            for b in range(B):
                start3dPosition_ClassProbability = b*C
                start3dPosition_Boxes = B*C+b*4
                start3dPosition_Prob = C*B+B*4 + b
                if(output[s1][s2][start3dPosition_Prob] > threshold_object):
                    prob = output[s1][s2][start3dPosition_Prob]
                    x = output[s1][s2][start3dPosition_Boxes]
                    y = output[s1][s2][start3dPosition_Boxes + 1]
                    w = output[s1][s2][start3dPosition_Boxes + 2]
                    h = output[s1][s2][start3dPosition_Boxes + 3]
                    class_prob = output[s1][s2][start3dPosition_ClassProbability:start3dPosition_ClassProbability + C]
                    class_with_prob = calculateClassProb(class_prob)
                    cluster = calculateClusterFromId(input, (s1,s2),S)
                    outputToBoundingBox.append({ 'id': (s1,s2), 
                      'x': x*cluster['width'] - cluster['width']/2 + cluster['x'],
                      'y': y*cluster['height'] -  cluster['height']/2 + cluster['y'], 
                      'width': w * image_width, 
                      'height' : h * image_height, 
                      'cluster_x': cluster['x'], 
                      'cluster_y': cluster['y'], 
                      'cluster_width': cluster['width'], 
                      'cluster_height': cluster['height'],
                      'object_probability': prob,
                      'class_probability': class_with_prob['probability'],
                      'class_name': class_with_prob['class_name']})
    return outputToBoundingBox
