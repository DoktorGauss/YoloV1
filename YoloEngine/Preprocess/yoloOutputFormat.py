import numpy as np
from operator import itemgetter
from PIL import Image

def convert_data_into_YOLO(
    label_data = None, #nparray of label data
    image_data = None, # np array of image data
    classes = None,
    inputShape=(448,610,3),  #inputshape of image
    S=(50,1), # segmentation
    B=1, # number of prediction in each segment
    C=4, # number of classes we try to predict in each segment
    message=''):
    yolo_label_data = prepareLabelData(label_data,classes, inputShape, S,B,C)
    print(message + 'yolo_label_data converted')
    yolo_image_data = prepareImageData(image_data, inputShape)
    print(message + 'yolo_image_data converted')
    return yolo_label_data, yolo_image_data

def prepareLabelData(
    label_data = None,
    classes = None,
    inputShape=(448,610,3),
    S=(50,1),
    B=1,
    C=4
    ):
    yolo_label_data = []
    for data in label_data:
        # get the real shape of the data image
        imageShape = (int(data['metric']['width']),int(data['metric']['height']),3)
        # rescale data to the inputShape
        data = data['data']
        rescaled_data = rescaleData(data, imageShape, inputShape)
        # prepare clusters for inputShape
        clusters = prepareClusters(S[0], S[1] ,inputShape)
        label_matrix = np.zeros([S[0], S[1], B*5 + C])
        for l in rescaled_data:
            xmin = int(l['xmin'])
            ymin = int(l['ymin'])
            xmax = int(l['xmax'])
            ymax = int(l['ymax'])
            cls = getIndexByName(l['name'],classes)
            x = (xmin + xmax) / 2 / inputShape[0]
            y = (ymin + ymax) / 2 / inputShape[1]
            w = (xmax - xmin) / inputShape[0]
            h = (ymax - ymin) / inputShape[1]
            loc = [S[1] * x, S[0] * y]
            loc_i = int(loc[1])
            loc_j = int(loc[0])
            y = loc[1] - loc_i
            x = loc[0] - loc_j

            if label_matrix[loc_i, loc_j, C+4] == 0:
                label_matrix[loc_i, loc_j, cls] = 1
                label_matrix[loc_i, loc_j, C:C+4] = [x, y, w, h]
                label_matrix[loc_i, loc_j, C+B*4] = 1  # response
        yolo_label_data.append(label_matrix)
    return yolo_label_data

def prepareImageData(image_data = None, inputShape=(448,610,3)):
    yolo_images = []
    for image in image_data:
        yolo_images.append(rescaleImage(image, inputShape))
    print('images yolo rescaled')
    return yolo_images

def pilToNumpy(img):
    return np.array(img)

def NumpyToPil(img):
    return Image.fromarray(img)

def rescaleImage(image, inputShape):
    pil_image = NumpyToPil(image)
    img = pil_image.resize(inputShape[:2], Image.ANTIALIAS)
    return pilToNumpy(img);

def rescaleData(data=None, imageShape=(940,1280,3), inputShape=(448,610,3)):
    xScale = inputShape[0]/imageShape[0]
    yScale = inputShape[1]/imageShape[1]
    # save the current values as old values
    for label_data in data:
        label_data['xmax_old'] = label_data['xmax']
        label_data['xmin_old'] = label_data['xmin']
        label_data['ymin_old'] = label_data['ymin']
        label_data['ymax_old'] = label_data['ymax']
        label_data['xmax'] = int(xScale * float(label_data['xmax']))
        label_data['xmin'] = int(xScale * float(label_data['xmin']))
        label_data['ymin'] = int(yScale * float(label_data['ymin']))
        label_data['ymax'] = int(yScale * float(label_data['ymax']))
    return data

def prepareClusters(sy = 50, sx = 1 ,inputShape = (448,610,3)):
    width = inputShape[0]
    height = inputShape[1]
    clusters = []
    cluster_width = width/sx
    cluster_height = height/sy
    for x in range(sx):
        for y in range(sy):
            clusters.append({ 
                'id': (int(y),int(x)), #position in 3d tuple (row, height)
                'b' : 0, 
                'x' : (x)*cluster_width + cluster_width/2, 
                'y' : (y)*cluster_height + cluster_height/2,
                'width' : cluster_width,
                'height' : cluster_height})
    return clusters

def prepareData(data):
    # calculate the mid (x,y) & w,h of each  bndBox an
    data['x'] = (float(data['xmax'])-float(data['xmin']))/2 + float(data['xmin'])
    data['y'] = (float(data['ymax'])-float(data['ymin']))/2 + float(data['ymin'])
    data['width'] = float(data['xmax'])-float(data['xmin'])
    data['height'] = float(data['ymax'])-float(data['ymin'])
    return data

def calculate_distances_eucl(data, clusters):
    distances = []
    for cluster in clusters:
        xw = np.abs(float(cluster['x']) - float(data['x']))
        yw = np.abs(float(cluster['y']) - float(data['y']))
        distance = np.sqrt(xw**2 + yw**2)
        distances.append({'distance': distance, 'cluster' : cluster})
    return sorted(distances, key=itemgetter('distance'))

def calculate_nearest_cluster(data, clusters):
    distances = calculate_distances_eucl(data, clusters)
    return distances[0]['cluster']

def prepareDataToYOLOClusterCalc(clusters, data, inputShape= (448,610,3), S=(50,1)):
    data = prepareData(data)
    nearest_cluster = calculate_nearest_cluster(data, clusters)
    data['x_old'] = data['x']
    data['y_old'] = data['y']
    data['width_old'] = data['width']
    data['height_old'] = data['height']
    data['nearest_cluster_x'] = nearest_cluster['x']
    data['nearest_cluster_y'] = nearest_cluster['y']
    data = calculateRelativePositionsToClusters(data,S, inputShape)
    data['id'] = nearest_cluster['id']
    return nearest_cluster, data


def calculateRelativePositionsToClusters(data, S,inputShape):
    image_w = inputShape[0]
    image_h = inputShape[1]
    sx = S[1]
    sy = S[0]

    data['x_old'] = data['x']
    data['y_old'] = data['y']
    data['width_old'] = data['width']
    data['height_old'] = data['height']

    xmin = data['xmin']
    xmax = data['xmax']
    ymin = data['ymin']
    ymax = data['ymax']

    x = (xmin + xmax) / 2 / image_w
    y = (ymin + ymax) / 2 / image_h
    w = (xmax - xmin) / image_w
    h = (ymax - ymin) / image_h

    loc = [sx * x, sy * y]
    loc_i = int(loc[1])
    loc_j = int(loc[0])

    data['x'] = loc[0] - loc_j
    data['y'] = loc[1] - loc_i
    data['width'] = w
    data['height'] = h
    data['position'] = [loc_i, loc_j]
    return data

def isValidClass(className, classes):
    return className in classes


def getIndexByName(name,classes):
    return classes.index(name)

def getTypeAsTuple(name,classes):
    tuple = np.zeros((len(classes)))
    index = classes.index(name)
    tuple[index] = 1
    return tuple
    #if name == 'UnternehmenHauptData': return [1,0,0,0]
    #elif name == 'BonHauptData':       return [0,1,0,0]
    #elif name == 'BonPosition':        return [0,0,1,0]
    #elif name == 'Zahlung':            return [0,0,0,1]
    ##elif name == 'Umsatzsteuer':       return [0,0,0,0,1,0,0,0]
    ##elif name == 'Transaktionsdaten':  return [0,0,0,0,0,1,0,0]
    ##elif name == 'Werbung':            return [0,0,0,0,0,0,1,0]
    ##elif name == 'Rabatt':             return [0,0,0,0,0,0,0,1]
    #else:                              return [0,0,0,0]
