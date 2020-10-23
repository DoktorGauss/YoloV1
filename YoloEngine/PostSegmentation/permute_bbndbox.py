import numpy as np
import itertools
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.patches as patches

def iou(boxA, boxB):
    # f_xmin = first[0]
    # f_ymin = first[1]
    # f_xmax = first[2]
    # f_ymax = first[3]

    # s_xmin = second[0]
    # s_ymin = second[1]
    # s_xmax = second[2]
    # s_ymax = second[3]
 
    # determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou


    

def calculate_possible_permutation(ordinal_image, ordinal_bbox):
    candidats = []
    matrix = np.zeros((ordinal_bbox.shape[0],ordinal_bbox.shape[0]))
    fn = lambda i,j : iou(ordinal_bbox[i], ordinal_bbox[j])
    for i in range(matrix.shape[1]):
        for j in range(matrix.shape[0]):
            matrix[i,j] = fn(i,j)
        if nointersaction(matrix[i, :],i): 
            candidats.append(ordinal_bbox[i,:])
    return len(candidats), np.asarray(candidats), matrix

def nointersaction(row,i):
    i_ = 0
    for element in row:
        if element > 0 and not i_==i:
            return False
        i_ += 1
    return True

def calculate_iou_matrix(ordinal_bbox):
    candidats = []
    matrix = np.zeros((ordinal_bbox.shape[0],ordinal_bbox.shape[0]))
    fn = lambda i,j: iou(ordinal_bbox[i], ordinal_bbox[j])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i,j] = fn(i,j)
    return matrix


#ordinal = senkrecht
def permute_image_by_permutation_matrix(ordinal_image, bndBoxes, iouMatrix):
    bndBoxes_class2 = filter_array(bndBoxes,4,2)
    indicies = np.arange(len(bndBoxes_class2)).reshape((len(bndBoxes_class2),1))
    bndBoxes_class2 = np.append(bndBoxes_class2, indicies, axis=1)
    bndBoxes_without_class2 = not_filter_array(bndBoxes,4,2)


    perm_images = []
    perm_labels = []
    #safe number of permutation (normal n!=n*(n-1)!   here = n*(n-1)-1 probability to repeat: 1/(n-2)!) 
    numberOfPermutations = len(bndBoxes_class2)*(len(bndBoxes_class2)-1) - 1
    for permutation in range(numberOfPermutations):
        perm_bndBoxes = np.random.permutation(bndBoxes_class2)
        perm_image, perm_bndBoxes = permute_image(ordinal_image,bndBoxes_class2,perm_bndBoxes)
        height, width = perm_image.shape[:2]
        old_height, old_width = ordinal_image.shape[:2]
        perm_bndBoxes = createAllBndBoxesnew(perm_bndBoxes,bndBoxes_without_class2)
        resized_perm_bndBoxes = scalemyBoundingboxes(perm_bndBoxes,(width,height),(old_width,old_height))
        pil_image = Image.fromarray(perm_image)
        pil_image = pil_image.resize((ordinal_image.shape[0],ordinal_image.shape[1]))
        resized_perm_image = np.asarray(pil_image,dtype='uint8')
        drawImageWithBndBoxes(resized_perm_image,resized_perm_bndBoxes, bndBoxes,ordinal_image)
        
    return perm_image


def createAllBndBoxesnew(perm_bndBoxes,bndBoxes_without_class2):
    perm_bndBoxes = perm_bndBoxes[:,:5]
    newBoundingBoxes = []
    if len(bndBoxes_without_class2) > 0:
        if bndBoxes_without_class2[0][4] == 0:
            return np.insert(bndBoxes_without_class2,1,perm_bndBoxes,axis=0)
        else:
            return np.concatenate(perm_bndBoxes,bndBoxes_without_class2)
    else:
        return perm_bndBoxes




def filter_array(bndBoxes, index, value):
    newarray = []
    for element in bndBoxes:
        if element[index] == value: newarray.append(element)
    return np.asarray(newarray)

def not_filter_array(bndBoxes, index, value):
    newarray = []
    for element in bndBoxes:
        if not element[index] == value: newarray.append(element)
    return np.asarray(newarray)



def permute_image(ordinal_image, candidats, permutation):
    height = ordinal_image.shape[1]
    width = ordinal_image.shape[0]
    ordinal_image.setflags(write=1)
    newBoundingBoxes = [candidats[0]]
    candidat_image = np.copy(ordinal_image)
    permutation_image = np.copy(ordinal_image)

    absolute_height_difference = 0

    for num_cand in range(1,len(candidats)):
        cand = candidats[num_cand]
        prev_cand = candidats[num_cand-1]
        perm = permutation[num_cand]
        cand_xmin = cand[0]
        cand_ymin = cand[1]
        cand_xmax = cand[2]
        cand_ymax = cand[3]
        prev_cand_xmin = prev_cand[0]
        prev_cand_ymin = prev_cand[1]
        prev_cand_xmax = prev_cand[2]
        prev_cand_ymax = prev_cand[3]
        perm_xmin = perm[0]
        perm_ymin = perm[1]
        perm_xmax = perm[2]
        perm_ymax = perm[3]
        cand_image = candidat_image[cand_ymin:cand_ymax,:,:]
        perm_image = permutation_image[perm_ymin:perm_ymax,:,:]
        cand_mean = np.mean(cand_image[:,cand_xmin:cand_xmax,:].flatten())
        cand_std = np.std(cand_image[:,perm_xmin:perm_xmax,:].flatten())
        current_intersaction_cands_prev = max(prev_cand_ymax - cand_ymin,0)


        #first case: perm < cand
        if cand_ymax - cand_ymin > perm_ymax - perm_ymin:
            current_height_difference =  perm_ymax - perm_ymin - (cand_ymax - cand_ymin)
            newOrdinalImage = np.zeros(
                (absolute_height_difference+height+current_height_difference+current_intersaction_cands_prev, #height
                width, #width
                3), dtype='uint8')
            #insert sofar permutated image 
            newOrdinalImage[:prev_cand_ymax+absolute_height_difference,:,:] = ordinal_image[:prev_cand_ymax+absolute_height_difference,:,:]
            newOrdinalImage[cand_ymin + current_intersaction_cands_prev +absolute_height_difference : cand_ymax + absolute_height_difference +current_intersaction_cands_prev + current_height_difference,:,:] = perm_image
            newOrdinalImage[cand_ymax+absolute_height_difference+current_height_difference+current_intersaction_cands_prev:,:,:] = ordinal_image[cand_ymax+absolute_height_difference:,:,:]
            ordinal_image = newOrdinalImage
            #ordinal_image[cand_ymin+absolute_height_difference: cand_ymin+inserted_height+absolute_height_difference,:,:] = perm_image
            perm_new_xmin = perm_xmin
            perm_new_xmax = perm_xmax
            perm_new_ymin = cand_ymin + current_intersaction_cands_prev +absolute_height_difference 
            perm_new_ymax = cand_ymax + absolute_height_difference +current_intersaction_cands_prev + current_height_difference
            cls_new = perm[4]
            id_new = perm[5]
            newBoundingBoxes.append([perm_new_xmin,perm_new_ymin,perm_new_xmax,perm_new_ymax,cls_new,id_new])
            absolute_height_difference += (current_height_difference+current_intersaction_cands_prev) 
        # second case: fits perfect
        elif cand_ymax - cand_ymin == perm_ymax - perm_ymin:
            ordinal_image[cand_ymin + current_intersaction_cands_prev +absolute_height_difference : cand_ymax+absolute_height_difference+current_intersaction_cands_prev,:,:] = perm_image
            perm_new_xmin = perm_xmin
            perm_new_xmax = perm_xmax
            perm_new_ymin = cand_ymin + current_intersaction_cands_prev +absolute_height_difference 
            perm_new_ymax = cand_ymax + absolute_height_difference +current_intersaction_cands_prev
            cls_new = perm[4]
            id_new = perm[5]
            newBoundingBoxes.append([perm_new_xmin,perm_new_ymin,perm_new_xmax,perm_new_ymax,cls_new,id_new])
        # third case: perm > cand
        else:
            # calculate current height difference
            current_height_difference = perm_ymax - perm_ymin - (cand_ymax - cand_ymin)
            newOrdinalImage = np.zeros(
                (absolute_height_difference+height+current_height_difference+current_intersaction_cands_prev, #height
                width, #width
                3), dtype='uint8')
            # create new image with new fitted shape
            newOrdinalImage[:prev_cand_ymax+absolute_height_difference,:,:] = ordinal_image[:prev_cand_ymax+absolute_height_difference,:,:]
            newOrdinalImage[ cand_ymin + current_intersaction_cands_prev +absolute_height_difference : cand_ymax + absolute_height_difference +current_intersaction_cands_prev + current_height_difference,:,:] = perm_image
            newOrdinalImage[cand_ymax+absolute_height_difference+current_height_difference+current_intersaction_cands_prev:,:,:] = ordinal_image[cand_ymax+absolute_height_difference:,:,:]
            ordinal_image = newOrdinalImage
            perm_new_xmin = perm_xmin
            perm_new_xmax = perm_xmax
            perm_new_ymin = cand_ymin + current_intersaction_cands_prev +absolute_height_difference 
            perm_new_ymax = cand_ymax + absolute_height_difference +current_intersaction_cands_prev + current_height_difference
            cls_new = perm[4]
            id_new = perm[5]
            newBoundingBoxes.append([perm_new_xmin,perm_new_ymin,perm_new_xmax,perm_new_ymax,cls_new,id_new])
            absolute_height_difference += (current_height_difference+current_intersaction_cands_prev) 
    return ordinal_image, np.asarray(newBoundingBoxes)




                            

        



def scalemyBoundingboxes(bbndBox, imageShape, yoloShape):
      xScale = yoloShape[0]/imageShape[0]
      yScale = yoloShape[1]/imageShape[1]
      bbndBox[:,0] = np.int_(bbndBox[:,0] * xScale)
      bbndBox[:,1] = np.int_(bbndBox[:,1] * yScale)
      bbndBox[:,2] = np.int_(bbndBox[:,2] * xScale)
      bbndBox[:,3] = np.int_(bbndBox[:,3] * yScale)
      return bbndBox

def drawImageWithBndBoxes(img,bndboxes):
    # Create figure and axes
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(img)
    for bbox in bndboxes:
        xmin = bbox[0]
        ymin = bbox[1]
        width = bbox[2] - xmin
        height = bbox[3] - ymin
        rect = patches.Rectangle((xmin,ymin),width,height,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    plt.show()