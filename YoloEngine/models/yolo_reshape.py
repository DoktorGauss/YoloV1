import tensorflow as tf
import tensorflow.keras.backend as kb


class Yolo_Reshape(tf.keras.layers.Layer):
  def __init__(self, target_shape, B=2, C=4):
    super(Yolo_Reshape, self).__init__()
    self.target_shape = tuple(target_shape)
    self.B = B
    self.C = C
    

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'target_shape': self.target_shape
    })
    return config

  def call(self, input):
    # grids 7x7
    S = [self.target_shape[0], self.target_shape[1]]
    # classes
    C = self.C
    # no of bounding boxes per grid
    B = self.B

    idx1 = S[0] * S[1] * C
    idx2 = idx1 + S[0] * S[1] * B
    
    # class probabilities

    class_probs = kb.reshape(input[:, :idx1], (kb.shape(input)[0],) + tuple([S[0], S[1], C]))
    class_probs = kb.sigmoid(class_probs)
    class_probs = kb.softmax(class_probs)
    #box_xy = K.sigmoid(feats[..., :2])
    #box_wh = K.exp(feats[..., 2:4])
    #box_confidence = K.sigmoid(feats[..., 4:5])
    #box_class_probs = K.softmax(feats[..., 5:])
    #confidence
    confs = kb.reshape(input[:, idx1:idx2], (kb.shape(input)[0],) + tuple([S[0], S[1], B]))
    confs = kb.sigmoid(confs)

    # boxes
    boxes = kb.reshape(input[:, idx2:], (kb.shape(input)[0],) + tuple([S[0], S[1], B * 4]))
    boxes = kb.sigmoid(boxes)

    outputs = kb.concatenate([class_probs,boxes,confs])
    return outputs

