import tensorflow as tf
import keras.backend as K

def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    from PIL import Image
    height, width, channel = tensor.shape
    image = Image.fromarray((tensor * 255).astype(np.uint8))
    return np.asarray(image,dtype='uint8')

class TensorBoardImage(keras.callbacks.Callback):
    def __init__(self, tag,validationset):
        super(TensorBoardImage, self).__init__()
        self.tag = tag
        self.validationset = validationset
        self.targets = []  # collect y_true batches
        self.outputs = []  # collect y_pred batches

        # the shape of these 2 variables will change according to batch shape
        # to handle the "last batch", specify `validate_shape=False`
        self.var_y_true = tf.Variable(0., validate_shape=False)
        self.var_y_pred = tf.Variable(0., validate_shape=False)

    def on_epoch_begin(self, epoch, logs={}):
        count = 0
        images, y_val = self.validationset.__getitem__(0)
        for image in images:
          normimage = image.copy()
          y_pred = self.model.predict(np.reshape(image,(-1,448,448,3)))
          bboxes = decode_netout(y_pred,20)
          image = make_image(image)
          image = draw_boxes(image,bboxes)
          #plt.imshow(image)
          #plt.show()
          print(bboxes)
          writer = tf.summary.create_file_writer('./logs/myimages')
          with writer.as_default():
              # other model code would go here
              tf.summary.image("my_metric"+str(count), tf.image.rgb_to_grayscale([normimage], name='test'), step=epoch)
              writer.flush()

          
          count += 1

        return