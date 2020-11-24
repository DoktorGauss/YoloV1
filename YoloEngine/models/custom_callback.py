
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras as keras

class CustomCallback(keras.callbacks.Callback):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.slidesize = 5
        self.losses = tf.reshape(tf.Tensor([]), (-1,self.slidesize))
        self.xy_losses = tf.reshape(tf.Tensor([]), (-1,self.slidesize))
        self.wh_losses = tf.reshape(tf.Tensor([]), (-1,self.slidesize))
        self.true_object_confidence_losses = tf.reshape(tf.Tensor([]), (-1,self.slidesize))
        self.false_object_confidence_losses =tf.reshape(tf.Tensor([]), (-1,self.slidesize))
        self.class_confidence_losses = tf.reshape(tf.Tensor([]), (-1,self.slidesize))







    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("\n Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):


        keys = list(logs.keys())
        print("\n Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print("\n Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        # if epoch == 1:
        #     print "in model loss weight set"
        #     self.alpha = self.alpha * 0.0
        #     self.beta = self.beta + 1.0
        #     print (epoch, K.get_value(self.alpha), K.get_value(self.beta))
        #     model.compile(optimizer=self.optimizer,
        #                   loss=self, loss_weights=[self.alpha, self.beta],
        #                   metrics=['accuracy'])

        #     sys.stdout.flush()

        keys = list(logs.keys())
        print("\n End epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        print("\n Start testing; got log keys: {}".format(keys))

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        print("\n Stop testing; got log keys: {}".format(keys))

    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        print("\n Start predicting; got log keys: {}".format(keys))

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        print("\n Stop predicting; got log keys: {}".format(keys))

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("\n ...Training: start of batch {}; got log keys: {}".format(batch, keys))

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
#         tf.summary.scalar('no_object_loss', no_object_loss)
#         tf.summary.scalar('object_loss', object_loss)
#         tf.summary.scalar('class_loss', class_loss)
#         tf.summary.scalar('box_loss_xy', box_loss_xy)
#         tf.summary.scalar('box_loss_wh', box_loss_wh)

        m = tf.keras.metrics.AUC(num_thresholds=3)

        # self.false_object_confidence_losses = kb.concatenate(self.false_object_confidence_losses,kb.reshape(logs["no_object_loss"], (_1,1)))
        # self.true_object_confidence_losses = kb.concatenate(self.true_object_confidence_losses,kb.reshape(logs["object_loss"], (_1,1)))
        # self.class_confidence_losses = kb.concatenate(self.false_object_confidence_losses,kb.reshape(logs["object_loss"], (_1,1)))
        # self.xy_losses = kb.concatenate(self.false_object_confidence_losses,kb.reshape(logs["box_loss_xy"], (_1,1)))
        # self.wh_losses = kb.concatenate(self.false_object_confidence_losses,kb.reshape(logs["box_loss_wh"], (_1,1)))



        # print(
        #     "The average loss for epoch {} is {:7.2f} "
        #     "and mean absolute error is {:7.2f}.".format(
        #         epoch, logs["loss"], logs["mean_absolute_error"]
        #     )




        print("\n ...Training: end of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("\n ...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("\n ...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("\n ...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("\n ...Predicting: end of batch {}; got log keys: {}".format(batch, keys))
        
        
class LossAndErrorPrintingCallback(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        print("\n For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

    def on_test_batch_end(self, batch, logs=None):
        print("\n For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

    def on_epoch_end(self, epoch, logs=None):
        print(
            "\n The average loss for epoch {} is {:7.2f} "
            "\n and mean absolute error is {:7.2f}.".format(
                epoch, logs["loss"], logs["mean_absolute_error"]
            )
        )

