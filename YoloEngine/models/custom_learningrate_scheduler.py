# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 10:11:59 2020

@author: Arbeit
"""
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.python.keras.callbacks import TensorBoard

class CustomLearningRateScheduler(keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

  Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
  """

    def __init__(self, schedule):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_batch_begin(self, batch, logs):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

        # Set the value back to the optimizer before this epoch starts
        #tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        tf.summary.scalar('learning_rate', lr, batch)
        #print("\nEpoch %05d: Learning rate is %6.8f." % (batch, lr))
        
    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch %05d: Learning rate is %6.8f." % (epoch, scheduled_lr))


LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    # (0, 0.0001),
    # (1, 0.00001),
    # (20,   0.000001),
    # (40,  0.0000005), #30
    # (70,  0.0000001),
    # (100,  0.00000005),
    # (120,  0.00000001),
    (0,  0.00013),
    (1,  0.00011),
    (3,  0.00010),
    (6,  0.00009),
    (10, 0.00008),
    (15, 0.00007),
    (21, 0.00005),
    (28, 0.00001),
    (100,0.000001),
    (150,0.0000001),
    (200,0.00000005),
    (250,0.00000003),
    (300,0.00000002),
    (400,0.000000001)
]


def lr_schedule(epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr




class XTensorBoard(TensorBoard):
    # def on_epoch_begin(self, epoch, logs=None):
    #     # get values
    #     lr = float(K.get_value(self.model.optimizer.lr))
    #     decay = float(K.get_value(self.model.optimizer.decay))
    #     # computer lr
    #     lr = lr * (1. / (1 + decay * epoch))
    #     K.set_value(self.model.optimizer.lr, lr)

    def on_batch_end(self, batch, logs):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        super().on_epoch_end(batch, logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        super().on_epoch_end(epoch, logs)