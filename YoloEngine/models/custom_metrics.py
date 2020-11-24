import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as kb


def true_positive_caller(S=(50,1), B=1, C=4):
    def true_positive(y_true, y_pred):
        true_class = y_true[..., : C]  # ? * 50 * 1 * (C*B)
        predict_class = y_pred[..., : C]  # ? * 50 * 1 * (C*B)
        metric = tf.keras.losses.categorical_crossentropy(true_class, predict_class)
        metric_list = kb.flatten(metric)
        return kb.mean(metric_list)
    return true_positive



def auc_metric(S,B,C,inputshape, a_metric = tf.keras.metrics.AUC(num_thresholds=3)):
    def _auc_metric(y_true, y_pred):
        y_true_confs_classes = y_true[...,:C]
        y_true_confs_object = y_true[..., 4*B+C:]
        y_true_confs = kb.concatenate([y_true_confs_classes,y_true_confs_object])
        y_true_confs = kb.reshape(y_true_confs, [S[0]*S[1]*(B+C)])

        y_pred_confs_classes = y_pred[...,:C]
        y_pred_confs_object = y_pred[..., 4*B+C:]
        y_pred_confs = kb.concatenate([y_pred_confs_classes,y_pred_confs_object])
        y_pred_confs = kb.reshape(y_pred_confs, [S[0]*S[1]*(B+C)])

        m = a_metric.update_state(y_true_confs, y_pred_confs)
        return m.outputs

    return _auc_metric

def false_negatives(S,B,C,inputshape, fn_metric = tf.keras.metrics.FalseNegatives()):
    def _false_negatives(y_true, y_pred):
        y_true_confs_classes = y_true[...,:C]
        y_true_confs_object = y_true[..., 4*B+C:]
        y_true_confs = kb.concatenate([y_true_confs_classes,y_true_confs_object])
        y_true_confs = kb.reshape(y_true_confs, [S[0]*S[1]*(B+C)])

        y_pred_confs_classes = y_pred[...,:C]
        y_pred_confs_object = y_pred[..., 4*B+C:]
        y_pred_confs = kb.concatenate([y_pred_confs_classes,y_pred_confs_object])
        y_pred_confs = kb.reshape(y_pred_confs, [S[0]*S[1]*(B+C)])

        m = fn_metric.update_state(y_true_confs, y_pred_confs)
        return m.outputs



    return _false_negatives

def false_positives(S,B,C,inputshape, fp_metric = tf.keras.metrics.FalsePositives()):
    def _false_positives(y_true, y_pred):
        y_true_confs_classes = y_true[...,:C]
        y_true_confs_object = y_true[..., 4*B+C:]
        y_true_confs = kb.concatenate([y_true_confs_classes,y_true_confs_object])
        y_true_confs = kb.reshape(y_true_confs, [S[0]*S[1]*(B+C)])

        y_pred_confs_classes = y_pred[...,:C]
        y_pred_confs_object = y_pred[..., 4*B+C:]
        y_pred_confs = kb.concatenate([y_pred_confs_classes,y_pred_confs_object])
        y_pred_confs = kb.reshape(y_pred_confs, [S[0]*S[1]*(B+C)])
        #kb.print_tensor('y_pred_confs',y_pred_confs[:5])
        #kb.print_tensor('y_true_confs',y_true_confs[:5])


        m = fp_metric.update_state(y_true_confs, y_pred_confs)
        # with sess.as_default():
        #     m.run(session=sess)
        return m.outputs

    return _false_positives


def true_negatives(S,B,C,inputshape, tn_metric = tf.keras.metrics.TrueNegatives()):
    def _true_negatives(y_true, y_pred):
        y_true_confs_classes = y_true[...,:C]
        y_true_confs_object = y_true[..., 4*B+C:]
        y_true_confs = kb.concatenate([y_true_confs_classes,y_true_confs_object])
        y_true_confs = kb.reshape(y_true_confs, [S[0]*S[1]*(B+C)])

        y_pred_confs_classes = y_pred[...,:C]
        y_pred_confs_object = y_pred[..., 4*B+C:]
        y_pred_confs = kb.concatenate([y_pred_confs_classes,y_pred_confs_object])
        y_pred_confs = kb.reshape(y_pred_confs, [S[0]*S[1]*(B+C)])

        m = tn_metric.update_state(y_true_confs, y_pred_confs)
        return m.outputs

    return _true_negatives

def true_positives(S,B,C,inputshape, tn_metric = tf.keras.metrics.TruePositives()):
    def _true_positives(y_true, y_pred):
        y_true_confs_classes = y_true[...,:C]
        y_true_confs_object = y_true[..., 4*B+C:]
        y_true_confs = kb.concatenate([y_true_confs_classes,y_true_confs_object])
        y_true_confs = kb.reshape(y_true_confs, [S[0]*S[1]*(B+C)])

        y_pred_confs_classes = y_pred[...,:C]
        y_pred_confs_object = y_pred[..., 4*B+C:]
        y_pred_confs = kb.concatenate([y_pred_confs_classes,y_pred_confs_object])
        y_pred_confs = kb.reshape(y_pred_confs, [S[0]*S[1]*(B+C)])

        m = tn_metric.update_state(y_true_confs, y_pred_confs)
        return m.outputs

    return _true_positives








class MyTruePositives(tf.keras.metrics.TruePositives):
    def __init__(self, name='MyTruePositives',S=(50,50),B=1,C=1, **kwargs):
      super(MyTruePositives, self).__init__(name=name, **kwargs)
      self.true_positives = self.add_weight(name='tp', initializer='zeros')
      self.S = S
      self.B = B
      self.C = C
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_confs_classes = y_true[...,:self.C]
        y_true_confs_object = y_true[..., 4*self.B+self.C:]
        y_true_confs = kb.concatenate([y_true_confs_classes,y_true_confs_object])
        y_true_confs = kb.reshape(y_true_confs, [self.S[0]*self.S[1]*(self.B+self.C)])

        y_pred_confs_classes = y_pred[...,:self.C]
        y_pred_confs_object = y_pred[..., 4*self.B+self.C:]
        y_pred_confs = kb.concatenate([y_pred_confs_classes,y_pred_confs_object])
        y_pred_confs = kb.reshape(y_pred_confs, [self.S[0]*self.S[1]*(self.B+self.C)])
        super().update_state(y_true_confs,y_pred_confs, None)
    def result(self):
        return super().result()


class MyTrueNegatives(tf.keras.metrics.TrueNegatives):
    def __init__(self, name='MyTrueNegatives',S=(50,50),B=1,C=1, **kwargs):
      super(MyTruePositives, self).__init__(name=name, **kwargs)
      self.true_positives = self.add_weight(name='tp', initializer='zeros')
      self.S = S
      self.B = B
      self.C = C
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_confs_classes = y_true[...,:self.C]
        y_true_confs_object = y_true[..., 4*self.B+self.C:]
        y_true_confs = kb.concatenate([y_true_confs_classes,y_true_confs_object])
        y_true_confs = kb.reshape(y_true_confs, [self.S[0]*self.S[1]*(self.B+self.C)])

        y_pred_confs_classes = y_pred[...,:self.C]
        y_pred_confs_object = y_pred[..., 4*self.B+self.C:]
        y_pred_confs = kb.concatenate([y_pred_confs_classes,y_pred_confs_object])
        y_pred_confs = kb.reshape(y_pred_confs, [self.S[0]*self.S[1]*(self.B+self.C)])
        super().update_state(y_true_confs,y_pred_confs, None)
    def result(self):
        return super().result()

class MyFalsePositives(tf.keras.metrics.FalsePositives):
    def __init__(self, name='MyFalsePositives',S=(50,50),B=1,C=1, **kwargs):
      super(MyTruePositives, self).__init__(name=name, **kwargs)
      self.true_positives = self.add_weight(name='tp', initializer='zeros')
      self.S = S
      self.B = B
      self.C = C
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_confs_classes = y_true[...,:self.C]
        y_true_confs_object = y_true[..., 4*self.B+self.C:]
        y_true_confs = kb.concatenate([y_true_confs_classes,y_true_confs_object])
        y_true_confs = kb.reshape(y_true_confs, [self.S[0]*self.S[1]*(self.B+self.C)])

        y_pred_confs_classes = y_pred[...,:self.C]
        y_pred_confs_object = y_pred[..., 4*self.B+self.C:]
        y_pred_confs = kb.concatenate([y_pred_confs_classes,y_pred_confs_object])
        y_pred_confs = kb.reshape(y_pred_confs, [self.S[0]*self.S[1]*(self.B+self.C)])
        super().update_state(y_true_confs,y_pred_confs, None)
        
    def result(self):
        return super().result()

class MyFalseNegatives(tf.keras.metrics.FalseNegatives):
    def __init__(self, name='MyFalseNegatives',S=(50,50),B=1,C=1, **kwargs):
      super(MyTruePositives, self).__init__(name=name, **kwargs)
      self.true_positives = self.add_weight(name='tp', initializer='zeros')
      self.S = S
      self.B = B
      self.C = C
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_confs_classes = y_true[...,:self.C]
        y_true_confs_object = y_true[..., 4*self.B+self.C:]
        y_true_confs = kb.concatenate([y_true_confs_classes,y_true_confs_object])
        y_true_confs = kb.reshape(y_true_confs, [self.S[0]*self.S[1]*(self.B+self.C)])

        y_pred_confs_classes = y_pred[...,:self.C]
        y_pred_confs_object = y_pred[..., 4*self.B+self.C:]
        y_pred_confs = kb.concatenate([y_pred_confs_classes,y_pred_confs_object])
        y_pred_confs = kb.reshape(y_pred_confs, [self.S[0]*self.S[1]*(self.B+self.C)])
        super().update_state(y_true_confs,y_pred_confs, None)
    def result(self):
        return super().result()

class MyAUC(tf.keras.metrics.AUC):
    def __init__(self, name='MyAUC',S=(50,50),B=1,C=1, **kwargs):
      super(MyTruePositives, self).__init__(name=name, **kwargs)
      self.true_positives = self.add_weight(name='tp', initializer='zeros')
      self.S = S
      self.B = B
      self.C = C
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_confs_classes = y_true[...,:self.C]
        y_true_confs_object = y_true[..., 4*self.B+self.C:]
        y_true_confs = kb.concatenate([y_true_confs_classes,y_true_confs_object])
        y_true_confs = kb.reshape(y_true_confs, [self.S[0]*self.S[1]*(self.B+self.C)])

        y_pred_confs_classes = y_pred[...,:self.C]
        y_pred_confs_object = y_pred[..., 4*self.B+self.C:]
        y_pred_confs = kb.concatenate([y_pred_confs_classes,y_pred_confs_object])
        y_pred_confs = kb.reshape(y_pred_confs, [self.S[0]*self.S[1]*(self.B+self.C)])
        super().update_state(y_true_confs,y_pred_confs, None)
    def result(self):
        return super().result()



# class MyTrueNegatives(tf.keras.metrics.):
#     def __init__(self, name='MyTruePositives',S=(50,50),B=1,C=1, **kwargs):
#       super(MyTruePositives, self).__init__(name=name, **kwargs)
#     #   self.true_positives = self.add_weight(name='tp', initializer='zeros')
#       self.S = S
#       self.B = B
#       self.C = C
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_true_confs_classes = y_true[...,:self.C]
#         y_true_confs_object = y_true[..., 4*self.B+self.C:]
#         y_true_confs = kb.concatenate([y_true_confs_classes,y_true_confs_object])
#         y_true_confs = kb.reshape(y_true_confs, [self.S[0]*self.S[1]*(self.B+self.C)])

#         y_pred_confs_classes = y_pred[...,:self.C]
#         y_pred_confs_object = y_pred[..., 4*self.B+self.C:]
#         y_pred_confs = kb.concatenate([y_pred_confs_classes,y_pred_confs_object])
#         y_pred_confs = kb.reshape(y_pred_confs, [self.S[0]*self.S[1]*(self.B+self.C)])
#         super().update_state(y_true_confs,y_pred_confs, None)
#     def result(self):
#         return super().result()