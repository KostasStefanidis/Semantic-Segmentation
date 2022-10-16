import tensorflow as tf
from keras.losses import Loss
from keras.losses import categorical_crossentropy
from keras import backend

class FocalTverskyLoss(Loss):
    def __init__(self, gamma=2, beta=0.7, class_weights=None):
        ''' In a multi-class scenario, when provided with probability maps on one end
        and one-hot encoded labels on the other, effectively performs multiple two-class problems
        calculate a vector of scores, one score per class and take its mean. Finally the loss is 
        computed as `1 - mean_score`
        '''
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        self.class_weights = list(class_weights)

    def call(self, y_true, y_pred):
        ''' Tensor must be in shape : `[batch_size, height, width, num_classes]`.
        Calculate the sum over the spatial dimensions height and width 
        -> `[batch_size, num_clasees]` then take the mean along the channel axis (last axis)
        
        Averaging over batch is done automatically by tensorflow.
        '''

        intersection = tf.reduce_sum(y_pred * y_true, axis=[1,2])
        false_positives = tf.reduce_sum((1 - y_true) * y_pred, axis=[1,2])
        false_negatives = tf.reduce_sum(y_true * (1 - y_pred), axis=[1,2])
        denominator =  intersection + self.beta * false_positives + (1 - self.beta) * false_negatives
        tversky_vector = tf.math.divide_no_nan(intersection, denominator)
        if self.class_weights is not None:
            tversky_vector = tversky_vector * self.class_weights  
        tversky_score = tf.reduce_mean(tversky_vector, axis=-1)
        return tf.pow(1 - tversky_score, 1/self.gamma)

class TverskyLoss(FocalTverskyLoss):
    def __init__(self, beta=0.7, class_weights=None):
        super().__init__(gamma=1, beta=beta, class_weights=class_weights)


class DiceLoss(TverskyLoss):
    def __init__(self, class_weights=None):
        super().__init__(beta=0.5, class_weights=class_weights)


class IoULoss(Loss):
    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights
        
    def call(self, y_true, y_pred):
        intersection = tf.reduce_sum(y_pred * y_true, axis=[1,2])
        denominator =  tf.reduce_sum(y_pred, axis=[1,2]) + tf.reduce_sum(y_true, axis=[1,2]) - intersection
        iou_vector = tf.math.divide_no_nan(intersection, denominator)
        if self.class_weights is not None:
            iou_vector = iou_vector * self.class_weights
        iou_score = tf.reduce_mean(iou_vector, axis=-1)
        return 1 - iou_score