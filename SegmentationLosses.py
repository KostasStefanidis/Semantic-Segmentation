import tensorflow as tf
from keras.losses import Loss
from keras.losses import categorical_crossentropy


def focal_crossentropy(y_true, y_pred, gamma=2):
        '''This is the focal crossentropy'''
        ce_loss = categorical_crossentropy(y_true, y_pred, axis=-1)
        if gamma==0:
            return tf.reduce_mean(ce_loss, axis=[1,2])
        pt = tf.exp(-ce_loss)
        focal_loss = tf.pow(1 - pt, gamma) * ce_loss
        return tf.reduce_mean(focal_loss, axis=[1,2])


class FocalTverskyLoss(Loss):
    def __init__(self, gamma=4/3, beta=0.7, class_weights=None):
        ''' In a multi-class scenario, when provided with probability maps on one end
        and one-hot encoded labels on the other, effectively performs multiple two-class problems
        calculate a vector of scores, one score per class and take its mean. Finally the loss is 
        computed as `1 - mean_score`
        
        Setting `gamma = 1` the FTL simplifies to TL (Tversky Loss)
        
        Setting `gamma = 1` and `beta = 0.5` the FTL simplifies to DL (Dice Loss)
        '''
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        self.class_weights = class_weights


    def call(self, y_true, y_pred):
        ''' Tensor must have a shape of : `[batch_size, height, width, num_classes]`.
        Calculate the score over the spatial dimensions height and width and subtract it from 1.
        -> `[batch_size, num_classes]` then take the mean along the channel axis (last axis)
        
        Averaging over the batch dimension is done automatically by tensorflow.
        '''
        intersection = tf.reduce_sum(y_pred * y_true, axis=[1,2])
        false_positives = tf.reduce_sum((1 - y_true) * y_pred, axis=[1,2])
        false_negatives = tf.reduce_sum(y_true * (1 - y_pred), axis=[1,2])
        
        denominator =  intersection + self.beta * false_positives + (1 - self.beta) * false_negatives
        tversky_vector = tf.math.divide_no_nan(intersection, denominator)
        
        if self.class_weights is not None:
            tversky_vector = tversky_vector * self.class_weights
        
        tversky_loss_vector = 1 - tversky_vector
        # raise the tversky loss of each class to the power of 1/gamma
        focal_tversky_loss_vector = tf.pow(tversky_loss_vector, 1./self.gamma)
        
        # return the mean of the loss vector and produce a scalar loss value
        return tf.reduce_mean(focal_tversky_loss_vector, axis=-1)


class TverskyLoss(FocalTverskyLoss):
    def __init__(self, beta=0.7, class_weights=None):
        '''
        Uses the Tversky similarity index which is a generalization 
        of the Dice score which allows for flexibility in balancing 
        False Positives and False Negatives. With highly imbalanced data 
        and small ROIs, False Negatives detections need to be weighted 
        higher than False Positives to improve recall rate.
        
        Setting `beta = 0.5` simplifies to Dice Loss.
        '''
        super().__init__(gamma=1.0, beta=beta, class_weights=class_weights)


class DiceLoss(TverskyLoss):
    def __init__(self, class_weights=None):
        '''
        '''
        super().__init__(beta=0.5, class_weights=class_weights)


class IoULoss(Loss):
    def __init__(self, class_weights=None):
        '''
        '''
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


class HybridLoss(DiceLoss):
    def __init__(self, class_weights=None):
        super().__init__(class_weights=class_weights)

    def call(self, y_true, y_pred):
        return super().call(y_true, y_pred) + tf.reduce_mean(categorical_crossentropy(y_true, y_pred), axis=[1,2])


class FocalHybridLoss(FocalTverskyLoss):
    def __init__(self, gamma=4/3, beta=0.7, class_weights=None):
        super().__init__(gamma=gamma, beta=beta, class_weights=class_weights)

    def call(self, y_true, y_pred):
        return super().call(y_true, y_pred) + focal_crossentropy(y_true, y_pred)