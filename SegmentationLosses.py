import tensorflow as tf
from keras.losses import Loss
from keras.losses import categorical_crossentropy


def focal_crossentropy(y_true, y_pred, gamma=2):
    """
    Args:
        y_true: The ground truth tensor.
        y_pred: The predicted output of the network.
        gamma (int, optional): The focal parameter. Defaults to 2.

    Returns:
        The focal crossentropy loss reduced along the spatial dimensions.
    """
    ce_loss = categorical_crossentropy(y_true, y_pred, axis=-1)
    if gamma==0:
        return tf.reduce_mean(ce_loss, axis=[1,2])
    pt = tf.exp(-ce_loss)
    focal_loss = tf.pow(1 - pt, gamma) * ce_loss
    return tf.reduce_mean(focal_loss, axis=[1,2])


class FocalTverskyLoss(Loss):
    def __init__(self, gamma=4/3, beta=0.7, ignore_class:int=None, class_weights=None):
        """
        Generalization of the Tversky Loss. Computes a Loss vector using the Tversky similarity index for each individual
        class and then the focal parameter is applied raising each Loss value to the power of `1/gamma` thus making classes
        with low loss values contribute less to the overall loss than classes with higher loss values. After the focal
        parameter a final reduction is performed by taking the mean of the contribution of each class.
        ...
        When `gamma = 1` the FTL simplifies to TL (Tversky Loss)
        When `gamma = 1` and `beta = 0.5` the FTL simplifies to DL (Dice Loss)

        Args:
            - `gamma` (float, optional): When `gamma = 1` the FTL simplifies to TL (Tversky Loss). Defaults to `4/3`.
            - `beta` (float, optional): Value for weighing false positives. False negatives are weighted by `1 - beta`.
                When `gamma = 1` and `beta = 0.5` the FTL simplifies to DL (Dice Loss). Defaults to `0.7`.
            - `ignore_class` (int, optional): The ID of a class to be ignored during computation of the loss function.
            - `class_weights` (list, optional): Class weights used for weighing the contribution of the tversky score 
                of each class to the overall tversky score. If this value is `None` the overall tversky score is computed 
                as the mean value of the tversky score of each class, otherwise it is computed as the weighted average 
                of each individual tversky score. Defaults to `None`.
                
        References:
            - [A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation](https://arxiv.org/abs/1810.07842)
        """
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        self.ignore_class = ignore_class    
        self.class_weights = class_weights


    def call(self, y_true:tf.Tensor, y_pred:tf.Tensor):
        '''
        Tensor must have a shape of : `[batch_size, height, width, num_classes]`.
        Calculate the score over the spatial dimensions (height and width) and subtract it from 1.
        -> `[batch_size, num_classes]` then take the mean along the channel axis (last axis)
        
        Averaging over the batch dimension is done automatically by tensorflow.
        '''
        
        intersection = tf.reduce_sum(y_pred * y_true, axis=[1,2])
        false_positives = tf.reduce_sum((1 - y_true) * y_pred, axis=[1,2])
        false_negatives = tf.reduce_sum(y_true * (1 - y_pred), axis=[1,2])
        
        denominator =  intersection + self.beta * false_positives + (1 - self.beta) * false_negatives
        tversky_vector = tf.math.divide_no_nan(intersection, denominator)
        
        if self.ignore_class is not None and self.class_weights is not None:
            raise ValueError('A value was given for both class_weights and ignore_class. Pass a value only for one of them!')
        
        if self.ignore_class is not None:
            class_weights = [1]*tversky_vector.shape[-1]
            class_weights[self.ignore_class] = 0
        else:
            class_weights = None
        
        if self.class_weights is not None:
            class_weights = tf.constant(self.class_weights, dtype=tversky_vector.dtype)
        else:
            class_weights = None
            
        weighted_tversky_vector = self.apply_class_weights(tversky_vector, class_weights)
        tversky_loss_vector = 1 - weighted_tversky_vector
        # raise the tversky loss of each class to the power of 1/gamma
        focal_tversky_loss_vector = tf.pow(tversky_loss_vector, 1./self.gamma)
        
        # return the mean of the loss vector and produce a scalar loss value
        return tf.reduce_mean(focal_tversky_loss_vector, axis=-1)
    
    
    def apply_class_weights(self, tensor, class_weights):
        if class_weights is None:
            pass
        else:
            tensor = tensor * class_weights
            valid_mask = tf.reshape(tf.where(class_weights), [-1])
            tensor = tf.gather(tensor, valid_mask, axis=-1)
            
        return tensor


class TverskyLoss(FocalTverskyLoss):
    def __init__(self, beta=0.7, ignore_class=None, class_weights=None):
        '''
        Uses the Tversky similarity index which is a generalization of the Dice score which allows for flexibility
        in balancing False Positives and False Negatives. With highly imbalanced data and small ROIs, False 
        Negatives detections need to be weighted higher than False Positives to improve recall rate.
        
        When `beta = 0.5` Tversky Loss simplifies to Dice Loss.
        
        Args:
            - `beta` (float, optional): Value for weighing false positives. False negatives are weighted by `1 - beta`.
            - `ignore_class` (int, optional): The ID of a class to be ignored during computation of the loss function.
            - `class_weights` (list, optional): Class weights used for weighing the contribution of the tversky score 
                of each class to the overall tversky score. If this value is `None` the overall tversky score is computed 
                as the mean value of the tversky score of each class, otherwise it is computed as the weighted average 
                of each individual tversky score. Defaults to `None`.
                
        References:
            - [Tversky loss function for image segmentation using 3D fully convolutional deep networks](https://arxiv.org/abs/1706.05721)
        '''
        super().__init__(gamma=1.0, beta=beta, ignore_class=ignore_class, class_weights=class_weights)


class DiceLoss(TverskyLoss):
    def __init__(self, ignore_class=None, class_weights=None):
        """
        In a multi-class scenario, when provided with probability maps on one end and one-hot encoded labels on the
        other, effectively performs multiple two-class problems. `Dice Loss` is computed by calculating the Dice
        score for each individual class, thus creating a Dice score vector with length equal to `num_classes` and 
        then taking the mean or the weighted average of the Dice score vector to produce a scalar `Dice Loss` value.
        Finally the loss is computed as `1 - mean_score`

        Args:
            - `class_weights` (list, optional): Class weights used for weighing the contribution of the dice score 
                of each class to the overall dice score. If this value is `None` the overall dice score is computed 
                as the mean value of the dice score of each class, otherwise it is computed as the weighted average 
                of each individual dice score. Defaults to `None`.
                
        References:
            - [Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations](https://arxiv.org/abs/1707.03237)
        """
        super().__init__(beta=0.5, ignore_class=ignore_class, class_weights=class_weights)


class IoULoss(Loss):
    def __init__(self, class_weights=None):
        """
        Args:
            - `ignore_class` (int, optional): The ID of a class to be ignored during computation of the loss function.
            - `class_weights` (list, optional): Class weights used for weighing the contribution of the IoU score 
                of each class to the overall IoU score. If this value is `None` the overall IoU score is computed 
                as the mean value of the IoU score of each class, otherwise it is computed as the weighted average 
                of each individual IoU score. Defaults to `None`.
        """
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
    def __init__(self, alpha1=1.0, alpha2=1.0, ignore_class=None, class_weights=None):
        """
        Computes a scalar loss value with a formula `Loss = alpha1 * DiceLoss + alpha2 * CrossentropyLoss`.
        The Dice Loss is computed by calculating the Dice score for each individual class, thus creating a 
        Dice score vector with length equal to num_classes and then taking the weighted average of the Dice 
        score vector to produce a scalar dice loss value. The Crossentropy Loss which is computed along the 
        channel axis calculates a value for each individual pixel thus producing an output of shape `(height, width)`
        so a further reduction along both spatial dimensions is needed in order to produce a scalar value so 
        that it can be added to the Dice Loss.

        Args:
            - `alpha1` (float, optional): Value for weighing the contribution of the Dice loss to the overall 
                loss value. Defaults to 1.0.
            - `alpha2` (float, optional): Value for weighing the contribution of the Crossentropy loss to the 
                overall loss value. Defaults to 1.0.
            - `ignore_class` (int, optional): The ID of a class to be ignored during computation of the loss function.
            - `class_weights` (list, optional): Class weights used for weighing the contribution of the dice score 
                of each class to the overall dice score. If this value is `None` the overall dice score is computed 
                as the mean value of the dice score of each class, otherwise it is computed as the weighted average 
                of each individual dice score. Defaults to `None`.
        """
        super().__init__(ignore_class=ignore_class, class_weights=class_weights)
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def call(self, y_true, y_pred):
        return self.alpha1 * super().call(y_true, y_pred) + self.alpha2 * tf.reduce_mean(categorical_crossentropy(y_true, y_pred), axis=[1,2])


class FocalHybridLoss(FocalTverskyLoss):
    def __init__(self, gamma1=4/3, gamma2=2, beta=0.5, alpha1=1.0, alpha2=1.0, ignore_class=None, class_weights=None):
        """
        Focal Hybrid Loss is Hybrid Loss variant with extra focal parameters. In Focal Hybrid Loss, Dice Loss is
        replaced by Focal Tversky Loss and Crossentropy Loss is replaced by Focal Crossentropy Loss.

        Args:
            - `gamma1` (_type_, optional): Focal parameter for the Focal Tversky Loss. Defaults to 4/3.
            - `gamma2` (int, optional): Focal parameter for the Focal Crossentropy Loss. Defaults to 2.
            - `beta` (float, optional): Value for weighing false positives. False negatives are weighted by `1 - beta`.
            - `alpha1` (float, optional): Value for weighing the contribution of the Dice loss to the overall 
                loss value. Defaults to 1.0.
            - `alpha2` (float, optional): Value for weighing the contribution of the Crossentropy loss to the 
                overall loss value. Defaults to 1.0.
            - `ignore_class` (int, optional): The ID of a class to be ignored during computation of the loss function.
            - `class_weights` (list, optional): Class weights used for weighing the contribution of the tversky score 
                of each class to the overall tversky score. If this value is `None` the overall tversky score is computed 
                as the mean value of the tversky score of each class, otherwise it is computed as the weighted average 
                of each individual tversky score. Defaults to `None`.
        """
        
        assert gamma1!=0, 'gamma1 must not be equal to 0'
        
        super().__init__(gamma=gamma1, beta=beta, ignore_class=ignore_class, class_weights=class_weights)
        self.gamma2 = gamma2
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def call(self, y_true, y_pred):
        return self.alpha1 * super().call(y_true, y_pred) + self.alpha2 * focal_crossentropy(y_true, y_pred, self.gamma2)