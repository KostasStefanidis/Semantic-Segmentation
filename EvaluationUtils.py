import tensorflow as tf
#from keras.metrics import base_metric # for tf 2.10
from keras.metrics import Metric # for tf 2.7
from keras import backend
import numpy as np
from keras.losses import Loss


class IoULoss(Loss):
    def __init__(self, class_weights=None):
        super().__init__()
        if class_weights is not None:
            class_weights = list(class_weights)
            self.class_weights = tf.convert_to_tensor(class_weights)
        else:
            self.class_weights = tf.ones(1)
        
    def call(self, y_true, y_pred):
        intersection = tf.reduce_sum(y_pred * y_true, axis=[1,2])
        denominator =  tf.reduce_sum(y_pred, axis=[1,2]) + tf.reduce_sum(y_true, axis=[1,2]) - intersection
        iou_vector = intersection/denominator
        # weigh the score of each class
        iou_vector = iou_vector * self.class_weights
        iou_score = tf.reduce_mean(iou_vector, axis=-1)
        return 1 - iou_score


class MeanIoU(Metric):
    def __init__(self,
                 num_classes:int,
                 name=None,
                 dtype=None,
                 ignore_class: int = None,
                 sparse_y_true: bool = False,
                 sparse_y_pred: bool = False,
                 axis=-1):
        
        super(MeanIoU, self).__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        self.ignore_class = ignore_class
        self.sparse_y_true = sparse_y_true
        self.sparse_y_pred = sparse_y_pred
        self.axis = axis

        # Variable to accumulate the predictions in the confusion matrix.
        self.total_cm = self.add_weight(
            'total_confusion_matrix',
            shape=(num_classes, num_classes),
            initializer=tf.compat.v1.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the confusion matrix statistics.
        Args:
        y_true: The ground truth values.
        y_pred: The predicted values.
        sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.
        Returns:
        Update op.
        """

        if not self.sparse_y_true:
            y_true = tf.argmax(y_true, axis=self.axis)
        if not self.sparse_y_pred:
            y_pred = tf.argmax(y_pred, axis=self.axis)
        
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)

        # Flatten the input if its rank > 1.
        if y_pred.shape.ndims > 1:
            y_pred = tf.reshape(y_pred, [-1])
        if y_true.shape.ndims > 1:
            y_true = tf.reshape(y_true, [-1])

        if self.ignore_class is not None:
            ignore_class = tf.cast(self.ignore_class, y_true.dtype)
            valid_mask = tf.not_equal(y_true, ignore_class)
            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]
        
        # Accumulate the prediction to current confusion matrix.
        current_cm = tf.math.confusion_matrix(
            y_true,
            y_pred,
            self.num_classes,
            weights=sample_weight,
            dtype=self._dtype)
        return self.total_cm.assign_add(current_cm)


    def result(self):
        """Compute the mean intersection-over-union via the confusion matrix."""
        sum_over_row = tf.cast(
            tf.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
        sum_over_col = tf.cast(
            tf.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
        true_positives = tf.cast(
            tf.linalg.tensor_diag_part(self.total_cm), dtype=self._dtype)

        # sum_over_row + sum_over_col =
        #     2 * true_positives + false_positives + false_negatives.
        denominator = sum_over_row + sum_over_col - true_positives

        # The mean is only computed over classes that appear in the
        # label or prediction tensor. If the denominator is 0, we need to
        # ignore the class.
        num_valid_entries = tf.reduce_sum(
            tf.cast(tf.not_equal(denominator, 0), dtype=self._dtype))

        iou = tf.math.divide_no_nan(true_positives, denominator)

        return tf.math.divide_no_nan(
            tf.reduce_sum(iou, name='mean_iou'), num_valid_entries)

    def reset_state(self):
        backend.set_value(
            self.total_cm, np.zeros((self.num_classes, self.num_classes))
            )

    def get_config(self):
        config = {'num_classes': self.num_classes}
        base_config = super(MeanIoU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def get_confusion_matrix(self):
        return self.total_cm.numpy()    
    

class ConfusionMatrix():
    def __init__(self, num_classes):
        self.total_cm = tf.zeros(shape=(num_classes, num_classes))
        self.num_classes = num_classes
        
    def update(self, y_true, y_pred):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1) #self.axis
        
        # Flatten the input if its rank > 1.
        if y_pred.shape.ndims > 1:
            y_pred = tf.reshape(y_pred, [-1])

        if y_true.shape.ndims > 1:
            y_true = tf.reshape(y_true, [-1])
        
        current_cm = tf.math.confusion_matrix(y_true,
                                              y_pred,
                                              self.num_classes,
                                              )
    
        self.total_cm.assign_add(current_cm)

    def __str__(self):
        print(self.total_cm.numpy())