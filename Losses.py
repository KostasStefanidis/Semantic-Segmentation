import tensorflow as tf
from keras import backend as K
from keras.losses import binary_crossentropy, categorical_crossentropy


beta = 0.25
alpha = 0.25
gamma = 2
epsilon = 1e-5
smooth = 1

class Semantic_loss_functions(object):
    def __init__(self):
        pass

    def jacard_coef(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1)
    
    def dice_coef(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

    def dice_score(self, y_true, y_pred):
        # In a multi-class scenario, when provided with probability maps on one end
        # and one-hot encoded labels on the other, it effectively performs multiple two-class problems
        # calculate a vector of dice scores per class and take its mean
        intersection = tf.reduce_sum(y_pred * y_true, axis=[0,1])
        denominator =  tf.reduce_sum(y_pred, axis=[0,1]) + tf.reduce_sum(y_true, axis=[0,1]) - intersection

        # divide numerator and denominator element-wise
        dice_vector = intersection/denominator
        dice_score = tf.reduce_mean(dice_vector, axis=-1)
        return dice_score

    def focal_loss_with_logits(self, logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(
            -logits)) * (weight_a + weight_b) + logits * weight_b


    def generalized_dice_coefficient(self, y_true, y_pred):
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (
                    K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return score


    def tversky_index(self, y_true, y_pred):
        y_true_pos = K.flatten(y_true)
        y_pred_pos = K.flatten(y_pred)
        true_pos = K.sum(y_true_pos * y_pred_pos)
        false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
        alpha = 0.7
        return (true_pos + smooth) / (true_pos + alpha * false_neg + (
                    1 - alpha) * false_pos + smooth)

##################################################################################

    def focal_loss(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(),
                                  1 - tf.keras.backend.epsilon())
        logits = tf.math.log(y_pred / (1 - y_pred))

        loss = self.focal_loss_with_logits(logits=logits, targets=y_true,
                                      alpha=alpha, gamma=gamma, y_pred=y_pred)
        return tf.reduce_mean(loss)

    def jacard_loss(self, y_true, y_pred):
        return 1 - self.jacard_coef(y_true, y_pred)

    def dice_loss(self, y_true, y_pred):
        loss = 1 - self.generalized_dice_coefficient(y_true, y_pred)
        return loss

    def tversky_loss(self, y_true, y_pred):
        return 1 - self.tversky_index(y_true, y_pred)

    def focal_tversky_loss(self, y_true, y_pred):
        pt_1 = self.tversky_index(y_true, y_pred)
        gamma = 0.75
        return K.pow((1 - pt_1), gamma)
    
    def ssim_loss(self, y_true, y_pred):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.expand_dims(y_true, axis=-1)
        y_pred = tf.expand_dims(y_pred, axis=-1)
        ssim = tf.image.ssim(y_true, y_pred, max(tf.reduce_max(y_true), tf.reduce_max(y_pred)))
        return 1 - ssim

    def hybrid_loss(self, y_true, y_pred):
        loss = categorical_crossentropy(y_true, y_pred) + self.dice_loss(y_true, y_pred)
        return loss

    def ssim_dice_loss(self, y_true, y_pred):
        loss = self.ssim_loss(y_true, y_pred) + self.dice_loss(y_true, y_pred)
        return loss    

    def dual_hybrid_loss(self, y_true, y_pred):
        loss = categorical_crossentropy(y_true, y_pred) + self.ssim_dice_loss(y_true, y_pred)
        return loss
    
    # def RMI(self, y_true, y_pred):
    #     return