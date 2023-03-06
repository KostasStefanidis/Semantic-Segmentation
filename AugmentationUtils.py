import tensorflow as tf
from keras.layers import RandomFlip, RandomBrightness, RandomContrast
import tensorflow_addons as tfa
from keras.models import Sequential
from keras.layers.preprocessing.image_preprocessing import BaseImageAugmentationLayer
import random


class RandomGaussianBlur(BaseImageAugmentationLayer):
    def __init__(self, max_sigma: float, min_kernel_size: int, max_kernel_size: int) -> None:
        super(RandomGaussianBlur, self).__init__()
        while(True):
            size = random.randint(min_kernel_size, max_kernel_size)
            if size%2==1:
                break
        self.kernel_size = (size, size)        
        if isinstance(max_sigma, (float, int)):
            self.sigma = random.uniform(0.0, max_sigma)

    def call(self, image):
        blured_image = tfa.image.gaussian_filter2d(image, filter_shape=self.kernel_size, sigma=self.sigma)
        return blured_image

class Augment(tf.keras.layers.Layer):
    def __init__(self, seed):
        super().__init__()
        #both use the same seed, so they'll make the same random changes.
        self.augment_images = self.augment(seed, mode='image')
        self.augment_gt_images = self.augment(seed, mode='label')

    def call(self, inputs, labels):
        inputs = self.augment_images(inputs)
        labels = self.augment_gt_images(labels)
        return inputs, labels
    
    def augment(self, seed, mode):
        model = Sequential()
        model.add(RandomFlip("horizontal", seed=seed))
        if mode=='image':
            model.add(RandomBrightness(0.15, seed=seed))
            model.add(RandomContrast(0.25, seed=seed))
            model.add(RandomGaussianBlur(max_sigma=2, min_kernel_size=3, max_kernel_size=11))        
        return model