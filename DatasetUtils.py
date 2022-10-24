import tensorflow as tf
from keras.layers import RandomContrast
from keras.models import Sequential
from keras import backend as K
from tensorflow import Tensor
import random
from keras.layers import RandomFlip
import tensorflow_addons as tfa

# dictionary that contains the mapping of the class numbers to rgb color values
color_dict = {0: [0, 0, 0],
              1: [0, 0, 0],
              2: [0, 0, 0],
              3: [0, 0, 0],
              4: [0, 0, 0],
              5: [111, 74, 0],
              6: [81, 0, 81],
              7: [128, 64,128],
              8: [244, 35,232],
              9: [250,170,160],
              10: [230,150,140],
              11: [ 70, 70, 70],
              12: [102,102,156],
              13: [190,153,153],
              14: [180,165,180],
              15: [150,100,100],
              16: [150,120, 90],
              17: [153,153,153],
              18: [153,153,153],
              19: [250,170, 30],
              20: [220,220,  0],
              21: [107,142, 35],
              22: [152,251,152],
              23: [70,130,180],
              24: [220, 20, 60],
              25: [255,  0,  0],
              26: [0,  0,142],
              27: [0,  0, 70],
              28: [0, 60,100],
              29: [0, 60,100],
              30: [0,  0,110],
              31: [0, 80,100],
              32: [0,  0,230],
              33: [119, 11, 32]
              }

#from keras.layers.preprocessing.image_preprocessing import BaseImageAugmentationLayer

# class RandomBrightness():
#     def __init__(self, max_delta: float, seed: int = None) -> None:
#         self.max_delta = max_delta
#         self.seed = seed
    
#     def call(self, x: Tensor):
#         return tf.image.random_brightness(x, self.max_delta, self.seed)
             
# class RandomGaussianBlur(BaseImageAugmentationLayer):
#     def __init__(self, max_sigma: float, min_kernel_size: int, max_kernel_size: int) -> None:
#         super(RandomGaussianBlur, self).__init__()
#         while(True):
#             size = random.randint(min_kernel_size, max_kernel_size)
#             if size%2==1:
#                 break
#         self.kernel_size = (size, size)        
#         if isinstance(max_sigma, (float, int)):
#             self.sigma = random.uniform(0.0, max_sigma)

#     def call(self, image):
#         blured_image = tfa.image.gaussian_filter2d(image, filter_shape=self.kernel_size, sigma=self.sigma)
#         return blured_image

class Augment(tf.keras.layers.Layer):
    def __init__(self, seed):
        super().__init__()
        #both use the same seed, so they'll make the same random changes.
        self.augment_inputs = self.augment(seed, mode='image')
        self.augment_labels = self.augment(seed, mode='label')

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels
    
    def augment(self, seed, mode):
        model = Sequential()
        model.add(RandomFlip("horizontal", seed=seed))
        if mode=='image':
            #model.add(RandomBrightness(0.1, seed=seed)) #This layer is only available in tensorflow 2.9
            model.add(RandomContrast(0.2, seed=seed))
            #model.add(RandomGaussianBlur(max_sigma=2, min_kernel_size=3, max_kernel_size=11))        
        return model


class Dataset():
    def __init__(self, num_classes: int, split:str, preprocessing='default', shuffle=True):
        assert split in ['train', 'validation', 'test'], 'split must one of: "train", "validation", "test".'
        assert num_classes in [20, 34], f'The num_classes argument must be either 20 or 34, instead the value given was {num_classes}'
        
        self.num_classes = num_classes
        self.split = split
        self.preprocessing = preprocessing
        self.shuffle = shuffle
        self.ignore_ids = [-1,0,1,2,3,4,5,6,9,10,14,15,16,18,29,30]
        self.eval_ids =   [7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]
        self.train_ids =  [0,1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18]
    
    
    def read_path(self, ds_path):
        ds = ds_path.map(tf.io.read_file)
        ds = ds.map(tf.image.decode_image)
        return ds


    def ds_from_path(self, img_path: str, label_path: str, seed: int):
        # either shuffle=None and the shuffling is done by passing a seed to the random generator
        # or shuffle=False and seed=None and we get the files in deterministic order
        if self.shuffle == True:
            seed = seed
            shuffle = None
        else:
            seed = None
            shuffle = False
        
        # Create a dataset of strings corresponding to file names matching img_path    
        img_path_ds = tf.data.Dataset.list_files(img_path, seed=seed, shuffle=shuffle)
        label_path_ds = tf.data.Dataset.list_files(label_path, seed=seed, shuffle=shuffle)
        
        # read and decode files
        img = self.read_path(img_path_ds)
        label = self.read_path(label_path_ds)
        
        dataset = tf.data.Dataset.zip((img, label))
        return dataset


    def create_patches(self, ds):
        ds = ds.map(lambda x, y: (tf.expand_dims(x, axis=0), tf.expand_dims(y, axis=0)),
                    num_parallel_calls=tf.data.AUTOTUNE)

        ds = ds.map(lambda image, label: (tf.image.extract_patches(image, sizes=[1,256,256,1], strides=[1,256,256,1], rates=[1,1,1,1], padding='VALID'),
                                  tf.image.extract_patches(label, sizes=[1,256,256,1], strides=[1,256,256,1], rates=[1,1,1,1], padding='VALID')),
                    num_parallel_calls=tf.data.AUTOTUNE)
        
        ds = ds.map(lambda image, label: (tf.reshape(image, [32,256,256,3]),
                                          tf.reshape(label, [32,256,256,1])),
                    num_parallel_calls=tf.data.AUTOTUNE)
        
        ds = ds.map(lambda image, label: (tf.cast(image, tf.uint8), tf.cast(label, tf.uint8)),
                    num_parallel_calls=tf.data.AUTOTUNE)
        
        return ds

    
    def set_shape_for_patches(self, images, labels):
        images.set_shape((32, 256, 256, 3))
        labels.set_shape((32, 256, 256, 1))
        return images, labels
    
    def set_shape(self, images, labels):
        images.set_shape((1024, 2048, 3))
        labels.set_shape((1024, 2048, 1))
        return images, labels
    
    
    def preprocess_image(self, image:tf.Tensor):
        # Layer for normalizing input image
        normalization_layer = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1)
        preprocessing_options = {
            'default': normalization_layer,
            #'DenseNet': tf.keras.applications.densenet.preprocess_input, #todo: solve issue
            'ResNet': tf.keras.applications.resnet.preprocess_input, 
            'EfficientNet' : tf.keras.applications.efficientnet.preprocess_input
        }
        preprocess_input = preprocessing_options[self.preprocessing]
        return preprocess_input(image)
    
    
    def preprocess_label(self, label:tf.Tensor):
        # squeeze axis' with dimension=1
        label = tf.cast(tf.squeeze(label), tf.int32)
        
        # Replace eval ids with train ids if number of classes = 20 (19 evaluated classes + 1 void class)
        if self.num_classes==20:    
            for id in self.ignore_ids:
                label = tf.where(label==id, 34, label)
            for train_id, eval_id in zip(self.train_ids, self.eval_ids):
                label = tf.where(label==eval_id, train_id, label)
            label = tf.where(label==34, 19, label)

        label = tf.one_hot(label, self.num_classes)

        # keep only classes to be evaluated, discard class no 19 which contains all the ignored classes condensed
        # label = tf.gather(label, indices=self.eval_ids, axis=-1)
        return label

    
    def preprocess_dataset(self, ds, use_patches: bool, augment: bool, seed: int):
        if use_patches:
            ds = self.create_patches(ds)
            ds = ds.map(self.set_shape_for_patches)
        else:
            ds = ds.map(self.set_shape)

        if augment:
            ds = ds.map(Augment(seed))
            ds = ds.map(lambda image, label: (image, tf.cast(label, tf.uint8)), 
                        num_parallel_calls=tf.data.AUTOTUNE)
        
        ds = ds.map(lambda image, label: (self.preprocess_image(image), label),
                    num_parallel_calls=tf.data.AUTOTUNE)
        
        # no pre-processing on the labels of the test set
        # no evaluation can be done on the test set, only prediction
        if self.split != 'test':
            ds = ds.map(lambda image, label: (image, self.preprocess_label(label)),
                        num_parallel_calls=tf.data.AUTOTUNE)           
        return ds


    def configure_for_performance(self, ds, batch: bool, batch_size: int):
        if batch:
            ds = ds.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds


    def create(self,
               img_path: str,
               label_path: str,
               batch_size: int = 1,
               use_patches: bool = False,
               augment: bool = False,
               seed = 42):
        '''
        Takes as input the file paths of images and labels and the number of classes in the given problem
        and returns a tf.data.Dataset object. Preprocessing includes normalization [-1, 1] of the images and the 
        one-hot encoding of the labels.
        '''
        if use_patches:
            batch = False
        else:
            batch = True
        ds = self.ds_from_path(img_path, label_path, seed)
        ds = self.preprocess_dataset(ds, use_patches, augment, seed)
        ds = self.configure_for_performance(ds, batch, batch_size)
        return ds