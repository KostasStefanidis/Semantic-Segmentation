import tensorflow as tf
from keras.models import Sequential
from tensorflow import Tensor
import random
from keras.layers import RandomFlip, RandomBrightness, RandomContrast
import tensorflow_addons as tfa
from keras.layers.preprocessing.image_preprocessing import BaseImageAugmentationLayer
from keras.applications import resnet, resnet_v2, efficientnet, efficientnet_v2, regnet
from keras.applications import mobilenet, mobilenet_v2, mobilenet_v3

# dictionary that contains the mapping of the class numbers to rgb color values
color_map =  {0: [0, 0, 0],
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


class Dataset():
    def __init__(self,
                 num_classes: int, 
                 split: str, 
                 preprocessing: str = 'default', 
                 mode: str = 'fine', 
                 shuffle=True):
        """
        Instantiate a Dataset object. Next call the `create()` method to create a pipeline that contains 
        parsing, decoding and preprossecing of the dataset images which yields, image and ground truth image
        pairs to feed into the network for either training, evalution or inference.
        
        
        Args:
            - `num_classes` (int): Number of classes. Available options: 20 or 34.
            - `split` (str): The split of the dataset to be used. Must be one of `"train"`, `"val"` or `"test"`.
            - `preprocessing` (str, optional): A string denoting the what type of preprocessing will be done to the images of the dataset.
               Available options: `"default"`, `"ResNet"`, `"EfficientNet"`, `"EfficientNetV2"`. Defaults to `'default'` 
               -> Normalize the pixel values to [-1, 1] interval.
            - `shuffle` (bool, optional): Whether or not to shuffle the elements of the dataset. Defaults to True.
        """
        
        assert split in ['train', 'val', 'test'], f'The split arguement must one of: "train", "val", "test", instead the value passed was {split}'
        assert num_classes in [20, 34], f'The num_classes argument must be either 20 or 34, instead the value passed was {num_classes}'
        
        self.num_classes = num_classes
        self.split = split
        self.preprocessing = preprocessing
        self.mode = mode
        self.shuffle = shuffle
        
        self.ignore_ids = [-1,0,1,2,3,4,5,6,9,10,14,15,16,18,29,30]
        self.eval_ids =   [7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]
        self.train_ids =  [0,1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18]
        
        if self.mode == 'fine':
            self.img_path = 'leftImg8bit_trainvaltest/leftImg8bit/'
            self.label_path = 'gtFine_trainvaltest/gtFine/'
        elif self.mode == 'coarse':
            self.img_path = 'leftImg8bit/train_extra/'
            self.label_path = 'gtCoarse/train_extra/'
        
        self.img_suffix = '*.png'
        self.label_suffix = f'*_gt{self.mode.capitalize()}_labelIds.png'
    
    
    def construct_path(self, data_path: str, subfolder: str):
        if subfolder == 'all':
            subfolder = '*'
        
        if self.mode == 'fine':
            image_path = data_path + self.img_path + self.split + '/' + subfolder + '/' + self.img_suffix
            label_path = data_path + self.label_path + self.split + '/' + subfolder + '/' + self.label_suffix
        elif self.mode == 'coarse':
            image_path = data_path + self.img_path + subfolder + '/' + self.img_suffix
            label_path = data_path + self.label_path + subfolder + '/' + self.label_suffix
        return image_path, label_path
    
    
    def decode_dataset(self, path_ds):
        ds = path_ds.map(tf.io.read_file)
        ds = ds.map(tf.image.decode_image)
        return ds


    def dataset_from_path(self, data_path: str, subfolder, seed: int):
        img_path, label_path = self.construct_path(data_path, subfolder)
        
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
        img = self.decode_dataset(img_path_ds)
        
        if self.split == 'testing':
            dataset = img
        else:
            label_path_ds = tf.data.Dataset.list_files(label_path, seed=seed, shuffle=shuffle)
            label = self.decode_dataset(label_path_ds)
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
    
    
    def preprocess_image(self, image: Tensor):
        # Layer for normalizing input image
        default_normalization_layer = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1)
        preprocessing_options = {
            'default': default_normalization_layer,
            'ResNet': resnet.preprocess_input,
            'ResNetV2' : resnet_v2.preprocess_input,
            'MobileNet' : mobilenet.preprocess_input,
            'MobileNetV2' : mobilenet_v2.preprocess_input,
            'MobileNetV3' : mobilenet_v3.preprocess_input,
            'EfficientNet' : efficientnet.preprocess_input,
            'EfficientNetV2' : efficientnet_v2.preprocess_input,
            'RegNet' : regnet.preprocess_input
        }
        preprocess_input = preprocessing_options[self.preprocessing]
        return preprocess_input(image)
    
    
    def preprocess_label(self, label: Tensor):
        label = tf.cast(tf.squeeze(label), tf.int32)
        
        # Map eval ids to train ids
        if self.num_classes==20:    
            for id in self.ignore_ids:
                label = tf.where(label==id, 34, label)
            for train_id, eval_id in zip(self.train_ids, self.eval_ids):
                label = tf.where(label==eval_id, train_id, label)
            label = tf.where(label==34, 19, label)

        label = tf.one_hot(label, self.num_classes, dtype=tf.float32)
        return label

    
    def preprocess_dataset(self, dataset: tf.data.Dataset, use_patches: bool, augment: bool, seed: int):
        if use_patches:
            dataset = self.create_patches(dataset)
            dataset = dataset.map(self.set_shape_for_patches)
        else:
            # the shapes must be explicitly set because tensorflow cannot infer them when reading the data from files
            dataset = dataset.map(self.set_shape)

        if self.split == 'testing':
            # in testing split there are only images and no ground truth
            dataset = dataset.map(lambda image: (self.preprocess_image(image)),
                    num_parallel_calls=tf.data.AUTOTUNE)
        else:
            # augmentation is done only for training set
            if augment:
                dataset = dataset.map(Augment(seed))
                dataset = dataset.map(lambda image, label: (image, tf.cast(label, tf.uint8)), 
                            num_parallel_calls=tf.data.AUTOTUNE)
            
            dataset = dataset.map(lambda image, label: (self.preprocess_image(image), self.preprocess_label(label)),
                        num_parallel_calls=tf.data.AUTOTUNE)

        return dataset


    def configure_dataset(self, dataset: tf.data.Dataset, batch: bool, batch_size: int, count: int =-1):
        if batch:
            dataset = dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.take(count)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset


    def create(self,
               data_path: str,
               batch_size: int = 1,
               count: int = -1,
               use_patches: bool = False,
               augment: bool = False,
               seed = 42):
        """ Create a dataset generator. The pre-processing pipeline consists of 1) optionally splitting each image to smaller patches, 2) optionally augmenting each image
        3) normalizing the input images and 4) optionally map the eval ids of the ground truth images to train ids and finally convert them to one-hot.

        Args:
            - `data_path` (str): The relative or absolute path of the directory containing the dataset folders. 
                Both `leftImg8bit_trainvaltest` and `gtFine_trainvaltest` directories must be in the `data_path` parent directory.
            - `subfolder` (str, optional): The subfolder to read images from. Defaults to 'all'.
            - `batch_size` (int, optional): The size of each batch of images. Essentially how many images will 
            be processed and will propagate through the network at the same time. Defaults to 1.
            - `count` (int, optional) : The number of elements i.e. (image, ground_truth) pairs that should be taken from the whole dataset. If count is -1,
                or if count is greater than the size of the whole dataset, then will contain all elements of this dataset. Defaults to -1.
            - `use_patches` (bool, optional): Whether or not to split the images into smaller patches. 
            Patch size is fixed to (256, 256) and the batch size is fixed to 32. When Defaults to False.
            - `augment` (bool, optional): Whether to use data augmentation or not. Defaults to False.
            - `seed` (int, optional): The seed used for the shuffling of the dataset elements.
                This value will also be used as a seed for the random transformations during augmentation. Defaults to 42.

        Returns:
            tf.data.Dataset
        """
        if use_patches:
            batch = False
        else:
            batch = True
        dataset = self.dataset_from_path(data_path, seed)
        dataset = self.preprocess_dataset(dataset, use_patches, augment, seed)
        dataset = self.configure_dataset(dataset, batch, batch_size, count)
        return dataset