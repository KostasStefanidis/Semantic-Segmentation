import tensorflow as tf
from tensorflow import Tensor
from keras.layers import Resizing
from keras.applications import resnet, resnet_v2, efficientnet, efficientnet_v2, regnet
from keras.applications import mobilenet, mobilenet_v2, mobilenet_v3
from .AugmentationUtils import Augment


class MapillaryDataset():
    def __init__(self,
                 height: int,
                 width: int,
                 split: str,
                 preprocessing: str = 'default',
                 version: str ='v1.2',
                 void_ids: list = None,
                 shuffle: bool = True):
        
        """
        Instantiate a Dataset object. Next call the `create()` method to create a pipeline that contains 
        parsing, decoding and preprossecing of the dataset images which yields, image and ground truth image
        pairs to feed into the network for either training, evalution or inference.
        
        Args:
            - `height` (int): Image height.
            - `width` (int): Image width.
            - `split` (str): The split of the dataset to be used. Must be one of `"training"`, `"validation"` or `"testing"`.
            - `preprocessing` (str, optional): A string denoting the what type of preprocessing will be done to the images of the dataset.
               Available options: `"default"`, `"ResNet"`, `"EfficientNet"`, `"EfficientNetV2"`. Defaults to `'default'` 
               -> Normalize the pixel values to [-1, 1] interval.
            - `version` (str): The version of Mapillary Vistas dataset. v1.2 -> 66 classes, v2.0 -> 124 classes
            - `shuffle` (bool, optional): Whether or not to shuffle the elements of the dataset. Defaults to True.
        """
        
        assert split in ['training', 'validation', 'testing'], f'The split arguement must one of: "training", "validation", "testing", instead the value passed was {split}'
        
        self.height = height
        self.width = width
    
        self.split = split
        self.preprocessing = preprocessing
        self.version = version
        self.shuffle = shuffle
        
        if version == 'v2.0':
            self.num_classes = 124
            self.void_ids = [118, 119, 120, 121, 122, 123]
        elif version == 'v1.2':
            self.num_classes = 66
            self.void_ids = [63, 64, 65]
        else:
            raise ValueError('Version must be either v1.2 or v2.0')
        
        self.num_classes = self.num_classes - len(self.void_ids) + 1
        
        self.img_path = 'images'
        self.label_path = f'{version}/instances'
        
        self.img_suffix = '*.jpg'
        self.label_suffix = f'*.png'
    
  
    def construct_path(self, data_path: str):
        image_path = data_path + self.split + '/' + self.img_path + '/' + self.img_suffix
        label_path = data_path + self.split + '/' + self.label_path + '/' + self.label_suffix
        return image_path, label_path


    def decode_dataset(self, path_ds: tf.data.Dataset):
        ds = path_ds.map(tf.io.read_file, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(tf.image.decode_image, num_parallel_calls=tf.data.AUTOTUNE)
        return ds


    def dataset_from_path(self, data_path: str):
        img_path, label_path = self.construct_path(data_path)
        
        # Create a dataset of strings corresponding to file names matching img_path    
        img_path_ds = tf.data.Dataset.list_files(img_path, shuffle=False)
        img = self.decode_dataset(img_path_ds)
        
        if self.split == 'testing':
            dataset = img
        else:
            label_path_ds = tf.data.Dataset.list_files(label_path, shuffle=False)
            label = self.decode_dataset(label_path_ds)
            dataset = tf.data.Dataset.zip((img, label))
        return dataset


    def set_shape_image(self, image):
        image.set_shape((self.height, self.width, 3))
        return image
        
    def set_shape_dataset(self, image, label):
        image.set_shape((self.height, self.width, 3))
        label.set_shape((self.height, self.width, 1))
        return image, label
    

    def preprocess_image(self, image: Tensor):
        cropping_layer = Resizing(height=self.height, width=self.width)
        
        image = cropping_layer(image)
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
        cropping_layer = Resizing(height=self.height, width=self.width)
        label = cropping_layer(label)
        label = tf.cast(tf.squeeze(label), tf.int32)
        # Map all void classes to 1 class
        for id in self.void_ids:
            if id == self.void_ids[0]:
                continue
            label = tf.where(label==id, self.void_ids[0], label)

        label = tf.one_hot(label, self.num_classes, dtype=tf.float32)
        return label


    def preprocess_dataset(self, dataset: tf.data.Dataset, augment: bool, seed: int):
        if self.split == 'testing':
            dataset = dataset.map(self.set_shape_image, num_parallel_calls=tf.data.AUTOTUNE)
            # in testing split there are only images and no ground truth
            dataset = dataset.map(lambda image: (self.preprocess_image(image)),
                    num_parallel_calls=tf.data.AUTOTUNE)
        else:
            dataset = dataset.map(self.set_shape_dataset, num_parallel_calls=tf.data.AUTOTUNE)
            # augmentation is done only for training set
            if augment:
                dataset = dataset.map(Augment(seed))
                dataset = dataset.map(lambda image, label: (image, tf.cast(label, tf.uint8)), 
                            num_parallel_calls=tf.data.AUTOTUNE)
            
            dataset = dataset.map(lambda image, label: (self.preprocess_image(image), self.preprocess_label(label)),
                        num_parallel_calls=tf.data.AUTOTUNE)

        return dataset


    def configure_dataset(self, dataset: tf.data.Dataset, batch_size: int, count: int =-1):
        dataset = dataset.take(count)
        dataset = dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        # if self.shuffle:
        #     dataset = dataset.shuffle(30, reshuffle_each_iteration=True)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset


    def create(self,
               data_path: str,
               batch_size: int = 1,
               count: int = -1,
               augment: bool = False,
               seed = 42):
        """ Create a dataset generator. The pre-processing pipeline consists of 1) optionally splitting each image to smaller patches, 2) optionally augmenting each image
        3) normalizing the input images and 4) optionally map the eval ids of the ground truth images to train ids and finally convert them to one-hot.

        Args:
            - `data_path` (str): The relative or absolute path of the directory containing the dataset folders. 
                Both `leftImg8bit_trainvaltest` and `gtFine_trainvaltest` directories must be in the `data_path` parent directory.
            - `batch_size` (int, optional): The size of each batch of images. Essentially how many images will 
            be processed and will propagate through the network at the same time. Defaults to 1.
            - `count` (int, optional) : The number of elements i.e. (image, ground_truth) pairs that should be taken from the whole dataset. If count is -1,
                or if count is greater than the size of the whole dataset, then will contain all elements of this dataset. Defaults to -1.
            - `augment` (bool, optional): Whether to use data augmentation or not. Defaults to False.
            - `seed` (int, optional): The seed used for the shuffling of the dataset elements.
                This value will also be used as a seed for the random transformations during augmentation. Defaults to 42.

        Returns:
            tf.data.Dataset
        """

        dataset = self.dataset_from_path(data_path)
        dataset = self.preprocess_dataset(dataset, augment, seed)
        dataset = self.configure_dataset(dataset, batch_size, count)
        return dataset