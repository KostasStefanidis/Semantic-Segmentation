# Semantic Segmentation on the Cityscapes Dataset

## 1. Image parsing and decoding
- Parse files which are under the following directory sctructure
    - `data_path` : the root folder which the dataset files are in
        - `leftImg8bit_trainvaltest` -> RGB Images
            - `leftImg8bit`
                - `train`
                - `val`
                - `test`
        - `gtFine_trainvaltest` -> Ground Truth Images
            - `gtFine`
                - `train`
                - `val`
                - `test`

Each of the train,val,test directories contain subdirectories with the name of a city. To use a whole split, `subfolder='all'` must be passed to the `Dataset.create()` method in order to read the images from the subfolders. For testing purposes a smaller number of images from the dataset can be used by passing `subfolder='<CityName>'` to the `create` method of the Dataset object. For example, passing `split='train'` to the Dataset constructor, and `subfolder='aachen'` to the `create` method will only read the 174 images in the folder `aachen` and convert them into a tf.data.Dataset. You can choose either all the subfolders or one of them not an arbitrary combination of them. After the images (x) and the ground truth images (y) are read and decoded, they are combined into a single object (x, y).

## 2. Preprocessing :
Generally images have a shape of `(batch_size, height, width, channels)`

1. Split the image into smaller patches with spatial resolution `(256, 256)`. Every image having a spatial resolution of `(1024, 2048)` produces 32 patches and all the patches belong to a batch. This means that when the patching technique is used the batch size is fixed to 32. After this operation the images have a shape of `(32, 256, 256, 3)` while the the ground truth images have a shape of `(32, 256, 256, 1)`. To enable patching set the `use_patches` arguement to `True`.

2. Perform data `Augmentation`
   - Randomly perform `horrizontal flipping` of images
   - Randomly adjust `brightness`
   - Randomly adjust `contrast`
   - Apply `gaussian blur` with random kernel size and variance

*NOTE : while all augmentations are performed on the images, only horrizontal flip is performed on the ground truth images, because changing the pixel values of the ground truth images means changing the class they belong to.*

3. Normalize images : 
   - `[-1, 1]` as default
   - If using a pretrained backbone normalize according to what the pretrained network expects at its input. The type of preprocessing is determined by the . To determine what type of preprocessing will be done to the images, the name of the pretrained network must be passed as the `preprocessing` arguement of the Dataset constructor. For example, if a version of the EfficientNet (i.e EfficientNetB0, EfficientNetB1, etc) network is used as a model backbone, then `preprocessing = "EfficientNet"` must be passed.

4. Preprocess ground truth images:
   - Map eval ids to train ids
   - Convert to `one-hot` encoding
   - After this operation ground truth have a shape of `(batch_size, 1024, 2048, num_classes)`
  
  Finally the dataset which is created is comprised by elements `(image, ground_truth)` with shape (`(batch_size, height, width, 3)`, `(batch_size, height, width, num_classes)`)

***

Supported Network families as backbone choices:
- EfficientNet
- EfficientNetV2
- ResNet