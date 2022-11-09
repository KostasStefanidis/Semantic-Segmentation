# Semantic Segmentation on the Cityscapes Dataset

1. Dataset handling
    - Parse the dataset:
        - `data_path`
            - `leftImg8bit_trainvaltest` -> RGB Images
                - `leftImg8bit`
                    - `train` -> train split
                        - `*`
                    - `val` -> validation split
                        - `*`
                    - `test` -> test split
                        - `*`
            - `gtFine_trainvaltest` -> Ground Truth Images
                - `gtFine`
                    - `train`
                        - `*`
                    - `val`
                        - `*`
                    - `test`
                        - `*`

    Each of the train,val,test directories contain subdirectories with the name of a city. 
    To use the whole split, `subfolder='all'` must be passed to the `Dataset.create` method
    in order to use all the subfolders.
    If you want to test with a smaller number of images you can enter `subfolder='<CityName>'`
    for each individual split. For example, `split='train'`, and `subfolder='aachen'` will only
    read the 174 images in the folder `aachen` and convert them into a tf.data.Dataset. 
    You can choose either all the subfolders or one of them not an arbitrary combination of them.
