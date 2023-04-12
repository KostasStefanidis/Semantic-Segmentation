import tensorflow as tf
from utils.datasets import CityscapesDataset, cityscapes_color_map
import os
import numpy as np
from argparse import ArgumentParser
import yaml

parser = ArgumentParser('')
parser.add_argument('--config', type=str, nargs='?')
parser.add_argument('--data_path', type=str, nargs='?')
parser.add_argument('--model_type', type=str, nargs='?', choices=['Unet', 'Residual_Unet', 'Attention_Unet', 'Unet_plus', 'DeepLabV3plus'])
parser.add_argument('--model_name', type=str, nargs='?')
parser.add_argument('--backbone', type=str, nargs='?', default='None')
parser.add_argument('--num_classes', type=int, nargs='?', default='20', choices=[20,34])
parser.add_argument('--split', type=str, nargs='?', choices=['train', 'val', 'test'], required=True)
args = parser.parse_args()

if args.config is None:
# parse arguments
    print('Reading configuration from cmd args')
    DATA_PATH = args.data_path
    MODEL_TYPE = args.model_type
    MODEL_NAME = args.model_name
    NUM_CLASSES = args.num_classes
    BACKBONE = args.backbone
    
else:
    # Read YAML file
    print('Reading configuration from config yaml')
    
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    LOGS_DIR = config['logs_dir']

    dataset_config = config['dataset']
    model_config = config['model']
    inference_config = config['inference_config']
    
    DATASET = dataset_config['name']
    DATA_PATH = dataset_config['path']
    VERSION = dataset_config['version']
    
    MODEL_TYPE = model_config['architecture']
    MODEL_NAME = model_config['name']
    BACKBONE = model_config['backbone']
    
    INFERENCE_OUTPUT_STRIDE = inference_config['output_stride']
    INFERENCE_PRECISION = inference_config['precision']
    

SPLIT = args.split
BATCH_SIZE = 1

# Define preprocessing according to the Backbone
if BACKBONE == 'None':
    PREPROCESSING = 'default'
    BACKBONE = None
elif 'ResNet' in BACKBONE:
    PREPROCESSING = 'ResNet'
    if 'V2' in BACKBONE:
        PREPROCESSING = 'ResNetV2'
elif 'EfficientNet' in BACKBONE:
    PREPROCESSING = 'EfficientNet'
elif 'EfficientNetV2' in BACKBONE:
    PREPROCESSING = 'EfficientNetV2'
elif 'MobileNet' == BACKBONE:
    PREPROCESSING = 'MobileNet'
elif 'MobileNetV2' == BACKBONE:
    PREPROCESSING = 'MobileNetV2'
elif 'MobileNetV3' in BACKBONE:
    PREPROCESSING = 'MobileNetV3'
elif 'RegNet' in BACKBONE:
    PREPROCESSING = 'RegNet'
else:
    raise ValueError(f'Enter a valid Backbone name, {BACKBONE} is invalid.')

if DATASET == 'Cityscapes':
    NUM_CLASSES = 20
    IGNORE_CLASS = 19
    INPUT_SHAPE = (1024, 2048, 3)
elif DATASET == 'Mapillary':
    INPUT_SHAPE = (1024, 1856, 3)
    if VERSION == 'v1.2':
        NUM_CLASSES = 64
        IGNORE_CLASS = 63
    elif VERSION == 'v2.0':
        NUM_CLASSES = 118
        IGNORE_CLASS = 117
    else:
        raise ValueError('Version of the Mapillary Vistas dataset should be either v1.2 or v2.0!')
else:
    raise ValueError(F'{DATASET} dataset is invalid. Available Datasets are: Cityscapes, Mapillary!')

tf.keras.backend.set_floatx(INFERENCE_PRECISION)

# ---------------------------Create Dataset stream--------------------------------
if DATASET == 'Cityscapes':
    train_ds = CityscapesDataset(NUM_CLASSES, 'train', PREPROCESSING, shuffle=True)
    train_ds = train_ds.create(DATA_PATH, 'all', BATCH_SIZE, -1, augment=False)

    val_ds = CityscapesDataset(NUM_CLASSES, 'val', PREPROCESSING, shuffle=False)
    val_ds = val_ds.create(DATA_PATH, 'all', BATCH_SIZE, -1, augment=False)
else:
    raise ValueError(f'Inference is not supported for {DATASET} Dataset')


MODEL_NAME = f'{MODEL_TYPE}/{MODEL_NAME}'
MODELS_DIR = 'saved_models'
pred_path = f'predictions/{MODEL_NAME}/{SPLIT}/grayscale'
rgb_path = f'predictions/{MODEL_NAME}/{SPLIT}/rgb'

os.makedirs(f'{rgb_path}', exist_ok=True)
os.makedirs(pred_path, exist_ok=True)

ds = CityscapesDataset(NUM_CLASSES, SPLIT, PREPROCESSING, shuffle=False)
ds = ds.create(DATA_PATH, 'all', BATCH_SIZE, augment=False)

# add val set filenames into a python list
img_path_ds = tf.data.Dataset.list_files(f'{DATA_PATH}/leftImg8bit_trainvaltest/leftImg8bit/{SPLIT}/*/*.png', shuffle=False)
img_name_list = []
for img_path in img_path_ds:
    split = tf.strings.split(img_path, sep='/').numpy()
    img_name = split[-1]
    img_name = img_name.decode()
    img_name_list.append(img_name)

model = tf.keras.models.load_model(f'{MODELS_DIR}/{DATASET}/{MODEL_NAME}', compile=False)

eval_ids =   [7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33, 0] # MAP VOID CLASS TO 0 -> TOTAL BLACK 
train_ids =  [0,1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19]

for dataset_elem, filename in zip(ds, img_name_list):
    input_image = dataset_elem[0]
    # MAKE PREDICTION
    prediction = model.predict_on_batch(input_image)
    # PREDICTED LABEL IN GREYSCALE
    prediction = tf.argmax(prediction, axis=-1)
    prediction = tf.cast(prediction, tf.uint8)
    prediction = tf.squeeze(prediction)
    
    if NUM_CLASSES == 20:
        # map the classes back to the eval_ids
        for train_id, eval_id in zip(reversed(train_ids), reversed(eval_ids)):        
            prediction = tf.where(prediction==train_id, eval_id, prediction)
            
    grayscale_prediction = tf.expand_dims(prediction, axis=-1)
    # save grayscale predictions in 'predictions' folder
    print(f'Saving prediciton for : {filename}')
    tf.keras.utils.save_img(f'{pred_path}/{filename}', grayscale_prediction, data_format='channels_last', scale=False)
    
    rgb_pred = np.zeros((*prediction.shape, 3)) 
    for key in cityscapes_color_map.keys():
        rgb_pred[prediction == key] = cityscapes_color_map[key]
        
    Y = 0.6
    rgb_img = Y * tf.squeeze(input_image).numpy() + (1-Y) * rgb_pred
    
    tf.keras.utils.save_img(f'{rgb_path}/{filename}', rgb_img, data_format='channels_last', scale=False)