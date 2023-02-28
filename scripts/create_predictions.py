import tensorflow as tf
from DatasetUtils import Dataset, color_map
import os
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser('')
parser.add_argument('--data_path', type=str, nargs='?', required=True)
parser.add_argument('--model_type', type=str, nargs='?', required=True, choices=['Unet', 'Residual_Unet', 'Attention_Unet', 'Unet_plus', 'DeepLabV3plus'])
parser.add_argument('--model_name', type=str, nargs='?', required=True)
parser.add_argument('--backbone', type=str, nargs='?', default='None')
parser.add_argument('--num_classes', type=int, nargs='?', default='20', choices=[20,34])
parser.add_argument('--split', type=str, nargs='?', choices=['train', 'val', 'test'], required=True)
args = parser.parse_args()

data_path = args.data_path
MODEL_TYPE = args.model_type
MODEL_NAME = args.model_name
NUM_CLASSES = args.num_classes
BACKBONE = args.backbone
SPLIT = args.split
BATCH_SIZE = 1

if BACKBONE == 'None':
    PREPROCESSING = 'default'
    BACKBONE = None
elif 'ResNet' in BACKBONE:
    PREPROCESSING = 'ResNet'
elif 'EfficientNet' in BACKBONE:
    PREPROCESSING = 'EfficientNet'
elif 'EfficientNetV2' in BACKBONE:
    PREPROCESSING = 'EfficientNetV2'
else:
    raise ValueError(f'Enter a valid Backbone name, {BACKBONE} is invalid.')

MODEL_NAME = f'{MODEL_TYPE}/{MODEL_NAME}'
MODELS_DIR = 'saved_models'
pred_path = f'predictions/{MODEL_NAME}/{SPLIT}/grayscale'
rgb_path = f'predictions/{MODEL_NAME}/{SPLIT}/rgb'

os.makedirs(f'{rgb_path}', exist_ok=True)
os.makedirs(pred_path, exist_ok=True)

ds = Dataset(NUM_CLASSES, SPLIT, PREPROCESSING, shuffle=False)
ds = ds.create(data_path, 'all', BATCH_SIZE, use_patches=False, augment=False)

# add val set filenames into a python list
img_path_ds = tf.data.Dataset.list_files(f'{data_path}/leftImg8bit_trainvaltest/leftImg8bit/{SPLIT}/*/*.png', shuffle=False)
img_name_list = []
for img_path in img_path_ds:
    split = tf.strings.split(img_path, sep='/').numpy()
    img_name = split[-1]
    img_name = img_name.decode()
    img_name_list.append(img_name)

model = tf.keras.models.load_model(f'{MODELS_DIR}/{MODEL_NAME}', compile=False)

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
    for key in color_map.keys():
        rgb_pred[prediction == key] = color_map[key]
        
    Y = 0.7
    rgb_img = Y * tf.squeeze(input_image).numpy() + (1-Y) * rgb_pred
    
    tf.keras.utils.save_img(f'{rgb_path}/{filename}', rgb_img, data_format='channels_last', scale=False)