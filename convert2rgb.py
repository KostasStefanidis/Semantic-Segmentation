import numpy as np
import os
import tensorflow as tf
import sys
from DatasetUtils import color_map
from argparse import ArgumentParser

parser = ArgumentParser('')
parser.add_argument('--model_type', type=str, nargs='?', required=True, choices=['Unet', 'Residual_Unet', 'Attention_Unet', 'Unet_plus', 'DeepLabV3plus'])
parser.add_argument('--model_name', type=str, nargs='?', required=True)
parser.add_argument('--split', type=str, nargs='?', choices=['train', 'val', 'test'], required=True)
args = parser.parse_args()

MODEL_TYPE = args.model_type
MODEL_NAME = args.model_name
SPLIT = args.split
MODELS_DIR = 'saved_models'
MODEL_NAME = f'{MODEL_TYPE}/{MODEL_NAME}'

pred_path = f'predictions/{MODEL_NAME}/{SPLIT}/grayscale'
rgb_path = f'predictions/{MODEL_NAME}/{SPLIT}/rgb'

os.makedirs(f'{rgb_path}', exist_ok=True)

grayscale_image_filenames = os.listdir(f'{pred_path}')

print('Converting to RGB : ')
for filename in grayscale_image_filenames:
    img = tf.io.read_file(f'{pred_path}/{filename}')
    img = tf.image.decode_png(img)
    img = tf.squeeze(img).numpy()
    rgb_img = np.zeros((*img.shape, 3)) 
    for key in color_map.keys():
        rgb_img[img == key] = color_map[key]
    
    print(f'Converting to RGB : {filename}')
    tf.keras.utils.save_img(f'{rgb_path}/{filename}', rgb_img, data_format='channels_last', scale=False)