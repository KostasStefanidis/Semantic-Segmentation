import numpy as np
import os
import tensorflow as tf
import sys
from DatasetUtils import color_dict
from argparse import ArgumentParser

parser = ArgumentParser('')
parser.add_argument('-t', type=str, nargs='?', required=True)
parser.add_argument('-m', type=str, nargs='?', required=True)
parser.add_argument('-s', type=str, nargs='?', choices=['train', 'validation', 'test'])
args = parser.parse_args()

MODEL_TYPE = args.t
MODEL_NAME = args.m
SPLIT = args.s
MODELS_DIR = 'saved_models'
MODEL_NAME = f'{MODEL_TYPE}/{MODEL_NAME}'

pred_path = f'predictions/{MODEL_NAME}/{SPLIT}/grayscale'
rgb_path = f'predictions/{MODEL_NAME}/{SPLIT}/rgb'

os.makedirs(f'{rgb_path}', exist_ok=True)

filenames = os.listdir(f'{pred_path}')

for filename in filenames:
    img = tf.io.read_file(f'{pred_path}/{filename}')
    img = tf.image.decode_png(img)
    img = tf.squeeze(img).numpy()
    rgb_img = np.zeros((*img.shape, 3)) 
    for key in color_dict.keys():
        rgb_img[img == key] = color_dict[key]
    tf.keras.utils.save_img(f'{rgb_path}/{filename}', rgb_img, data_format='channels_last', scale=False)