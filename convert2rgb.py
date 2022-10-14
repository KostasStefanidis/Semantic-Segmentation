import numpy as np
import os
import tensorflow as tf
import sys

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

MODELS_DIR = 'saved_models'
MODEL_TYPE = str(sys.argv[1])
MODEL_NAME = str(sys.argv[2])
MODEL_NAME = f'{MODEL_TYPE}/{MODEL_NAME}'
SPLIT = str(sys.argv[3])
assert SPLIT in ['validation', 'test']

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