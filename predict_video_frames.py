import tensorflow as tf
from DatasetUtils import Dataset
import os
import imageio
import numpy as np
from DatasetUtils import color_map

data_path = '/home/kstef/dataset/'
MODEL_TYPE = 'DeepLabV3plus'
MODEL_NAME = 'FocalHybrid_LeakyRelu_batch-3_Dropout-0.0_EfficientNetV2M'
NUM_CLASSES = 20
BACKBONE = 'EfficientNetV2M'
BATCH_SIZE = 1
subfolder = 'stuttgart_02'

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

demo_video_path = f'predictions/demoVideo/{MODEL_NAME}/{subfolder}/'
pred_frame_path = f'{demo_video_path}/frames'
pred_video_path = f'{demo_video_path}/video'
input_video_filename = f'{pred_video_path}/input.gif'
pred_video_filename = f'{pred_video_path}/prediction.gif'

os.makedirs(pred_frame_path, exist_ok=True)
os.makedirs(pred_video_path, exist_ok=True)

ds = Dataset(NUM_CLASSES, 'test', PREPROCESSING, shuffle=False, mode='video')
ds = ds.create(data_path, subfolder, BATCH_SIZE, use_patches=False, augment=False)

frames = next(iter(ds))[0].numpy()

def to_gif(images, filename):
  converted_images = images.astype(np.uint8)
  imageio.mimsave(f'./{filename}', converted_images, fps=10)

# 
#to_gif(frames, input_video_filename)

# add val set filenames into a python list
img_path_ds = tf.data.Dataset.list_files(f'{data_path}/demoVideo/{subfolder}/*.png', shuffle=False)
img_name_list = []
for img_path in img_path_ds:
    split = tf.strings.split(img_path, sep='/').numpy()
    img_name = split[-1]
    img_name = img_name.decode()
    img_name_list.append(img_name)
    
model = tf.keras.models.load_model(f'{MODELS_DIR}/{MODEL_NAME}', compile=False)

eval_ids =   [7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33, 0] # MAP VOID CLASS TO 0 -> TOTAL BLACK 
train_ids =  [0,1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19]

for dataset_elem, name in zip(ds, img_name_list):
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
            
    # Convert and save directly to RGB
    prediction = prediction.numpy()
    rgb_img = np.zeros((*prediction.shape, 3)) 
    for key in color_map.keys():
        rgb_img[prediction == key] = color_map[key]
        
    # save predictions in 'predictions' folder
    print('Saving RGB prediciton for : ', name)
    tf.keras.utils.save_img(f'{pred_frame_path}/{name}', rgb_img, data_format='channels_last', scale=False)
    

# pred_path = f'predictions/{MODEL_NAME}/grayscale'
# rgb_path = f'predictions/{MODEL_NAME}/rgb'

# os.makedirs(f'{rgb_path}', exist_ok=True)

# grayscale_image_filenames = os.listdir(f'{pred_path}')

# for filename in grayscale_image_filenames:
#     img = tf.io.read_file(f'{pred_path}/{filename}')
#     img = tf.image.decode_png(img)
#     img = tf.squeeze(img).numpy()
#     rgb_img = np.zeros((*img.shape, 3)) 
#     for key in color_map.keys():
#         rgb_img[img == key] = color_map[key]
    
#     print(f'Converting to RGB : {filename}')
#     tf.keras.utils.save_img(f'{rgb_path}/{filename}', rgb_img, data_format='channels_last', scale=False)