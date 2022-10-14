import tensorflow as tf
import numpy as np
from keras import backend as K
from DatasetUtils import Dataset
from EvaluationUtils import MeanIoU, ConfusionMatrix
from Losses import Semantic_loss_functions
from EvaluationUtils import ConfusionMatrix
import sys
import os
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt

model_dir = '/home/kstef/kostas/saved_models'
MODEL_TYPE = str(sys.argv[1])
MODEL_NAME = str(sys.argv[2])
NUM_CLASSES = int(sys.argv[3])
BATCH_SIZE = 1
PREPROCESSING = 'default'

ignore_ids = [0,1,2,3,4,5,6,9,10,14,15,16,18,29,30]
class_names = ['road', 'sidewalk', 'building', 'wall', 'fence',
               'pole', 'traffic light', 'traffic sign', 'vegetation',
               'terrain', 'sky', 'person', 'rider', 'car', 'truck',
               'bus', 'train', 'motorcycle', 'bicycle', 'void']

#############################################################################################
data_path = ''  
img_path = 'leftImg8bit_trainvaltest/leftImg8bit'
label_path = 'gtFine_trainvaltest/gtFine'
val_path = '/val'
val = '/*'
img_type = '/*.png'
label_type = '/*_gtFine_labelIds.png'
img_val_path = data_path + img_path + val_path + val + img_type
label_val_path =  data_path + label_path + val_path + val + label_type

val_ds = Dataset(NUM_CLASSES, 'validation', PREPROCESSING, shuffle=False)
val_ds = val_ds.create(img_val_path, label_val_path, BATCH_SIZE, use_patches=False, augment=False)

s = Semantic_loss_functions()
loss = s.dice_loss

if NUM_CLASSES==34:
    ignore_class = ignore_ids
else:
    ignore_class = 19

mean_iou = MeanIoU(NUM_CLASSES, name='MeanIoU', ignore_class=None)
mean_iou_ignore = MeanIoU(NUM_CLASSES, name='MeanIoU_ignore', ignore_class=ignore_class)
metrics = [s.jacard_coef, mean_iou, mean_iou_ignore]

model_filepath = f'{model_dir}/{MODEL_TYPE}/{MODEL_NAME}'
model = tf.keras.models.load_model(model_filepath, compile=False)
model.compile(loss=loss, metrics=metrics)

# calculate parameters
trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
totalParams = trainableParams + nonTrainableParams

print(f'{MODEL_TYPE}/{MODEL_NAME}')
print('---------------------------------------')
print('Total params: ', totalParams)
print('Trainable params: ', trainableParams)
print('Non-Trainable params: ', nonTrainableParams)
print()

print('Model Evaluation')
score = model.evaluate(val_ds, verbose=2)
print()

confusion_matrix = mean_iou.get_confusion_matrix()
plt.rcParams["figure.figsize"] = (25,20)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                              display_labels=class_names
                              )

disp.plot(cmap='YlGnBu')
conf_matrix_dir = f'Confusion_matrix/{MODEL_TYPE}'
os.makedirs(conf_matrix_dir, exist_ok=True)
plt.savefig(f'{conf_matrix_dir}/{MODEL_NAME}.png', bbox_inches='tight')