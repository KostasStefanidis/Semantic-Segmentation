import tensorflow as tf
import numpy as np
from keras import backend as K
from DatasetUtils import Dataset
from EvaluationUtils import MeanIoU
import os
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from SegmentationLosses import IoULoss, DiceLoss, TverskyLoss, FocalTverskyLoss, HybridLoss, FocalHybridLoss
from argparse import ArgumentParser

parser = ArgumentParser('')
parser.add_argument('-t', type=str, nargs='?', required=True)
parser.add_argument('-m', type=str, nargs='?', required=True)
parser.add_argument('-n', type=int, nargs='?', default='20', choices=[20,34])
parser.add_argument('-p', type=str, nargs='?', default='default', choices=['default', 'EfficientNet', 'ResNet'])
args = parser.parse_args()

MODEL_TYPE = args.t
MODEL_NAME = args.m
NUM_CLASSES = args.n
PREPROCESSING = args.p
BATCH_SIZE = 1

ignore_ids = [0,1,2,3,4,5,6,9,10,14,15,16,18,29,30]
class_names = ['road', 'sidewalk', 'building', 'wall', 'fence',
               'pole', 'traffic light', 'traffic sign', 'vegetation',
               'terrain', 'sky', 'person', 'rider', 'car', 'truck',
               'bus', 'train', 'motorcycle', 'bicycle', 'void']

model_dir = '/home/kstef/kostas/saved_models'
#############################################################################################
data_path = ''  

val_ds = Dataset(NUM_CLASSES, 'val', PREPROCESSING, shuffle=False)
val_ds = val_ds.create(data_path, 'all', BATCH_SIZE, use_patches=False, augment=False)

if NUM_CLASSES==34:
    ignore_class = ignore_ids
else:
    ignore_class = 19

loss = HybridLoss()

mean_iou = MeanIoU(NUM_CLASSES, name='MeanIoU', ignore_class=None)
mean_iou_ignore = MeanIoU(NUM_CLASSES, name='MeanIoU_ignore', ignore_class=ignore_class)
iou_0 = MeanIoU(NUM_CLASSES, name='IoU_road', target_class_ids=[0])
iou_1 = MeanIoU(NUM_CLASSES, name='IoU_sidewalk', target_class_ids=[1])
iou_2 = MeanIoU(NUM_CLASSES, name='IoU_building', target_class_ids=[2])
iou_3 = MeanIoU(NUM_CLASSES, name='IoU_wall', target_class_ids=[3])
iou_4 = MeanIoU(NUM_CLASSES, name='IoU_fence', target_class_ids=[4])
iou_5 = MeanIoU(NUM_CLASSES, name='IoU_pole', target_class_ids=[5])
iou_6 = MeanIoU(NUM_CLASSES, name='IoU_traffic light', target_class_ids=[6])
iou_7 = MeanIoU(NUM_CLASSES, name='IoU_traffic sign', target_class_ids=[7])
iou_8 = MeanIoU(NUM_CLASSES, name='IoU_vegetation', target_class_ids=[8])
iou_9 = MeanIoU(NUM_CLASSES, name='IoU_terrain', target_class_ids=[9])
iou_10 = MeanIoU(NUM_CLASSES, name='IoU_sky', target_class_ids=[10])
iou_11 = MeanIoU(NUM_CLASSES, name='IoU_person', target_class_ids=[11])
iou_12 = MeanIoU(NUM_CLASSES, name='IoU_rider', target_class_ids=[12])
iou_13 = MeanIoU(NUM_CLASSES, name='IoU_car', target_class_ids=[13])
iou_14 = MeanIoU(NUM_CLASSES, name='IoU_truck', target_class_ids=[14])
iou_15 = MeanIoU(NUM_CLASSES, name='IoU_bus', target_class_ids=[15])
iou_16 = MeanIoU(NUM_CLASSES, name='IoU_train', target_class_ids=[16])
iou_17 = MeanIoU(NUM_CLASSES, name='IoU_motorcycle', target_class_ids=[17])
iou_18 = MeanIoU(NUM_CLASSES, name='IoU_bicycle', target_class_ids=[18])
metrics = [mean_iou, mean_iou_ignore,
           iou_0, iou_1, iou_2, iou_3,
           iou_4, iou_5, iou_6, iou_7,
           iou_8, iou_9, iou_10, iou_11,
           iou_12, iou_13, iou_14, iou_15,
           iou_16, iou_17, iou_18]

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