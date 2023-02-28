import tensorflow as tf
import numpy as np
from keras import backend as K
import os
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from ..utils.SegmentationLosses import IoULoss, DiceLoss, TverskyLoss, FocalTverskyLoss, HybridLoss, FocalHybridLoss
from ..utils.DatasetUtils import Dataset
from ..utils.EvaluationUtils import MeanIoU
from ..utils.SegmentationModels import  Unet, Residual_Unet, Attention_Unet, Unet_plus, DeepLabV3plus
from argparse import ArgumentParser

parser = ArgumentParser('')
parser.add_argument('--data_path', type=str, nargs='?', required=True)
parser.add_argument('--model_type', type=str, nargs='?', required=True, choices=['Unet', 'Residual_Unet', 'Attention_Unet', 'Unet_plus', 'DeepLabV3plus'])
parser.add_argument('--model_name', type=str, nargs='?', required=True)
parser.add_argument('--backbone', type=str, nargs='?', default='None')
parser.add_argument('--output_stride', type=int, nargs='?', default=32)
parser.add_argument('--loss', type=str, nargs='?', default='dice', choices=['DiceLoss', 'IoULoss', 'TverskyLoss', 'FocalTverskyLoss', 'HybridLoss', 'FocalHybridLoss'])
parser.add_argument('--activation', type=str, nargs='?', default='relu')
parser.add_argument('--dropout', type=float, nargs='?', default=0.0)
parser.add_argument('--num_classes', type=int, nargs='?', default='20', choices=[20,34])
args = parser.parse_args()

DATA_PATH = args.data_path
MODEL_TYPE = args.model_type
MODEL_NAME = args.model_name
BACKBONE = args.backbone
OUTPUT_STRIDE = args.output_stride
LOSS = args.loss
BATCH_SIZE = 1
ACTIVATION = args.activation
DROPOUT_RATE = args.dropout
NUM_CLASSES = args.num_classes

FILTERS = [16,32,64,128,256]
INPUT_SHAPE = (1024, 2048, 3)


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

ignore_ids = [0,1,2,3,4,5,6,9,10,14,15,16,18,29,30]
if NUM_CLASSES==34:
    ignore_class = ignore_ids
else:
    ignore_class = 19
    
class_names = ['road', 'sidewalk', 'building', 'wall', 'fence',
               'pole', 'traffic light', 'traffic sign', 'vegetation',
               'terrain', 'sky', 'person', 'rider', 'car', 'truck',
               'bus', 'train', 'motorcycle', 'bicycle', 'void']

val_ds = Dataset(NUM_CLASSES, 'val', PREPROCESSING, shuffle=False)
val_ds = val_ds.create(DATA_PATH, 'all', BATCH_SIZE, use_patches=False, augment=False)

loss_func = eval(LOSS)
loss = loss_func()

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

model_dir = 'saved_models'
model_filepath = f'{model_dir}/{MODEL_TYPE}/{MODEL_NAME}'

# model_function = eval(MODEL_TYPE)
# model = model_function(input_shape=INPUT_SHAPE,
#                        filters=FILTERS,
#                        num_classes=NUM_CLASSES,
#                        output_stride=OUTPUT_STRIDE,
#                        activation=ACTIVATION,
#                        dropout_rate=DROPOUT_RATE,
#                        backbone_name=BACKBONE,
#                        freeze_backbone=True
#                        )

# model.load_weights(model_filepath)

model = tf.keras.models.load_model(model_filepath, compile=False)

model.compile(loss=loss, metrics=metrics)

print('Model Evaluation')
score = model.evaluate(val_ds, verbose=2)
print()
model.summary()

# SAVE CONFUSION MATRIX TO AS A PNG FILE
# confusion_matrix = mean_iou.get_confusion_matrix()
# plt.rcParams["figure.figsize"] = (25,20)
# disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
#                               display_labels=class_names
#                               )

# disp.plot(cmap='YlGnBu')
# conf_matrix_dir = f'Confusion_matrix/{MODEL_TYPE}'
# os.makedirs(conf_matrix_dir, exist_ok=True)
# plt.savefig(f'{conf_matrix_dir}/{MODEL_NAME}.png', bbox_inches='tight')