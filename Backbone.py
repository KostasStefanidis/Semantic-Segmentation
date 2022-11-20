import os
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from keras.optimizers import Adam, SGD
from tensorflow_addons.optimizers import SGDW, AdamW, AdaBelief
from keras import mixed_precision
from SegmentationLosses import IoULoss, DiceLoss, TverskyLoss, FocalTverskyLoss, HybridLoss, FocalHybridLoss
from DatasetUtils import Dataset
from EvaluationUtils import MeanIoU
from SegmentationModels import Residual_Unet, DeepLabV3plus
from tensorflow_addons.optimizers import CyclicalLearningRate
from argparse import ArgumentParser
from keras.optimizers.schedules import PolynomialDecay

parser = ArgumentParser('')
parser.add_argument('-t', type=str, nargs='?', required=True)
parser.add_argument('-m', type=str, nargs='?', required=True)
parser.add_argument('-n', type=int, nargs='?', default='20', choices=[20,34])
parser.add_argument('-p', type=str, nargs='?', default='default', choices=['default', 'EfficientNet', 'ResNet'])
parser.add_argument('-e', type=int, nargs='?', default='30')
parser.add_argument('-b', type=int, nargs='?', default='3')
args = parser.parse_args()

MODEL_TYPE = args.t
MODEL_NAME = args.m
NUM_CLASSES = args.n
PREPROCESSING = args.p
EPOCHS = args.e
BATCH_SIZE = args.b
FINAL_EPOCHS = 60
FILTERS = [16,32,64,128,256]
INPUT_SHAPE = (1024, 2048, 3)
ACTIVATION = 'leaky_relu'
DROPOUT_RATE = 0
DROPOUT_OFFSET = 0.02
BACKBONE = PREPROCESSING
BACKBONE_VERSION = 'V2S'
BACKBONE_NAME = BACKBONE + BACKBONE_VERSION

base_lr = 0.001

ignore_ids = [0,1,2,3,4,5,6,9,10,14,15,16,18,29,30]

data_path = ''

# -------------------------------CALLBACKS---------------------------------------------------
checkpoint_filepath = f'saved_models/{MODEL_TYPE}/{MODEL_NAME}'
model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,                                           
                                            save_weights_only=False,
                                            monitor='val_MeanIoU',
                                            mode='max',
                                            save_best_only=True,
                                            verbose=0)

log_dir = f'Tensorboard_logs/{MODEL_TYPE}/{MODEL_NAME}'
tensorboard_callback = TensorBoard(log_dir=log_dir,
                                histogram_freq=0,
                                write_graph=False,
                                write_steps_per_second=False)

callbacks = [tensorboard_callback]
# -------------------------------------------------------------------------------------------

train_ds = Dataset(NUM_CLASSES, 'train', PREPROCESSING, shuffle=True)
train_ds = train_ds.create(data_path, 'all', BATCH_SIZE, use_patches=False, augment=True)

val_ds = Dataset(NUM_CLASSES, 'val', PREPROCESSING, shuffle=False)
val_ds = val_ds.create(data_path, 'all', BATCH_SIZE, use_patches=False, augment=False)

loss = HybridLoss()

optimizer = Adam()

if NUM_CLASSES==34:
    ignore_class = ignore_ids
else:
    ignore_class = 19

mean_iou = MeanIoU(NUM_CLASSES, name='MeanIoU', ignore_class=None)
mean_iou_ignore = MeanIoU(NUM_CLASSES, name='MeanIoU_ignore', ignore_class=ignore_class)
metrics = [mean_iou, mean_iou_ignore]

model = DeepLabV3plus(input_shape=INPUT_SHAPE,
                      filters=FILTERS,
                      num_classes=NUM_CLASSES,
                      activation='leaky_relu',
                      dropout_rate=DROPOUT_RATE,
                      backbone_name=BACKBONE_NAME,
                      freeze_backbone=True
                      )

model.summary()
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
callbacks = [model_checkpoint_callback, tensorboard_callback]
# train model with backbone frozen
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=EPOCHS,
                    callbacks = callbacks,
                    verbose = 1
                    )