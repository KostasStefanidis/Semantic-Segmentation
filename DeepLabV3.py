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
from SegmentationModels import DeepLabV3plus
from tensorflow_addons.optimizers import CyclicalLearningRate
from argparse import ArgumentParser
from keras.optimizers.schedules import PolynomialDecay

parser = ArgumentParser('')
parser.add_argument('--data_path', type=str, nargs='?', required=True)
parser.add_argument('--model_type', type=str, nargs='?', required=True)
parser.add_argument('--model_name', type=str, nargs='?', required=True)
parser.add_argument('--num_classes', type=int, nargs='?', default='20', choices=[20,34])
parser.add_argument('--preprocessing', type=str, nargs='?', default='default', choices=['default', 'EfficientNet', 'EfficientNetV2', 'ResNet'])
parser.add_argument('--epochs', type=int, nargs='?', default='60')
parser.add_argument('--batch_size', type=int, nargs='?', default='3')
args = parser.parse_args()

data_path = args.data_path
MODEL_TYPE = args.model_type
MODEL_NAME = args.model_name
NUM_CLASSES = args.num_classes
PREPROCESSING = args.preprocessing
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
FINAL_EPOCHS = 60
FILTERS = [16,32,64,128,256]
INPUT_SHAPE = (1024, 2048, 3)
ACTIVATION = 'leaky_relu'
DROPOUT_RATE = 0.1
DROPOUT_OFFSET = 0.02
BACKBONE = PREPROCESSING
BACKBONE_VERSION = 'B4'
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

# csv_logs_dir = f'CSV_logs/{MODEL_TYPE}/{MODEL_NAME}.csv'
# os.makedirs(csv_logs_dir, exist_ok=True)
# csv_callback = CSVLogger(csv_logs_dir)

callbacks = [model_checkpoint_callback, tensorboard_callback]
# -------------------------------------------------------------------------------------------

train_ds = Dataset(NUM_CLASSES, 'train', PREPROCESSING, shuffle=True)
train_ds = train_ds.create(data_path, 'all', BATCH_SIZE, use_patches=False, augment=False)

val_ds = Dataset(NUM_CLASSES, 'val', PREPROCESSING, shuffle=False)
val_ds = val_ds.create(data_path, 'all', BATCH_SIZE, use_patches=False, augment=False)

steps_per_epoch = 992
MIN_LR = 1e-4
MAX_LR = 2e-3
lr_schedule = CyclicalLearningRate(initial_learning_rate=MIN_LR,
                                   maximal_learning_rate=MAX_LR,
                                   scale_fn=lambda x: 0.95**x,
                                   step_size=4*steps_per_epoch,
                                   )
lr_schedule = PolynomialDecay(initial_learning_rate=MAX_LR,
                              end_learning_rate= MIN_LR,
                              decay_steps=EPOCHS*steps_per_epoch,
                              power=2,
                              cycle=False)

#loss = IoULoss(class_weights=np.load('class_weights/class_weights.npy'))
loss = HybridLoss()
#loss = FocalHybridLoss(beta=0.5)

optimizer = Adam(base_lr)
# optimizer = AdamW(1e-5)
# optimizer = AdaBelief()
# optimizer = SGD(learning_rate=0.01, momentum=0.9)
#optimizer = SGDW(learning_rate=0.01, momentum=0.9, weight_decay=1e-5)

if NUM_CLASSES==34:
    ignore_class = ignore_ids
else:
    ignore_class = 19

mean_iou = MeanIoU(NUM_CLASSES, name='MeanIoU', ignore_class=None)
mean_iou_ignore = MeanIoU(NUM_CLASSES, name='MeanIoU_ignore', ignore_class=ignore_class)
metrics = [mean_iou]

model = DeepLabV3plus(input_shape=INPUT_SHAPE,
                      filters=FILTERS,
                      num_classes=NUM_CLASSES,
                      activation='leaky_relu',
                      dropout_rate=DROPOUT_RATE,
                      dropout_type='spatial',
                      scale_dropout=False,
                      dropout_offset=DROPOUT_OFFSET,
                      backbone_name=BACKBONE_NAME,
                      freeze_backbone=True
                      )

model.summary()
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# train model with backbone frozen
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=EPOCHS,
                    callbacks = callbacks,
                    verbose = 1
                    )

# Fine tune the model unfreezing a part of the backbone
fine_tune_model = DeepLabV3plus(input_shape=INPUT_SHAPE,
                                filters=FILTERS,
                                num_classes=NUM_CLASSES,
                                activation='leaky_relu',
                                dropout_rate=DROPOUT_RATE,
                                dropout_type='spatial',
                                scale_dropout=False,
                                dropout_offset=DROPOUT_OFFSET,
                                backbone_name=BACKBONE_NAME,
                                freeze_backbone=False,
                                unfreeze_at='block6a_expand_activation'
                                )

fine_tune_model.load_weights(checkpoint_filepath)
fine_tune_model.summary()

optimizer = Adam(base_lr/10)
fine_tune_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

BATCH_SIZE = 1
train_ds = Dataset(NUM_CLASSES, 'train', PREPROCESSING, shuffle=True)
train_ds = train_ds.create(data_path, 'all', BATCH_SIZE, use_patches=False, augment=False)

val_ds = Dataset(NUM_CLASSES, 'val', PREPROCESSING, shuffle=False)
val_ds = val_ds.create(data_path, 'all', BATCH_SIZE, use_patches=False, augment=False)

history = fine_tune_model.fit(train_ds,
                            validation_data=val_ds,
                            initial_epoch=EPOCHS,
                            epochs=FINAL_EPOCHS,
                            callbacks = callbacks,
                            verbose = 1
                            )