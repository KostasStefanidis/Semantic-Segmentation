import os
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam, SGD
from tensorflow_addons.optimizers import SGDW, AdamW, AdaBelief
from keras import mixed_precision
from SegmentationLosses import IoULoss, DiceLoss, TverskyLoss, FocalTverskyLoss, HybridLoss, FocalHybridLoss
from DatasetUtils import Dataset
from EvaluationUtils import MeanIoU
from SegmentationModels import  Unet, Residual_Unet, Attention_Unet, DeepLabV3plus
from tensorflow_addons.optimizers import CyclicalLearningRate
from argparse import ArgumentParser

parser = ArgumentParser('')
parser.add_argument('--data_path', type=str, nargs='?', required=True)
parser.add_argument('--model_type', type=str, nargs='?', required=True, choices=['Unet', 'Residual_Unet', 'Attention_Unet', 'DeepLabV3plus'])
parser.add_argument('--model_name', type=str, nargs='?', required=True)
parser.add_argument('--backbone', type=str, nargs='?', default='None', choices=['None', 'EfficientNet', 'EfficientNetV2', 'ResNet'])
parser.add_argument('--loss', type=str, nargs='?', default='dice', choices=['DiceLoss', 'IoULoss', 'TverskyLoss', 'FocalTverskyLoss', 'HybridLoss', 'FocalHybridLoss'])
parser.add_argument('--num_classes', type=int, nargs='?', default='20', choices=[20,34])
parser.add_argument('--epochs', type=int, nargs='?', default='60')
parser.add_argument('--batch_size', type=int, nargs='?', default='3')
args = parser.parse_args()

# parse arguments
data_path = args.data_path
MODEL_TYPE = args.model_type
MODEL_NAME = args.model_name
NUM_CLASSES = args.num_classes
BACKBONE = args.backbone
LOSS = args.loss
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

# define other constants
FINAL_EPOCHS = 60
FILTERS = [16,32,64,128,256]
INPUT_SHAPE = (1024, 2048, 3)
ACTIVATION = 'leaky_relu'
DROPOUT_RATE = 0.1
DROPOUT_OFFSET = 0.02

if BACKBONE == 'None':
    PREPROCESSING = 'default'
    BACKBONE_NAME = None
else:
    PREPROCESSING = BACKBONE
    BACKBONE_VERSION = 'S'
    BACKBONE_NAME = BACKBONE + BACKBONE_VERSION

ignore_ids = [0,1,2,3,4,5,6,9,10,14,15,16,18,29,30]
if NUM_CLASSES==34:
    ignore_class = ignore_ids
else:
    ignore_class = 19

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

callbacks = [model_checkpoint_callback, tensorboard_callback]
# -------------------------------------------------------------------------------------------

# Create Dataset pipeline
train_ds = Dataset(NUM_CLASSES, 'train', PREPROCESSING, shuffle=True)
train_ds = train_ds.create(data_path, 'all', BATCH_SIZE, use_patches=False, augment=False)

val_ds = Dataset(NUM_CLASSES, 'val', PREPROCESSING, shuffle=False)
val_ds = val_ds.create(data_path, 'all', BATCH_SIZE, use_patches=False, augment=False)

# Instantiate Model
model_function = eval(MODEL_TYPE)
model = model_function(input_shape=INPUT_SHAPE,
                        filters=FILTERS,
                        num_classes=NUM_CLASSES,
                        activation=ACTIVATION,
                        dropout_rate=DROPOUT_RATE,
                        dropout_type='normal',
                        backbone_name=BACKBONE_NAME,
                        freeze_backbone=True
                        )
    
model.summary()

loss_func = eval(LOSS)
loss = loss_func()

optimizer = Adam()

mean_iou = MeanIoU(NUM_CLASSES, name='MeanIoU', ignore_class=None)
mean_iou_ignore = MeanIoU(NUM_CLASSES, name='MeanIoU_ignore', ignore_class=ignore_class)
metrics = [mean_iou, mean_iou_ignore]

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=EPOCHS,
                    callbacks = callbacks,
                    verbose = 1
                    )