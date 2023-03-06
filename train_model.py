import os
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam, SGD, Adadelta, Nadam
from tensorflow_addons.optimizers import SGDW, AdamW, AdaBelief
from keras import mixed_precision
from SegmentationLosses import IoULoss, DiceLoss, TverskyLoss, FocalTverskyLoss, HybridLoss, FocalHybridLoss
from DatasetUtils import CityscapesDataset
from EvaluationUtils import MeanIoU
from SegmentationModels import  Unet, Residual_Unet, Attention_Unet, Unet_plus, DeepLabV3plus
from tensorflow_addons.optimizers import CyclicalLearningRate
from argparse import ArgumentParser

parser = ArgumentParser('')
parser.add_argument('--data_path', type=str, nargs='?', required=True)
parser.add_argument('--model_type', type=str, nargs='?', required=True, choices=['Unet', 'Residual_Unet', 'Attention_Unet', 'Unet_plus', 'DeepLabV3plus'])
parser.add_argument('--model_name', type=str, nargs='?', required=True)
parser.add_argument('--backbone', type=str, nargs='?', default='None')
parser.add_argument('--output_stride', type=int, nargs='?', default=32)
parser.add_argument('--optimizer', type=str, nargs='?', default='Adam', choices=['Adam', 'Adadelta', 'Nadam', 'AdaBelief', 'AdamW', 'SGDW'])
parser.add_argument('--unfreeze_at', type=str, nargs='?')
parser.add_argument('--loss', type=str, nargs='?', default='dice', choices=['DiceLoss', 'IoULoss', 'TverskyLoss', 'FocalTverskyLoss', 'HybridLoss', 'FocalHybridLoss'])
parser.add_argument('--batch_size', type=int, nargs='?', default='3')
parser.add_argument('--activation', type=str, nargs='?', default='relu')
parser.add_argument('--dropout', type=float, nargs='?', default=0.0)
parser.add_argument('--augment', type=bool, nargs='?', default=False)
parser.add_argument('--num_classes', type=int, nargs='?', default='20', choices=[20,34])
parser.add_argument('--epochs', type=int, nargs='?', default='20')
parser.add_argument('--final_epochs', type=int, nargs='?', default='60')
args = parser.parse_args()

# parse arguments
DATA_PATH = args.data_path
MODEL_TYPE = args.model_type
MODEL_NAME = args.model_name
BACKBONE = args.backbone
OUTPUT_STRIDE = args.output_stride
OPTIMIZER_STR = args.optimizer
UNFREEZE_AT = args.unfreeze_at
LOSS = args.loss
BATCH_SIZE = args.batch_size
ACTIVATION = args.activation
DROPOUT_RATE = args.dropout
AUGMENT = args.augment
NUM_CLASSES = args.num_classes
EPOCHS = args.epochs
FINAL_EPOCHS = args.final_epochs

# Number of dataset elements
COUNT = -1

# define other constants
FILTERS = [16,32,64,128,256]
INPUT_SHAPE = (1024, 2048, 3)
initial_lr = 0.001
end_lr = 0.0001

if BACKBONE == 'None':
    PREPROCESSING = 'default'
    BACKBONE = None
elif 'ResNet' in BACKBONE:
    PREPROCESSING = 'ResNet'
    if 'V2' in BACKBONE:
        PREPROCESSING = 'ResNetV2'
elif 'EfficientNet' in BACKBONE:
    PREPROCESSING = 'EfficientNet'
elif 'EfficientNetV2' in BACKBONE:
    PREPROCESSING = 'EfficientNetV2'
elif 'MobileNet' == BACKBONE:
    PREPROCESSING = 'MobileNet'
elif 'MobileNetV2' == BACKBONE:
    PREPROCESSING = 'MobileNetV2'
elif 'MobileNetV3' in BACKBONE:
    PREPROCESSING = 'MobileNetV3'
elif 'RegNet' in BACKBONE:
    PREPROCESSING = 'RegNet'
else:
    raise ValueError(f'Enter a valid Backbone name, {BACKBONE} is invalid.')

ignore_ids = [0,1,2,3,4,5,6,9,10,14,15,16,18,29,30]
if NUM_CLASSES==34:
    ignore_class = ignore_ids
else:
    ignore_class = 19

# ---------------------------Create Dataset stream--------------------------------
train_ds = CityscapesDataset(NUM_CLASSES, 'train', PREPROCESSING, shuffle=True)
train_ds = train_ds.create(DATA_PATH, 'all', BATCH_SIZE, COUNT, augment=False)

val_ds = CityscapesDataset(NUM_CLASSES, 'val', PREPROCESSING, shuffle=False)
val_ds = val_ds.create(DATA_PATH, 'all', BATCH_SIZE, COUNT, augment=False)

steps_per_epoch = train_ds.cardinality().numpy()

# ---------------------------------------CALLBACKS-------------------------------------------
if BACKBONE is None:
    save_best_only = True
    save_freq = 'epoch'
else:
    save_best_only = False
    save_freq = int(EPOCHS*steps_per_epoch) # save the model only at the last epoch of the main training

checkpoint_filepath = f'saved_models/{MODEL_TYPE}/{MODEL_NAME}'
model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,                                           
                                            save_weights_only=False,
                                            monitor='val_MeanIoU_ignore',
                                            mode='max',
                                            save_freq=save_freq, 
                                            save_best_only=save_best_only,
                                            verbose=0)

log_dir = f'Tensorboard_logs/{MODEL_TYPE}/{MODEL_NAME}'
tensorboard_callback = TensorBoard(log_dir=log_dir,
                                   histogram_freq=0,
                                   write_graph=False,
                                   write_steps_per_second=False)

callbacks = [model_checkpoint_callback, tensorboard_callback]
# -------------------------------------------------------------------------------------------

loss_func = eval(LOSS)
loss = loss_func()

schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=15*992,
    end_learning_rate=end_lr,
    power=2,
    cycle=False,
    name=None
    )

step = tf.Variable(0, trainable=False)

lr_schedule = schedule(step)

wd = 5e-5
wd_schedule = lambda : wd * schedule(step)

optimizer_dict = {
    'Adam' : Adam(initial_lr) if BACKBONE is None else Adam(lr_schedule),
    'Nadam' : Nadam(lr_schedule),
    'Adadelta' : Adadelta(lr_schedule),
    'AdamW' : AdamW(learning_rate=lr_schedule, weight_decay=wd),
    'AdaBelief' : AdaBelief(learning_rate=lr_schedule, weight_decay=wd),
    'SGDW' : SGDW(learning_rate=lr_schedule, weight_decay=wd, momentum=0.9)
}

optimizer = optimizer_dict[OPTIMIZER_STR]

mean_iou = MeanIoU(NUM_CLASSES, name='MeanIoU', ignore_class=None)
mean_iou_ignore = MeanIoU(NUM_CLASSES, name='MeanIoU_ignore', ignore_class=ignore_class)
metrics = [mean_iou_ignore]

# Instantiate Model
model_function = eval(MODEL_TYPE)
model = model_function(input_shape=INPUT_SHAPE,
                       filters=FILTERS,
                       num_classes=NUM_CLASSES,
                       output_stride=OUTPUT_STRIDE,
                       activation=ACTIVATION,
                       dropout_rate=DROPOUT_RATE,
                       backbone_name=BACKBONE,
                       freeze_backbone=True
                       )
    
model.summary()

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=EPOCHS,
                    callbacks = callbacks,
                    verbose = 1
                    )

# FINE TUNE MODEL
if BACKBONE is not None:
    #* After unfreezing the final backbone weights the barch size might need to be reduced to
    #* prevent OOM. Re-define the dataset streams with new batch size
    train_ds = CityscapesDataset(NUM_CLASSES, 'train', PREPROCESSING, shuffle=True)
    train_ds = train_ds.create(DATA_PATH, 'all', BATCH_SIZE-1, COUNT, augment=False)

    val_ds = CityscapesDataset(NUM_CLASSES, 'val', PREPROCESSING, shuffle=False)
    val_ds = val_ds.create(DATA_PATH, 'all', BATCH_SIZE-1, COUNT, augment=False)
    
    # Re-define checkpoint callback to save only the best model
    checkpoint_filepath = f'saved_models/{MODEL_TYPE}/{MODEL_NAME}'
    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,                                           
                                                save_weights_only=False,
                                                monitor='val_MeanIoU_ignore',
                                                mode='max',
                                                save_best_only=True,
                                                verbose=0)
    
    callbacks = [model_checkpoint_callback, tensorboard_callback]
    
    # instantiate model again with the last part of the encoder (Backbone) un-frozen
    model = model_function(input_shape=INPUT_SHAPE,
                           filters=FILTERS,
                           num_classes=NUM_CLASSES,
                           output_stride=OUTPUT_STRIDE,
                           activation=ACTIVATION,
                           dropout_rate=DROPOUT_RATE,
                           backbone_name=BACKBONE,
                           freeze_backbone=False,
                           unfreeze_at=UNFREEZE_AT
                           )
    
    # load the saved weights into the model to fine tune the high level features of the feature extractor
    # Fine tune the encoder network with a lower learning rate
    model.load_weights(checkpoint_filepath)
    
    model.summary()
    
    optimizer_dict = {
    'Adam' : Adam(end_lr),
    'Nadam' : Nadam(end_lr),
    'Adadelta' : Adadelta(end_lr),
    'AdamW' : AdamW(learning_rate=end_lr, weight_decay=wd),
    'AdaBelief' : AdaBelief(learning_rate=end_lr, weight_decay=wd),
    'SGDW' : SGDW(learning_rate=end_lr, weight_decay=wd, momentum=0.9)
    }

    optimizer = optimizer_dict[OPTIMIZER_STR]
    
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        initial_epoch=EPOCHS,
                        epochs=FINAL_EPOCHS,
                        callbacks = callbacks,
                        verbose = 1
                        )