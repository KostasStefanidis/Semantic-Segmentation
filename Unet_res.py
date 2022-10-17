import sys
import os
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow_addons.optimizers import SGDW, AdamW, AdaBelief
from tensorflow.keras import mixed_precision
from SegmentationLosses import IoULoss, DiceLoss, TverskyLoss, FocalTverskyLoss
from DatasetUtils import Dataset
from EvaluationUtils import MeanIoU
from SegmentationModels import Residual_Unet
from tensorflow_addons.optimizers import CyclicalLearningRate
#mixed_precision.set_global_policy('mixed_float16') # -> can use larger batch size (double)

MODEL_TYPE = sys.argv[1]
MODEL_NAME = sys.argv[2]
NUM_CLASSES = int(sys.argv[3])
PREPROCESSING = 'default'
EPOCHS = 60
FILTERS = [16,32,64,128,256]
INPUT_SHAPE = (1024, 2048, 3)
BATCH_SIZE = 3
ACTIVATION = 'leaky_relu'
DROPOUT_RATE = 0.0
DROPOUT_OFFSET = 0.02

ignore_ids = [0,1,2,3,4,5,6,9,10,14,15,16,18,29,30]

##########################################################################
data_path = ''  
img_path = 'leftImg8bit_trainvaltest/leftImg8bit'
label_path = 'gtFine_trainvaltest/gtFine'
train_path = '/train'
val_path = '/val'
test_path = '/test'
train = '/*'
val = '/*'
test = '/*'
img_type = '/*.png'
label_type = '/*_gtFine_labelIds.png'

img_train_path = data_path + img_path + train_path + train + img_type
img_val_path = data_path + img_path + val_path + val + img_type
img_test_path = data_path + img_path + test_path + test + img_type

label_train_path = data_path + label_path + train_path + train + label_type
label_val_path =  data_path + label_path + val_path + val + label_type
label_test_path = data_path + label_path + test_path + test + label_type

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
train_ds = train_ds.create(img_train_path, label_train_path, BATCH_SIZE, use_patches=False, augment=False)

val_ds = Dataset(NUM_CLASSES, 'validation', PREPROCESSING, shuffle=False)
val_ds = val_ds.create(img_val_path, label_val_path, BATCH_SIZE, use_patches=False, augment=False)

model = Residual_Unet(input_shape=INPUT_SHAPE,
                      filters=FILTERS,
                      num_classes=NUM_CLASSES,
                      activation=ACTIVATION,
                      dropout_rate=DROPOUT_RATE,
                      dropout_type='normal',
                      scale_dropout=False,
                      dropout_offset=DROPOUT_OFFSET,
                      )

model.summary()

loss = IoULoss()

optimizer = AdamW(weight_decay=1e-5)
#optimizer = Adam()

if NUM_CLASSES==34:
    ignore_class = ignore_ids
else:
    ignore_class = 19

mean_iou = MeanIoU(NUM_CLASSES, name='MeanIoU', ignore_class=None)
mean_iou_ignore = MeanIoU(NUM_CLASSES, name='MeanIoU_ignore', ignore_class=ignore_class)
metrics = [mean_iou]

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=EPOCHS,
                    callbacks = callbacks,
                    verbose = 1
                    )