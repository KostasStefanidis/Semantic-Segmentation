import os
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam, SGD, Adadelta
from tensorflow_addons.optimizers import SGDW, AdamW, AdaBelief
from SegmentationLosses import IoULoss, DiceLoss, TverskyLoss, FocalTverskyLoss, HybridLoss, FocalHybridLoss
from CityscapesUtils import CityscapesDataset
from MapillaryUtils import MapillaryDataset
from EvaluationUtils import MeanIoU
from SegmentationModels import  Unet, Residual_Unet, Attention_Unet, Unet_plus, DeepLabV3plus
from tensorflow_addons.optimizers import CyclicalLearningRate
from argparse import ArgumentParser
import yaml

parser = ArgumentParser('')
parser.add_argument('--config', type=str, nargs='?')
parser.add_argument('--data_path', type=str, nargs='?')
parser.add_argument('--dataset', type=str, nargs='?', default='Cityscapes', choices=['Cityscapes', 'Mapillary'])
parser.add_argument('--model_type', type=str, nargs='?', choices=['Unet', 'Residual_Unet', 'Attention_Unet', 'Unet_plus', 'DeepLabV3plus'])
parser.add_argument('--model_name', type=str, nargs='?')
parser.add_argument('--backbone', type=str, nargs='?', default='None')
parser.add_argument('--output_stride', type=int, nargs='?', default=32)
parser.add_argument('--unfreeze_at', type=str, nargs='?')
parser.add_argument('--activation', type=str, nargs='?', default='relu')
parser.add_argument('--dropout', type=float, nargs='?', default=0.0)
parser.add_argument('--optimizer', type=str, nargs='?', default='Adam', choices=['Adam', 'Adadelta', 'Nadam', 'AdaBelief', 'AdamW', 'SGDW'])
parser.add_argument('--loss', type=str, nargs='?', default='FocalHybridLoss', choices=['DiceLoss', 'IoULoss', 'TverskyLoss', 'FocalTverskyLoss', 'HybridLoss', 'FocalHybridLoss'])
parser.add_argument('--batch_size', type=int, nargs='?', default='3')
parser.add_argument('--augment', type=bool, nargs='?', default=False)
parser.add_argument('--epochs', type=int, nargs='?', default='20')
parser.add_argument('--final_epochs', type=int, nargs='?', default='60')
args = parser.parse_args()


if args.config is None:
    # parse arguments
    print('Reading configuration from cmd args')
    DATA_PATH = args.data_path
    DATASET = args.dataset
    MODEL_TYPE = args.model_type
    MODEL_NAME = args.model_name
    BACKBONE = args.backbone
    OUTPUT_STRIDE = args.output_stride
    OPTIMIZER_NAME = args.optimizer
    UNFREEZE_AT = args.unfreeze_at
    LOSS = args.loss
    BATCH_SIZE = args.batch_size
    ACTIVATION = args.activation
    DROPOUT_RATE = args.dropout
    AUGMENT = args.augment
    EPOCHS = args.epochs
    FINAL_EPOCHS = args.final_epochs
    
else:
    # Read YAML file
    print('Reading configuration from config yaml')
    
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)

    LOGS_DIR = config['logs_dir']

    model_config = config['model']
    dataset_config = config['dataset']
    train_config = config['train_config']

    # Dataset Configuration
    DATASET = dataset_config['name']
    DATA_PATH = dataset_config['path']
    VERSION = dataset_config['version']
    NUM_TRAIN_IMAGES = dataset_config['num_train_images']
    NUM_EVAL_IMAGES = dataset_config['num_eval_images']
    CACHE = dataset_config['cache']
    CACHE_FILE = dataset_config['cache_file']
    SEED = dataset_config['seed']

    # Model Configuration
    MODEL_TYPE = model_config['architecture']
    MODEL_NAME = model_config['name']
    BACKBONE = model_config['backbone']
    UNFREEZE_AT = model_config['unfreeze_at']
    INPUT_SHAPE = model_config['input_shape']
    OUTPUT_STRIDE = model_config['output_stride']
    FILTERS = model_config['filters']
    ACTIVATION = model_config['activation']
    DROPOUT_RATE = model_config['dropout_rate']

    # Training Configuration
    PRETRAINED_WEIGHTS = model_config['pretrained_weights']
    
    BATCH_SIZE_PER_REPLICA = train_config['batch_size']
    EPOCHS = train_config['epochs']
    FINAL_EPOCHS = train_config['final_epochs']
    AUGMENT = train_config['augment']
    MIXED_PRECISION = train_config['mixed_precision']
    LOSS = train_config['loss']

    optimizer_config = train_config['optimizer']
    OPTIMIZER_NAME = optimizer_config['name']
    WEIGHT_DECAY = optimizer_config['weight_decay']
    MOMENTUM = optimizer_config['momentum']
    START_LR = optimizer_config['schedule']['start_lr']
    END_LR = optimizer_config['schedule']['end_lr']
    LR_DECAY_EPOCHS = optimizer_config['schedule']['decay_epochs']
    POWER = optimizer_config['schedule']['power']

    DISTRIBUTE_STRATEGY = train_config['distribute']['strategy']
    DEVICES = train_config['distribute']['devices']


if DATASET == 'Cityscapes':
    NUM_CLASSES = 20
    IGNORE_CLASS = 19
    INPUT_SHAPE = (1024, 2048, 3)
elif DATASET == 'Mapillary':
    INPUT_SHAPE = (1024, 1856, 3)
    if VERSION == 'v1.2':
        NUM_CLASSES = 64
        IGNORE_CLASS = 63
    elif VERSION == 'v2.0':
        NUM_CLASSES = 118
        IGNORE_CLASS = 117
    else:
        raise ValueError('Version of the Mapillary Vistas dataset should be either v1.2 or v2.0!')
else:
    raise ValueError(F'{DATASET} dataset is invalid. Available Datasets are: Cityscapes, Mapillary!')

# Define preprocessing according to the Backbone
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

if MIXED_PRECISION:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
if len(DEVICES) > 1: 
    # if more than 1 devices are specified in the configuration
    # -> use Mirrored Strategy with specified devices
    strategy = tf.distribute.MirroredStrategy(DEVICES)
else:
    # Use the Default Strategy
    strategy = tf.distribute.get_strategy()

BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync


# ---------------------------Create Dataset stream--------------------------------
if DATASET == 'Cityscapes':
    train_ds = CityscapesDataset(num_classes=NUM_CLASSES, 
                                 split='train', 
                                 preprocessing=PREPROCESSING, 
                                 shuffle=True, 
                                 cache=CACHE,
                                 cache_file=CACHE_FILE
                                 )
    train_ds = train_ds.create(DATA_PATH, 'all', BATCH_SIZE, NUM_TRAIN_IMAGES, augment=False, seed=SEED)

    val_ds = CityscapesDataset(num_classes=NUM_CLASSES, 
                               split='val', 
                               preprocessing=PREPROCESSING, 
                               shuffle=False,
                               cache=CACHE,
                               cache_file=CACHE_FILE
                               )
    val_ds = val_ds.create(DATA_PATH, 'all', BATCH_SIZE, NUM_EVAL_IMAGES, seed=SEED)
    
elif DATASET == 'Mapillary':
    train_ds = MapillaryDataset(height=1024, width=1856,
                                split='training',
                                preprocessing=PREPROCESSING,
                                version=VERSION,
                                shuffle=True,
                                )
    train_ds = train_ds.create(DATA_PATH, BATCH_SIZE, NUM_TRAIN_IMAGES, augment=False, seed=SEED)

    val_ds = MapillaryDataset(height=1024, width=1856,
                              split='validation',
                              preprocessing=PREPROCESSING,
                              version=VERSION,
                              shuffle=False)
    val_ds = val_ds.create(DATA_PATH, BATCH_SIZE, NUM_EVAL_IMAGES, seed=SEED)

# Make the data pipeline distribute aware
train_ds = strategy.experimental_distribute_dataset(train_ds)
val_ds = strategy.experimental_distribute_dataset(val_ds)

steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
validation_steps = tf.data.experimental.cardinality(val_ds).numpy()


# ---------------------------------------CALLBACKS-------------------------------------------
if BACKBONE is None:
    save_best_only = True
    save_freq = 'epoch'
else:
    save_best_only = False
    save_freq = int(EPOCHS*steps_per_epoch) # save the model only at the last epoch of the main training

checkpoint_filepath = f'saved_models/{DATASET}/{MODEL_TYPE}/{MODEL_NAME}'
model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,                                           
                                            save_weights_only=False,
                                            monitor='val_MeanIoU_ignore',
                                            mode='max',
                                            save_freq='epoch', 
                                            save_best_only=save_best_only,
                                            verbose=0)
#{LOGS_DIR}/
tensorboard_log_dir = f'Tensorboard_logs/{DATASET}/{MODEL_TYPE}/{MODEL_NAME}'
tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir,
                                   histogram_freq=0,
                                   write_graph=False,
                                   write_steps_per_second=False)

callbacks = [model_checkpoint_callback, tensorboard_callback]
# -------------------------------------------------------------------------------------------

lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=START_LR,
    decay_steps=LR_DECAY_EPOCHS*steps_per_epoch,
    end_learning_rate=END_LR,
    power=POWER,
    cycle=False,
    name=None
    )

optimizer_dict = {
    'Adam' : Adam(lr_schedule),
    'Adadelta' : Adadelta(lr_schedule),
    'AdamW' : AdamW(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY),
    'AdaBelief' : AdaBelief(learning_rate=lr_schedule),
    'SGDW' : SGDW(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
}

with strategy.scope():
    loss_func = eval(LOSS)
    loss = loss_func()
    
    optimizer = optimizer_dict[OPTIMIZER_NAME]

    mean_iou = MeanIoU(NUM_CLASSES, name='MeanIoU', ignore_class=None)
    mean_iou_ignore = MeanIoU(NUM_CLASSES, name='MeanIoU_ignore', ignore_class=IGNORE_CLASS)
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
                           freeze_backbone=True,
                           weights=PRETRAINED_WEIGHTS
                           )

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)


model.summary()

history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=EPOCHS,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps,
                    callbacks = callbacks,
                    verbose = 1
                    )

# FINE TUNE MODEL
if BACKBONE is not None:    
    # Re-define checkpoint callback to save only the best model
    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,                                           
                                                save_weights_only=False,
                                                monitor='val_MeanIoU_ignore',
                                                mode='max',
                                                save_best_only=True,
                                                verbose=0)
    
    callbacks = [model_checkpoint_callback, tensorboard_callback]
    
    optimizer_dict = {
    'Adam' : Adam(END_LR),
    'Adadelta' : Adadelta(END_LR),
    'AdamW' : AdamW(learning_rate=END_LR, weight_decay=WEIGHT_DECAY),
    'AdaBelief' : AdaBelief(learning_rate=END_LR, weight_decay=WEIGHT_DECAY),
    'SGDW' : SGDW(learning_rate=END_LR, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
    }
    
    with strategy.scope():
        
        loss_func = eval(LOSS)
        loss = loss_func()
        
        optimizer = optimizer_dict[OPTIMIZER_NAME]

        mean_iou = MeanIoU(NUM_CLASSES, name='MeanIoU', ignore_class=None)
        mean_iou_ignore = MeanIoU(NUM_CLASSES, name='MeanIoU_ignore', ignore_class=IGNORE_CLASS)
        metrics = [mean_iou_ignore]
        
        # instantiate model again with the last part of the encoder (Backbone) un-frozen
        model = model_function(input_shape=INPUT_SHAPE,
                               filters=FILTERS,
                               num_classes=NUM_CLASSES,
                               output_stride=OUTPUT_STRIDE,
                               activation=ACTIVATION,
                               dropout_rate=DROPOUT_RATE,
                               backbone_name=BACKBONE,
                               freeze_backbone=False,
                               unfreeze_at=UNFREEZE_AT,
                               )
        
        # load the saved weights into the model to fine tune the high level features of the feature extractor
        # Fine tune the encoder network with a lower learning rate
        model.load_weights(checkpoint_filepath)
        
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    
    model.summary()
    
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        initial_epoch=EPOCHS,
                        epochs=FINAL_EPOCHS,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        callbacks = callbacks,
                        verbose = 1
                        )
    
    # TODO: write callback to save model trunk to avoid the following 
    if DATASET == 'Mapillary':
        best_model = tf.keras.models.load_model(checkpoint_filepath, compile=False)
        trunk = best_model.get_layer('Trunk')
        trunk.save_weights(f'pretrained_Mapillary_models/{MODEL_TYPE}/{MODEL_NAME}/trunk')