import tensorflow as tf
from tensorflow import Tensor
from keras import backend as K
from keras import Model
from keras.initializers import HeNormal
from keras.layers import Add, Multiply
from keras.layers import GlobalAveragePooling2D, Reshape, Resizing
from keras.layers import Conv2D, Conv2DTranspose, Concatenate, Dense, SeparableConv2D
from keras.layers import Dropout, SpatialDropout2D
from keras.layers import MaxPooling2D, UpSampling2D
from keras.layers import BatchNormalization, Activation
from keras.applications.resnet import ResNet50,ResNet101,ResNet152
from keras.applications.resnet_v2 import ResNet50V2,ResNet101V2,ResNet152V2
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v3 import MobileNetV3Small, MobileNetV3Large
# from keras.applications.regnet import R
from keras.applications.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2
from keras.applications.efficientnet import EfficientNetB3, EfficientNetB4, EfficientNetB5
from keras.applications.efficientnet import EfficientNetB6, EfficientNetB7
from efficientnet_v2 import EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2
from efficientnet_v2 import EfficientNetV2B3, EfficientNetV2S, EfficientNetV2M, EfficientNetV2L

from warnings import warn


KERNEL_INITIALIZER = HeNormal(42)


def get_backbone(backbone_name: str, input_tensor: Tensor, freeze_backbone: bool, unfreeze_at: str, output_stride: int = None, depth: int = None) -> Model:    
    backbone_layers = {
        'ResNet50': ('conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out'),
        'ResNet101': ('conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block23_out', 'conv5_block3_out'),
        'ResNet152': ('conv1_relu', 'conv2_block3_out', 'conv3_block8_out', 'conv4_block36_out', 'conv5_block3_out'),
        'ResNet50V2': ('conv1_conv', 'conv2_block3_1_relu', 'conv3_block4_1_relu', 'conv4_block6_1_relu', 'post_relu'),
        'ResNet101V2': ('conv1_conv', 'conv2_block3_1_relu', 'conv3_block4_1_relu', 'conv4_block23_1_relu', 'post_relu'),
        'ResNet152V2': ('conv1_conv', 'conv2_block3_1_relu', 'conv3_block8_1_relu', 'conv4_block36_1_relu', 'post_relu'),
        'MobileNet' : ('conv_pw_1_relu', 'conv_pw_3_relu', 'conv_pw_5_relu', 'conv_pw_11_relu', 'conv_pw_13_relu'),
        'MobileNetV2' : ('block_1_expand_relu', 'block_3_expand_relu', 'block_6_expand_relu', 'block_13_expand_relu', 'out_relu'),
        'MobileNetV3Small' : ('multiply', 're_lu_3', 'multiply_1', 'multiply_11', 'multiply_17'),
        'MobileNetV3Large' : ('re_lu_2', 're_lu_6', 'multiply_1', 'multiply_13', 'multiply_19'),
        'EfficientNetB0': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
        'EfficientNetB1': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
        'EfficientNetB2': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
        'EfficientNetB3': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
        'EfficientNetB4': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
        'EfficientNetB5': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
        'EfficientNetB6': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
        'EfficientNetB7': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
        'EfficientNetV2B0': ('block1a_project_activation', 'block2b_add', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
        'EfficientNetV2B1': ('block1b_add', 'block2c_add', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
        'EfficientNetV2B2': ('block1b_add', 'block2c_add', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
        'EfficientNetV2B3': ('block1b_add', 'block2c_add', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
        'EfficientNetV2S' : ('block1b_add', 'block2d_add', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
        'EfficientNetV2M' : ('block1c_add', 'block2e_add', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
        'EfficientNetV2L' : ('block1d_add', 'block2g_add', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    }
    
    if output_stride is None:
        output_stride = 32
    
    if output_stride != 32 and 'EfficientNetV2' not in backbone_name:
        raise NotImplementedError(f'output_stride other than 32 is not implemented for backbone {backbone_name}. To specify a different value for output_stride use EfficientNetV2 as network backbone.')
    
    backbone_func = eval(backbone_name)
    
    if 'EfficientNetV2' in backbone_name:
        backbone_ = backbone_func(output_stride=output_stride,
                                  include_top=False,
                                  weights='imagenet',
                                  input_tensor=input_tensor,
                                  pooling=None)
    else:
        backbone_ = backbone_func(include_top=False,
                                  weights='imagenet',
                                  input_tensor=input_tensor,
                                  pooling=None)
    
    layer_names = backbone_layers[backbone_name]
    if depth is None:
        depth = len(layer_names)

    X_skip = []
    # get the output of intermediate backbone layers to use them as skip connections
    for i in range(depth):
        X_skip.append(backbone_.get_layer(layer_names[i]).output)
        
    backbone = Model(inputs=input_tensor, outputs=X_skip, name=f'{backbone_name}_backbone')
    
    if freeze_backbone:
        backbone.trainable = False
    elif unfreeze_at is not None:
        layer_dict = {layer.name: i for i,layer in enumerate(backbone.layers)}
        unfreeze_index = layer_dict[unfreeze_at]
        for layer in backbone.layers[:unfreeze_index]:
            layer.trainable = False
    return backbone


def dropout_layer(input_tensor: Tensor, dropout_type: str, dropout_rate: float) -> Tensor:
    if dropout_type==None or dropout_rate==0. :
        return input_tensor
    elif dropout_type=='normal':
        Dropout_layer = Dropout(dropout_rate)
    elif dropout_type=='spatial':
        Dropout_layer = SpatialDropout2D(dropout_rate)
    else:
        raise NotImplementedError(f"{dropout_type}" ' type of dropout is not supported. Available options : "normal", "spatial", "gaussian", "alpha".') 
    return Dropout_layer(input_tensor)


def conv_block(input_tensor: Tensor,
               filters: int,
               dropout_rate: float,
               dropout_type: str,
               activation: str,
               unet_type: str) -> Tensor:
    
    residual = Conv2D(filters, kernel_size=1, padding='same', kernel_initializer=KERNEL_INITIALIZER)(input_tensor)
    
    x = Conv2D(filters, kernel_size=3, padding='same', kernel_initializer=KERNEL_INITIALIZER)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    
    x = Conv2D(filters, kernel_size=3, padding='same', kernel_initializer=KERNEL_INITIALIZER)(x)
    x = BatchNormalization()(x)
    
    if unet_type == 'residual' or unet_type=='attention':
        x = Add()([x, residual])
    
    x = Activation(activation)(x)
    x = dropout_layer(x, dropout_type, dropout_rate)
    
    return x


def downsampling_block(input_tensor: Tensor,
                       filters: int,
                       dropout_rate: float,
                       dropout_type: str,
                       activation: str,
                       unet_type:str
                       ) -> tuple:
    
    x = conv_block(input_tensor, filters, dropout_rate, dropout_type, activation, unet_type)  
    skip_connection = x
    
    downsampled_x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    
    return downsampled_x, skip_connection


def visual_attention_block(encoder_input: Tensor, decoder_input: Tensor) -> Tensor:
    
    input_shape = K.int_shape(encoder_input)
    num_nodes = input_shape[-1] #channels last
    r = 4
    
    # channel attention
    channel_att = GlobalAveragePooling2D()(decoder_input)
    channel_att = Dense(num_nodes/r, activation='relu', input_shape=(num_nodes,))(channel_att)
    channel_att = Dense(num_nodes, activation='sigmoid')(channel_att)
    #channel_att = tf.broadcast_to(channel_att, shape=input_shape)
    
    # spatial attention
    spatial_att = tf.reduce_mean(decoder_input, axis=-1)
    shape = K.int_shape(spatial_att)
    # Reshape because depth dimension is lost when taking then mean along the depth axis
    spatial_att = Reshape((shape[1], shape[2], 1))(spatial_att)
    spatial_att = Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid', kernel_initializer=KERNEL_INITIALIZER)(spatial_att)
    
    channel_att_output = Multiply()([channel_att, encoder_input]) 
    output = Multiply()([spatial_att, channel_att_output])        
    return output


def upsample_and_concat(input_tensor: Tensor,
                        skip_connection: Tensor,
                        filters: int,
                        dropout_rate: float,
                        dropout_type: str,
                        activation: str,
                        unet_type: str
                        )-> Tensor :
    """
    Upsampling block for Unet architecture. Takes as inputs an input tensor which is upsampled with the Transpose Convolution
    operation, and a skip connection tensor which is concatenated with the upsampled tensor. If skip connection is None it
    means the concatenation operation is ommited as there is there is nothing to concatenate the upsampled features with, and
    only upsampling and Convolutions are applied to the input tensor.
    """
    
    up = Conv2DTranspose(filters=filters, kernel_size=2, strides=2, padding="same", kernel_initializer=KERNEL_INITIALIZER)(input_tensor)
    up = BatchNormalization()(up)
    up = Activation(activation)(up)

    if skip_connection is not None:    
        if unet_type == 'attention':
            att = visual_attention_block(skip_connection, up)   
            # Concatenate the feutures
            up = Concatenate(axis=-1)([up, att])
        else:
            up = Concatenate(axis=-1)([up, skip_connection])
    
    x = conv_block(up, filters, dropout_rate, dropout_type, activation, unet_type)  
    return x


def ASPP(input_tensor: Tensor, filters: int, activation: str, dilation_rates, dropout_type, dropout_rate):
    x1 = Conv2D(filters, kernel_size=1, dilation_rate=1, padding='same', kernel_initializer=KERNEL_INITIALIZER)(input_tensor)
    x1 = BatchNormalization()(x1)
    x1 = Activation(activation)(x1)
    x1 = dropout_layer(x1, dropout_type, dropout_rate)
    
    x2 = Conv2D(filters, kernel_size=3, dilation_rate=dilation_rates[0], padding='same', kernel_initializer=KERNEL_INITIALIZER)(input_tensor)
    x2 = BatchNormalization()(x2)
    x2 = Activation(activation)(x2)
    x2 = dropout_layer(x2, dropout_type, dropout_rate)
    
    x3 = Conv2D(filters, kernel_size=3, dilation_rate=dilation_rates[1], padding='same', kernel_initializer=KERNEL_INITIALIZER)(input_tensor)
    x3 = BatchNormalization()(x3)
    x3 = Activation(activation)(x3)
    x3 = dropout_layer(x3, dropout_type, dropout_rate)
    
    x4 = Conv2D(filters, kernel_size=3, dilation_rate=dilation_rates[2], padding='same', kernel_initializer=KERNEL_INITIALIZER)(input_tensor)
    x4 = BatchNormalization()(x4)
    x4 = Activation(activation)(x4)
    x4 = dropout_layer(x4, dropout_type, dropout_rate)
    
    x = Concatenate()([x1,x2,x3,x4])
    return x


def segmentation_head(input_tensor:Tensor, num_classes:int, output_activation='softmax'):
    output = Conv2D(num_classes, kernel_size=1, kernel_initializer=KERNEL_INITIALIZER)(input_tensor)
    return Activation(output_activation, name='output_activation', dtype='float32')(output)


def DeepLabV3plus(input_shape: tuple,
                  filters: tuple, # not needed
                  num_classes: int,
                  output_stride: int = None,
                  activation: str = 'relu', 
                  dropout_rate = 0.0,
                  dropout_type = 'normal',
                  backbone_name = None,
                  freeze_backbone = True,
                  unfreeze_at = None,
                  ):
    
    """
    Instantiate a DeepLabV3+ model.

    Args:
        - `input_shape` (tuple): The shape of the input images.
        - `filters`(list, tuple): A list or tuple containing the number of filters that Convolution layers will have in each stage of the network.
            The length of the filters determines how many stages the network will have.
        - `num_classes` (int): The number of classes to segment the images to. The final Convolution layer has `num_classes` number of filters
            and followed by a softmax activation, the network produces a probability vector for each pixel, that represents the probabilities 
            for the given pixel to belong to each class.
        - `output_stride` (int): The the ratio of input image spatial resolution to the encoder output resolution .
        - `activation` (str, optional): The activation function to be used throughout the network. Defaults to 'relu'.
        - `dropout_rate` (float, optional): The dropout rate used in the dropout layers. Defaults to 0.0.
        - `dropout_type` (str, optional): The type of dropout layers to be used. Options are 'normal' and 'spatial'. Defaults to 'normal'.
        - `backbone_name` (str, optional): The name of the pre-trained backbone. Set to None to use a conventional U-net. Defaults to None.
        - `freeze_backbone` (bool, optional): Whether to freeze the backbone or not. Freezing the backbone means that the the weights of the encoder
            will not get updated during training and only the decoder will be trained. Defaults to True.
        - `unfreeze_at` (str, optional): The name of the layer at which to unfreeze the backbone at. The backbone will be frozen from the first layer
            to the `unfreeze_at` while all the layers after that will be unfrozen and updated during training. For this take effect set 
            `freeze_backbone=False` Defaults to None.
        - `kernel_initializer` (optional): The kernel initializer for the Convolution layers. Defaults to HeNormal().

    Returns:
        tf.keras.Model: A keras Model with the DeepLabV3+ architecture.
    
    References:
        - [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)
    """
    
    assert output_stride in [8,16,32,None]
    
    dilation_rates = {
        32 : [3, 6, 9],
        16 : [6,12,18],
        8 : [12,24,36],
    }
    
    if output_stride is None:
        output_stride = 32
    
    first_upsampling_factor = int(output_stride / 4)
    aspp_dilation_rates = dilation_rates[output_stride]
    
    
    
    input_tensor = tf.keras.Input(shape=input_shape)
    
    backbone = get_backbone(backbone_name=backbone_name,
                            output_stride=output_stride,
                            input_tensor=input_tensor,
                            freeze_backbone=freeze_backbone,
                            unfreeze_at=unfreeze_at)
    Skip = backbone(input_tensor, training=False)

    # Bottleneck
    x = Skip[-1]
    
    low_level_features = Conv2D(48, kernel_size=1, dilation_rate=1, padding='same', kernel_initializer=KERNEL_INITIALIZER)(Skip[1])
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    
    x = ASPP(input_tensor=x, 
             filters=256, 
             activation=activation,
             dilation_rates=aspp_dilation_rates,
             dropout_type=dropout_type, 
             dropout_rate=dropout_rate)
    
    # 1x1 mapping of the spatial_pyramid features
    x = Conv2D(256, kernel_size=1, padding='same', kernel_initializer=KERNEL_INITIALIZER)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = dropout_layer(x, dropout_type, dropout_rate)
    
    # Decoder module
    x = UpSampling2D(size=first_upsampling_factor, interpolation='bilinear')(x)
    x = Concatenate(axis=-1)([x, low_level_features])
    
    x = Conv2D(256, kernel_size=3, padding='same', kernel_initializer=KERNEL_INITIALIZER)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(256, kernel_size=3, padding='same', kernel_initializer=KERNEL_INITIALIZER)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = dropout_layer(x, dropout_type, dropout_rate)
    
    x = UpSampling2D(size=4, interpolation='bilinear')(x)

    output = segmentation_head(x, num_classes=num_classes)
    
    model = Model(inputs=input_tensor, outputs=output, name=f'DeepLabV3plus')
    
    input_spatial_resolution = model.input_shape[1:3]
    output_spatial_resolution = model.output_shape[1:3]
    assert input_spatial_resolution == output_spatial_resolution, f'Model output spatial resolution must be equal to input spatial resolution.'
    
    return model


def base_Unet(unet_type: str,
              input_shape: tuple,
              filters: tuple, 
              num_classes: int,
              activation: str,
              dropout_rate: float,
              dropout_type: str,              
              scale_dropout: bool,
              dropout_offset: float,
              backbone_name : str,
              freeze_backbone : bool,
              unfreeze_at : str,
              output_stride : int = None
              ):

    if output_stride is None:
        output_stride = 32
        
    if output_stride != 32:
        raise ValueError('Unet models are not implemented with backbones and output stride other than 32')
    
    depth = len(filters)

    if isinstance(dropout_rate, (list, tuple)):
        assert len(dropout_rate)== depth, 'dropout_rate length does not match model depth. The length of dropout_rate needs to match the length of filters if passed as a list or tuple'
    else:
        offset = dropout_offset if scale_dropout else 0
        dropout_rate = [dropout_rate + offset*i for i in range(depth)]
        
    if isinstance(dropout_type, (list, tuple)):
        assert len(dropout_type)==depth, 'dropout_type length does not match model depth. The length of dropout_type needs to match the length of filters if passed as a list or tuple'
    else:
        dropout_type = [dropout_type] * depth

    # ---------------------------------------------------------------------------------------------------------------------------
    input_tensor = tf.keras.Input(shape=input_shape)
    
    if backbone_name is None:
        # Εncoder
        x, skip = downsampling_block(input_tensor, filters[0], dropout_rate[0], dropout_type[0], activation, unet_type)
        Skip = [skip]
        for i in range(1, depth-1):
            x, skip = downsampling_block(x, filters[i], dropout_rate[i], dropout_type[i], activation, unet_type)
            Skip.append(skip)
        
        # Bottleneck
        x = conv_block(x, filters[-1], dropout_rate[-1], dropout_type[-1], activation, unet_type)
        
        # Decoder
        for i in range(depth-2, -1, -1):
            x = upsample_and_concat(x, Skip[i], filters[i], dropout_rate[i], dropout_type[i], activation, unet_type)

    else:
        # when using bakcbone because the stem performs downsampling and we have another 4 downsampling layers
        # (5 in total) we need to perform upsampling 5 times and not 4 (as without backbone) -> 6 total Unet levels
        # but no skip connection from the top level (full spatial dimension)
        
        # Pre-trained Backbone as Encoder
        backbone = get_backbone(backbone_name=backbone_name,
                            output_stride=output_stride,
                            input_tensor=input_tensor,
                            freeze_backbone=freeze_backbone,
                            unfreeze_at=unfreeze_at,
                            depth=depth)
        Skip = backbone(input_tensor, training=False)
        
        # Bottleneck
        x = Skip[-1]
        Skip.insert(0, None) # means that there are no features to concatenate with
    
        # iterate 4 times
        for i in range(depth-1, -1, -1):
            x = upsample_and_concat(x, Skip[i], filters[i], dropout_rate[i], dropout_type[i], activation, unet_type)
        
    output = segmentation_head(x, num_classes=num_classes)
    
    model = Model(inputs=input_tensor, outputs=output, name=f'{unet_type}_U-net' if unet_type != 'normal' else 'U-net')
    
    input_spatial_resolution = model.input_shape[1:3]
    output_spatial_resolution = model.output_shape[1:3]
    assert input_spatial_resolution == output_spatial_resolution, f'Model output spatial resolution must be equal to input spatial resolution.'
    
    return model

              
def Unet(input_shape: tuple,
         filters: tuple, 
         num_classes: int,
         activation: str = 'relu',
         dropout_rate: float = 0.0,
         dropout_type = 'normal',
         scale_dropout = False,
         dropout_offset = 0.01,
         backbone_name = None,
         freeze_backbone = True,
         unfreeze_at = None,
         output_stride = None
         ):
    
    """
    Instantiate a U-net model.

    Args:
        - `input_shape` (tuple): The shape of the input images.
        - `filters`(list, tuple): A list or tuple containing the number of filters that Convolution layers will have in each stage of the network.
            The length of the filters determines how many stages the network will have.
        - `num_classes` (int): The number of classes to segment the images to. The final Convolution layer has `num_classes` number of filters
            and followed by a softmax activation, the network produces a probability vector for each pixel, that represents the probabilities 
            for the given pixel to belong to each class.
        - `activation` (str, optional): The activation function to be used throughout the network. Defaults to 'relu'.
        - `dropout_rate` (float, optional): The dropout rate used in the dropout layers. Defaults to 0.0.
        - `dropout_type` (str, optional): The type of dropout layers to be used. Options are 'normal' and 'spatial'. Defaults to 'normal'.
        - `scale_dropout` (bool, optional): Whether to increase the dropout rate the deeper the network goes.
        - `dropout_offset` (float, optional): The amount by which to increase the dropout rate for every stage of the network. Defaults to 0.01.
        - `backbone_name` (str, optional): The name of the pre-trained backbone. Set to None to use a conventional U-net. Defaults to None.
        - `freeze_backbone` (bool, optional): Whether to freeze the backbone or not. Freezing the backbone means that the the weights of the encoder
            will not get updated during training and only the decoder will be trained. Defaults to True.
        - `unfreeze_at` (str, optional): The name of the layer at which to unfreeze the backbone at. The backbone will be frozen from the first layer
            to the `unfreeze_at` while all the layers after that will be unfrozen and updated during training. For this take effect set 
            `freeze_backbone=False` Defaults to None.
        - `kernel_initializer` (optional): The kernel initializer for the Convolution layers. Defaults to HeNormal().

    Returns:
        tf.keras.Model: A keras Model with the U-net architecture.
        
    References:
        - [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
    """
    
    return  base_Unet('normal', 
                      input_shape=input_shape, 
                      filters=filters, 
                      num_classes=num_classes, 
                      activation=activation,
                      dropout_rate=dropout_rate, 
                      dropout_type=dropout_type, 
                      scale_dropout=scale_dropout, 
                      dropout_offset=dropout_offset,
                      backbone_name=backbone_name,
                      freeze_backbone=freeze_backbone,
                      unfreeze_at=unfreeze_at,
                      output_stride=output_stride)


def Residual_Unet(input_shape: tuple,
                  filters: tuple,
                  num_classes: int,
                  activation: str = 'relu',
                  dropout_rate: float = 0.0,
                  dropout_type: str = 'normal',
                  scale_dropout: bool = False,
                  dropout_offset: float = 0.01,
                  backbone_name = None,
                  freeze_backbone = True,
                  unfreeze_at = None,
                  output_stride = None
                  ):
    
    """
    Instantiate a U-net model with a modified basic basic block with residual connections to improve network learning capacity.

    Args:
        - `input_shape` (tuple): The shape of the input images.
        - `filters`(list, tuple): A list or tuple containing the number of filters that Convolution layers will have in each stage of the network.
            The length of the filters determines how many stages the network will have.
        - `num_classes` (int): The number of classes to segment the images to. The final Convolution layer has `num_classes` number of filters
            and followed by a softmax activation, the network produces a probability vector for each pixel, that represents the probabilities 
            for the given pixel to belong to each class.
        - `activation` (str, optional): The activation function to be used throughout the network. Defaults to 'relu'.
        - `dropout_rate` (float, optional): The dropout rate used in the dropout layers. Defaults to 0.0.
        - `dropout_type` (str, optional): The type of dropout layers to be used. Options are 'normal' and 'spatial'. Defaults to 'normal'.
        - `scale_dropout` (bool, optional): Whether to increase the dropout rate the deeper the network goes. Defaults to False.
        - `dropout_offset` (float, optional): The amount by which to increase the dropout rate for every stage of the network. Defaults to 0.01.
        - `backbone_name` (str, optional): The name of the pre-trained backbone. Set to None to instantiate a conventional U-net. Defaults to None.
        - `freeze_backbone` (bool, optional): Whether to freeze the backbone or not. Freezing the backbone means that the the weights of the encoder
            will not get updated during training and only the decoder will be trained. Defaults to True.
        - `unfreeze_at` (str, optional): The name of the layer at which to unfreeze the backbone at. The backbone will be frozen from the first layer
            to the `unfreeze_at` while all the layers after that will be unfrozen and updated during training. For this take effect set 
            `freeze_backbone=False` Defaults to None.
        - `kernel_initializer` (optional): The kernel initializer for the Convolution layers. Defaults to HeNormal().

    Returns:
        tf.keras.Model: A keras Model with the U-net architecture.
        
    References:
        - [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
        - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
    """

    return  base_Unet('residual', 
                      input_shape=input_shape, 
                      filters=filters, 
                      num_classes=num_classes, 
                      activation=activation,
                      dropout_rate=dropout_rate, 
                      dropout_type=dropout_type, 
                      scale_dropout=scale_dropout, 
                      dropout_offset=dropout_offset,
                      backbone_name=backbone_name,
                      freeze_backbone=freeze_backbone,
                      unfreeze_at=unfreeze_at,
                      output_stride=output_stride,
                      )


def Attention_Unet(input_shape: tuple,
                   filters: tuple,
                   num_classes: int,
                   activation: str = 'relu',
                   dropout_rate: float = 0.0,
                   dropout_type: str = 'normal',
                   scale_dropout: bool = False,
                   dropout_offset: float = 0.01,
                   backbone_name = None,
                   freeze_backbone = True,
                   unfreeze_at = None,
                   output_stride = None
                   ):
    """
    Instantiate a U-net model that uses attention modules in each decoder block to improve segmentation results by weighing 
    with a modified basic basic block with residual connections to improve network learning capacity.

    Args:
        - `input_shape` (tuple): The shape of the input images.
        - `filters`(list, tuple): A list or tuple containing the number of filters that Convolution layers will have in each stage of the network.
            The length of the filters determines how many stages the network will have.
        - `num_classes` (int): The number of classes to segment the images to. The final Convolution layer has `num_classes` number of filters
            and followed by a softmax activation, the network produces a probability vector for each pixel, that represents the probabilities 
            for the given pixel to belong to each class.
        - `activation` (str, optional): The activation function to be used throughout the network. Defaults to 'relu'.
        - `dropout_rate` (float, optional): The dropout rate used in the dropout layers. Defaults to 0.0.
        - `dropout_type` (str, optional): The type of dropout layers to be used. Options are 'normal' and 'spatial'. Defaults to 'normal'.
        - `scale_dropout` (bool, optional): Whether to increase the dropout rate the deeper the network goes. Defaults to False.
        - `dropout_offset` (float, optional): The amount by which to increase the dropout rate for every stage of the network. Defaults to 0.01.
        - `backbone_name` (str, optional): The name of the pre-trained backbone. Set to None to instantiate a conventional U-net. Defaults to None.
        - `freeze_backbone` (bool, optional): Whether to freeze the backbone or not. Freezing the backbone means that the the weights of the encoder
            will not get updated during training and only the decoder will be trained. Defaults to True.
        - `unfreeze_at` (str, optional): The name of the layer at which to unfreeze the backbone at. The backbone will be frozen from the first layer
            to the `unfreeze_at` while all the layers after that will be unfrozen and updated during training. For this take effect set 
            `freeze_backbone=False` Defaults to None.
        - `kernel_initializer` (optional): The kernel initializer for the Convolution layers. Defaults to HeNormal().

    Returns:
        tf.keras.Model: A keras Model with the U-net architecture.
        
        
    References:
        - [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
        - [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999)
        - [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)
    """
    
    return  base_Unet('attention', 
                      input_shape=input_shape, 
                      filters=input_shape, 
                      num_classes=num_classes, 
                      activation=activation,
                      dropout_rate=activation,
                      dropout_type=dropout_type, 
                      scale_dropout=scale_dropout,
                      dropout_offset=scale_dropout,
                      backbone_name=backbone_name,
                      freeze_backbone=freeze_backbone,
                      unfreeze_at=unfreeze_at,
                      output_stride = output_stride,
                      )


def Unet_plus(input_shape: tuple,
              filters: tuple, 
              num_classes: int,
              activation: str = 'relu',
              dropout_rate: float = 0.0, 
              dropout_type: str = 'normal',
              backbone_name = None,
              freeze_backbone = True,
              unfreeze_at = None,
              deep_supervision: bool = False,
              attention: bool = False,
              output_stride = None,
              ):
    
    """
    Instantiate a U-net++ model.

    Args:
        - `input_shape` (tuple): The shape of the input images.
        - `filters`(list, tuple): A list or tuple containing the number of filters that Convolution layers will have in each stage of the network.
            The length of the filters determines how many stages the network will have.
        - `num_classes` (int): The number of classes to segment the images to. The final Convolution layer has `num_classes` number of filters
            and followed by a softmax activation, the network produces a probability vector for each pixel, that represents the probabilities 
            for the given pixel to belong to each class.
        - `activation` (str, optional): The activation function to be used throughout the network. Defaults to 'relu'.
        - `dropout_rate` (float, optional): The dropout rate used in the dropout layers. Defaults to 0.0.
        - `dropout_type` (str, optional): The type of dropout layers to be used. Options are 'normal' and 'spatial'. Defaults to 'normal'.
        - `scale_dropout` (bool, optional): Whether to scale the dropout rate the deeper the network goes. The dropout rate is increased by 
            `dropout_offset` for every stage of the network. Defaults to False.
        - `dropout_offset` (float, optional): The amount by which to increase the dropout rate for every stage of the network. Defaults to 0.01.
        - `deep_supervision` (bool, optional): Whether to use deep supervision. Deep supervision may produce better results and it enables model 
            pruning. Defaults to False.
        - `attention` (bool, optional): Whether to use attention modules. Defaults to False.
        - `kernel_initializer` (, optional): The kernel initializer for the Convolution layers. Defaults to HeNormal().
        
    Returns:
        tf.keras.Model: A keras Model with the U-net++ architecture.
           
    References:
        - [UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/abs/1807.10165)
    """
    
    depth = len(filters)
    assert depth==6, f'Expected a list of filters with length : 5. Instead got length : {depth}.'
    
    if attention:
        unet_type = 'attention'
    else:
        unet_type = 'normal'
        
    
    input_tensor = tf.keras.Input(shape=input_shape)

    if backbone_name is None:
        x0_0, skip0_0 = downsampling_block(input_tensor, filters[0], dropout_rate, dropout_type, activation, unet_type)
        x1_0, skip1_0 = downsampling_block(x0_0, filters[1], dropout_rate, dropout_type, activation, unet_type)
        x2_0, skip2_0 = downsampling_block(x1_0, filters[2], dropout_rate, dropout_type, activation, unet_type)
        x3_0, skip3_0 = downsampling_block(x2_0, filters[3], dropout_rate, dropout_type, activation, unet_type)
        x4_0, skip4_0 = downsampling_block(x3_0, filters[4], dropout_rate, dropout_type, activation, unet_type)
        x5_0 = conv_block(x4_0, filters[5], dropout_rate, dropout_type, activation, unet_type)
        skip5_0 = x5_0

    else:
        # Pre-trained Backbone as Encoder
        backbone = get_backbone(backbone_name, 
                                input_tensor=input_tensor, 
                                freeze_backbone=freeze_backbone, 
                                depth=depth-1,
                                unfreeze_at=unfreeze_at,
                                #output_stride=output_stride,
                                )
        Skip = backbone(input_tensor, training=False)
        
        Skip.insert(0, None)
        skip0_0, skip1_0, skip2_0, skip3_0, skip4_0, skip5_0 = Skip
        # skip0_0 is None
        
    x0_1 = upsample_and_concat(skip1_0, skip0_0, filters[0], dropout_rate, dropout_type, activation, unet_type)
    x1_1 = upsample_and_concat(skip2_0, skip1_0, filters[1], dropout_rate, dropout_type, activation, unet_type)
    x2_1 = upsample_and_concat(skip3_0, skip2_0, filters[2], dropout_rate, dropout_type, activation, unet_type)
    x3_1 = upsample_and_concat(skip4_0, skip3_0, filters[3], dropout_rate, dropout_type, activation, unet_type)
    x4_1 = upsample_and_concat(skip5_0, skip4_0, filters[4], dropout_rate, dropout_type, activation, unet_type)
    
    skip0_2 = x0_1 if skip0_0 is None else Concatenate(axis=-1)([x0_1, skip0_0])
    x0_2 = upsample_and_concat(x1_1, skip0_2, filters[0], dropout_rate, dropout_type, activation, unet_type)    
    skip1_2 = Concatenate(axis=-1)([x1_1, skip1_0])
    x1_2 = upsample_and_concat(x2_1, skip1_2, filters[1], dropout_rate, dropout_type, activation, unet_type)
    skip2_2 = Concatenate(axis=-1)([x2_1, skip2_0])
    x2_2 = upsample_and_concat(x3_1, skip2_2, filters[2], dropout_rate, dropout_type, activation, unet_type)
    skip3_2 = Concatenate(axis=-1)([x3_1, skip3_0])
    x3_2 = upsample_and_concat(x4_1, skip3_2, filters[3], dropout_rate, dropout_type, activation, unet_type)
    
    skip0_3 = Concatenate(axis=-1)([x0_2, x0_1]) if skip0_0 is None else Concatenate(axis=-1)([x0_2, x0_1, skip0_0])
    x0_3 = upsample_and_concat(x1_2, skip0_3, filters[0], dropout_rate, dropout_type, activation, unet_type)
    skip1_3 = Concatenate(axis=-1)([x1_2, x1_1, skip1_0])
    x1_3 = upsample_and_concat(x2_2, skip1_3, filters[1], dropout_rate, dropout_type, activation, unet_type)
    skip2_3 = Concatenate(axis=-1)([x2_2, x2_1, skip2_0])
    x2_3 = upsample_and_concat(x3_2, skip2_3, filters[2], dropout_rate, dropout_type, activation, unet_type)
    
    skip0_4 = Concatenate(axis=-1)([x0_3, x0_2, x0_1]) if skip0_0 is None else Concatenate(axis=-1)([x0_3, x0_2, x0_1, skip0_0])
    x0_4 = upsample_and_concat(x1_3, skip0_4, filters[0], dropout_rate, dropout_type, activation, unet_type)
    skip1_4 = Concatenate(axis=-1)([x1_3, x1_2, x1_1, skip1_0])
    x1_4 = upsample_and_concat(x2_3, skip1_4, filters[1], dropout_rate, dropout_type, activation, unet_type)
    
    skip0_5 = Concatenate(axis=-1)([x0_4, x0_3, x0_2, x0_1]) if skip0_0 is None else Concatenate(axis=-1)([x0_4, x0_3, x0_2, x0_1, skip0_0])
    x0_5 = upsample_and_concat(x1_4, skip0_5, filters[0], dropout_rate, dropout_type, activation, unet_type)
    
    output_1 = Conv2D(num_classes, kernel_size=1, kernel_initializer=KERNEL_INITIALIZER)(x0_1)
    output_1 = Activation('softmax', name='output_1', dtype='float32')(output_1)
    
    output_2 = Conv2D(num_classes, kernel_size=1, kernel_initializer=KERNEL_INITIALIZER)(x0_2)
    output_2 = Activation('softmax', name='output_2', dtype='float32')(output_2)
    
    output_3 = Conv2D(num_classes, kernel_size=1, kernel_initializer=KERNEL_INITIALIZER)(x0_3)
    output_3 = Activation('softmax', name='output_3', dtype='float32')(output_3)
    
    output_4 = Conv2D(num_classes, kernel_size=1, kernel_initializer=KERNEL_INITIALIZER)(x0_4)
    output_4 = Activation('softmax', name='output_4', dtype='float32')(output_4)
    
    output_5 = Conv2D(num_classes, kernel_size=1, kernel_initializer=KERNEL_INITIALIZER)(x0_5)
    output_5 = Activation('softmax', name='output_5', dtype='float32')(output_5)
    
    Unet_pp_L5 = Model(inputs=input_tensor, outputs=output_5, name='Unet_pp_L5')
    Unet_pp = Model(inputs=input_tensor, outputs=[output_1,
                                                  output_2,
                                                  output_3,
                                                  output_4,
                                                  output_5] , name='Unet_pp')
    
    if deep_supervision:
        model = Unet_pp
    else:
        model = Unet_pp_L5
    
    return model


class Unet_pp():
    def __init__(self, deep_supervision: bool = False) -> None:
        self.deep_supervision = deep_supervision
    
    
    def build(self,
              input_shape: tuple,
              filters: tuple, 
              num_classes: int,
              activation: str = 'relu',
              dropout_rate: float = 0.0, 
              dropout_type: str = 'normal',
              backbone_name = None,
              freeze_backbone = True,
              unfreeze_at = None,
              kernel_initializer = HeNormal(42),
              attention: bool = False
              ):
        
        self.depth = len(filters)
        
        self.model = Unet_plus(input_shape = input_shape,
                                filters = filters, 
                                num_classes = num_classes,
                                activation = activation,
                                dropout_rate = dropout_rate, 
                                dropout_type = dropout_type,
                                backbone_name = backbone_name,
                                freeze_backbone = freeze_backbone,
                                unfreeze_at = unfreeze_at,
                                kernel_initializer = kernel_initializer,
                                deep_supervision = self.deep_supervision,
                                attention = attention)
    
    
    def load(self, path:str, load_weights_only=False):
        if load_weights_only:
            self.model = self.model.load_weights(path)
        self.model = tf.keras.models.load_model(path, compile=False)
    
    
    def compile(self, loss, optimizer, metrics):
        if self.deep_supervision:
            self.model.compile(loss={'output_1': loss,
                                     'output_2': loss,
                                     'output_3': loss,
                                     'output_4': loss,
                                     'output_5': loss},
                               optimizer=optimizer,
                               metrics=metrics)
        else:
            self.model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
    
    
    def prune(self, stage):
        if self.deep_supervision:
            output_layer_name = f'output_{stage}'
            output_layer = self.model.get_layer(output_layer_name).output
            input_layer = self.model.input
            self.model = Model(inputs=input_layer, outputs=output_layer)
        else:
            print("This model was instantiated without Deep supervision enabled so it cannot be pruned! The prunning opperation will be skipped")


    def get_model(self):
        return self.model
