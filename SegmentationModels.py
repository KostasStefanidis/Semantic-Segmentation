import tensorflow as tf
from tensorflow import Tensor
from keras import backend as K
from keras import Model
from keras.initializers import HeNormal
from keras.layers import Add, Multiply
from keras.layers import GlobalAveragePooling2D, Reshape
from keras.layers import Conv2D, Conv2DTranspose, Concatenate, Dense
from keras.layers import Dropout, SpatialDropout2D
from keras.layers import MaxPooling2D, UpSampling2D
from keras.layers import BatchNormalization, Activation
from keras.applications.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2
from keras.applications.efficientnet import EfficientNetB3, EfficientNetB4, EfficientNetB5
from keras.applications.efficientnet import EfficientNetB6, EfficientNetB7
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.resnet import ResNet50,ResNet101,ResNet152


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
               kernel_initializer: str,
               unet_type: str) -> Tensor:
    
    # no dilation in residual connection
    residual = Conv2D(filters, kernel_size=1, padding='same', kernel_initializer=kernel_initializer)(input_tensor)
    
    x = Conv2D(filters, kernel_size=3, padding='same', kernel_initializer=kernel_initializer)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    
    x = Conv2D(filters, kernel_size=3, padding='same', kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization()(x)
    
    if unet_type == 'residual' or unet_type=='attention':
        # test residual block with 3 convolutions
        # x = Activation(activation)(x)
        # x = Conv2D(filters, kernel_size=(3, 3), padding='same', kernel_initializer=kernel_initializer)(x)
        # x = BatchNormalization()(x)
        x = Add()([x, residual])
    
    x = Activation(activation)(x)
    x = dropout_layer(x, dropout_type, dropout_rate)
    
    return x


def downsampling_block(input_tensor: Tensor,
                       filters: int,
                       dropout_rate: float,
                       dropout_type: str,
                       activation: str,
                       kernel_initializer: str,
                       unet_type:str
                       ) -> tuple:
    
    x = conv_block(input_tensor, filters, dropout_rate, dropout_type, activation, kernel_initializer, unet_type)  
    skip_connection = x
    
    downsampled_x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    
    return downsampled_x, skip_connection


def visual_attention_block(encoder_input: Tensor, decoder_input: Tensor, kernel_initializer: str) -> Tensor:
    
    input_shape = K.int_shape(encoder_input)
    num_nodes = input_shape[-1] #channels last
    r = 4
    
    # channel attention
    channel_att = GlobalAveragePooling2D()(decoder_input)
    channel_att = Dense(num_nodes/r, activation='relu', input_shape=(num_nodes,))(channel_att)
    channel_att = Dense(num_nodes, activation='sigmoid')(channel_att)
    channel_att = tf.broadcast_to(channel_att, shape=input_shape)
    
    spatial_att = tf.reduce_mean(decoder_input, axis=-1)
    shape = K.int_shape(spatial_att)
    # Reshape because depth dimension is lost when taking then mean along the depth axis
    spatial_att = Reshape((shape[1], shape[2], 1))(spatial_att)
    spatial_att = Conv2D(1, kernel_size=3, padding='same', activation='sigmoid', kernel_initializer=kernel_initializer)(spatial_att)
    spatial_att = tf.broadcast_to(spatial_att, shape=input_shape)
    
    channel_att_output = Multiply()([channel_att, encoder_input]) 
    output = Multiply()([spatial_att, channel_att_output])        
    return output


def upsampling_block(input_tensor: Tensor,
                     skip_connection: Tensor,
                     filters: int,
                     dropout_rate: float,
                     dropout_type: str,
                     activation: str,
                     kernel_initializer: str,
                     unet_type: str
                     )-> Tensor :
    """
    Upsampling block for Unet architecture. Takes as inputs an input tensor which is upsampled with the Transpose Convolution
    operation, and a skip connection tensor which is concatenated with the upsampled tensor. If skip connection is None it
    means the concatenation operation is ommited as there is there is nothing to concatenate the upsampled features with, and
    only upsampling and Convolutions are applied to the input tensor.
    """
    
    up = Conv2DTranspose(filters=filters, kernel_size=2, strides=2, padding="same", kernel_initializer=kernel_initializer)(input_tensor)
    up = BatchNormalization()(up)
    up = Activation(activation)(up)

    
    if skip_connection is not None:    
        if unet_type == 'attention':
            att = visual_attention_block(skip_connection, up, kernel_initializer)   
            # Concatenate the feutures
            up = Concatenate(axis=-1)([up, att])
        else:
            up = Concatenate(axis=-1)([up, skip_connection])
    
    x = conv_block(up, filters, dropout_rate, dropout_type, activation, kernel_initializer, unet_type)  
    return x


def get_backbone(backbone_name: str, input_tensor: Tensor, freeze_backbone: bool, depth: int) -> Model:    
    backbone_layers = {
    'ResNet50': ('conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out'),
    'ResNet101': ('conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block23_out', 'conv5_block3_out'),
    'ResNet152': ('conv1_relu', 'conv2_block3_out', 'conv3_block8_out', 'conv4_block36_out', 'conv5_block3_out'),
    'ResNet50V2': ('conv1_conv', 'conv2_block3_1_relu', 'conv3_block4_1_relu', 'conv4_block6_1_relu', 'post_relu'),
    'ResNet101V2': ('conv1_conv', 'conv2_block3_1_relu', 'conv3_block4_1_relu', 'conv4_block23_1_relu', 'post_relu'),
    'ResNet152V2': ('conv1_conv', 'conv2_block3_1_relu', 'conv3_block8_1_relu', 'conv4_block36_1_relu', 'post_relu'),
    'DenseNet121': ('conv1/relu', 'pool2_conv', 'pool3_conv', 'pool4_conv', 'relu'),
    'DenseNet169': ('conv1/relu', 'pool2_conv', 'pool3_conv', 'pool4_conv', 'relu'),
    'DenseNet201': ('conv1/relu', 'pool2_conv', 'pool3_conv', 'pool4_conv', 'relu'),
    'EfficientNetB0': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB1': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB2': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB3': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB4': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB5': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB6': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB7': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetV2B0': ('block1a_project_activation', 'block2b_add', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetV2B1': ('block1b_add', 'block2c_add', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation')
    }
    
    layer_names = backbone_layers[backbone_name]
    
    backbone_func = eval(backbone_name)
    backbone_ = backbone_func(include_top=False,
                              weights='imagenet',
                              input_tensor=input_tensor,
                              pooling=None)

    X_skip = []
    # get the output of intermediate backbone layers to use them as skip connections
    for i in range(depth):
        X_skip.append(backbone_.get_layer(layer_names[i]).output)
    
    backbone = Model(inputs=input_tensor, outputs=X_skip, name=f'{backbone_name}_backbone')
    if freeze_backbone:
        backbone.trainable = False
    
    return backbone


def base_Unet(unet_type: str,
              input_shape: tuple,
              filters: tuple, 
              num_classes: int,
              activation: str,
              dropout_rate: float, 
              dropout_type: str,              
              scale_dropout: bool,
              dropout_offset: float,
              kernel_initializer,
              backbone_name = None,
              freeze_backbone = True
              ):

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
    x0 = tf.keras.Input(shape=input_shape)
    
    if backbone_name is None:
        # Εncoder
        x, skip = downsampling_block(x0, filters[0], dropout_rate[0], dropout_type[0], activation, kernel_initializer, unet_type)
        Skip = [skip]
        for i in range(1, depth-1):
            x, skip = downsampling_block(x, filters[i], dropout_rate[i], dropout_type[i], activation, kernel_initializer, unet_type)
            Skip.append(skip)
        
        # Bottleneck
        x = conv_block(x, filters[-1], dropout_rate[-1], dropout_type[-1], activation, kernel_initializer, unet_type)
        
        # Decoder
        for i in range(depth-2, -1, -1):
            x = upsampling_block(x, Skip[i], filters[i], dropout_rate[i], dropout_type[i], activation, kernel_initializer, unet_type)

    else:
        # when using bakcbone because the stem performs downsampling and we have another 4 downsampling layers
        # (5 in total) we need to perform upsampling 5 times and not 4 (as without backbone)
        
        # Pre-trained Backbone as Encoder
        backbone = get_backbone(backbone_name, input_tensor=x0, freeze_backbone=freeze_backbone, depth=depth)
        Skip = backbone(x0, training=False)
        
        # Bottleneck
        x = Skip[-1]
        Skip.insert(0, None) # means that there are no features to concatenate with
    
        # iterate 4 times
        for i in range(depth-1, -1, -1):
            x = upsampling_block(x, Skip[i], filters[i], dropout_rate[i], dropout_type[i], activation, kernel_initializer, unet_type)
        
    output = Conv2D(num_classes, kernel_size=(1,1), kernel_initializer=kernel_initializer)(x)
    output = Activation('softmax', name='output', dtype='float32')(output)
    model = Model(inputs=x0, outputs=output, name=f'{unet_type}_U-net' if unet_type != 'normal' else 'U-net')
    return model


def DeepLabeV3plus():
    return


def Unet(input_shape: tuple,
         filters: tuple, 
         num_classes: int,
         dropout_rate: float,
         activation: str,
         dropout_type = 'normal',
         scale_dropout = False,
         dropout_offset = 0.01,
         kernel_initializer = HeNormal(42),
         backbone_name = None,
         freeze_backbone= True
         ):

    return  base_Unet('normal', 
                      input_shape, 
                      filters, 
                      num_classes, 
                      activation,
                      dropout_rate, 
                      dropout_type, 
                      scale_dropout, 
                      dropout_offset,
                      kernel_initializer,
                      backbone_name=backbone_name,
                      freeze_backbone=freeze_backbone)


def Residual_Unet(input_shape: tuple,
                  filters: tuple,
                  num_classes: int,
                  dropout_rate: float,
                  activation: str,               
                  dropout_type: str = 'normal',
                  scale_dropout: bool = False,
                  dropout_offset: float = 0.01,
                  kernel_initializer = HeNormal(42),
                  backbone_name = None,
                  freeze_backbone= True                      
                  ):
    
    return  base_Unet('residual', 
                      input_shape, 
                      filters, 
                      num_classes, 
                      activation,
                      dropout_rate, 
                      dropout_type, 
                      scale_dropout, 
                      dropout_offset,
                      kernel_initializer,
                      backbone_name=backbone_name,
                      freeze_backbone=freeze_backbone)


def Attention_Unet(input_shape: tuple,
                   filters: tuple, 
                   num_classes: int, 
                   dropout_rate: float,
                   activation: str,
                   dropout_type: str = 'normal',
                   scale_dropout: bool = False,
                   dropout_offset: float = 0.01,
                   kernel_initializer = HeNormal(42),
                   backbone_name = None,
                   freeze_backbone= True 
                   ):
    
    return  base_Unet('attention', 
                      input_shape, 
                      filters, 
                      num_classes, 
                      activation,
                      dropout_rate,
                      dropout_type, 
                      scale_dropout, 
                      dropout_offset,
                      kernel_initializer,
                      backbone_name=backbone_name,
                      freeze_backbone=freeze_backbone)


def Unet_plus(input_shape: tuple,
              filters: tuple, 
              num_classes: int, 
              dropout_rate: float, 
              dropout_type: str,
              activation: str,
              scale_dropout: bool = True,
              dropout_offset: float = 0.01,
              kernel_initializer = HeNormal(42),
              deep_supervision: bool = False,
              attention: bool = False
              ):
    
    depth = len(filters)
    
    assert depth==5, f'Expected a list of filters with length : 5. Instead got length : {depth}.'
    
    if attention:
        unet_type = 'attention'
    else:
        unet_type = 'normal'
       
    offset = dropout_offset if scale_dropout else 0
    
    if isinstance(dropout_rate, (list, tuple)):
        assert len(dropout_rate)== depth, 'dropout_rate length does not match model depth. The length of dropout_rate needs to match the length of filters if passed as a list or tuple'
    else:
        offset = dropout_offset if scale_dropout else 0
        dropout_rate = [dropout_rate + offset*i for i in range(depth)]
        
    if isinstance(dropout_type, (list, tuple)):
        assert len(dropout_type)==depth, 'dropout_type length does not match model depth. The length of dropout_type needs to match the length of filters if passed as a list or tuple'
    else:
        dropout_type = [dropout_type] * depth
    
    
    
    x0_0 = tf.keras.Input(shape=input_shape)
    
    x1_0, skip1_0 = downsampling_block(x0_0, filters[0], dropout_rate[0], dropout_type[0], activation, kernel_initializer, unet_type)
    x2_0, skip2_0 = downsampling_block(x1_0, filters[1], dropout_rate[1], dropout_type[1], activation, kernel_initializer, unet_type)
    x3_0, skip3_0 = downsampling_block(x2_0, filters[2], dropout_rate[2], dropout_type[2], activation, kernel_initializer, unet_type)
    x4_0, skip4_0 = downsampling_block(x3_0, filters[3], dropout_rate[3], dropout_type[3], activation, kernel_initializer, unet_type)
    
    x4_0 = conv_block(x4_0, filters[4], dropout_rate[4], dropout_type[4], activation, kernel_initializer, unet_type)
    
    x0_1 = upsampling_block(x1_0, skip1_0, filters[0], dropout_rate[0], dropout_type[0], activation, kernel_initializer, unet_type)
    x1_1 = upsampling_block(x2_0, skip2_0, filters[1], dropout_rate[1], dropout_type[1], activation, kernel_initializer, unet_type)
    x2_1 = upsampling_block(x3_0, skip3_0, filters[2], dropout_rate[2], dropout_type[2], activation, kernel_initializer, unet_type)
    x3_1 = upsampling_block(x4_0, skip4_0, filters[3], dropout_rate[3], dropout_type[3], activation, kernel_initializer, unet_type)
    
    skip0_2 = Concatenate(axis=-1)([x0_1, skip1_0])
    x0_2 = upsampling_block(x1_1, skip0_2, filters[0], dropout_rate[0], dropout_type[0], activation, kernel_initializer, unet_type)    
    skip1_2 = Concatenate(axis=-1)([x1_1, skip2_0])
    x1_2 = upsampling_block(x2_1, skip1_2, filters[1], dropout_rate[1], dropout_type[1], activation, kernel_initializer, unet_type)
    skip2_2 = Concatenate(axis=-1)([x2_1, skip3_0])
    x2_2 = upsampling_block(x3_1, skip2_2, filters[2], dropout_rate[2], dropout_type[2], activation, kernel_initializer, unet_type)
    
    skip0_3 = Concatenate(axis=-1)([x0_2, x0_1, skip1_0])
    x0_3 = upsampling_block(x1_2, skip0_3, filters[0], dropout_rate[0], dropout_type[0], activation, kernel_initializer, unet_type)
    skip1_3 = Concatenate(axis=-1)([x1_2, x1_1, skip2_0])
    x1_3 = upsampling_block(x2_2, skip1_3, filters[1], dropout_rate[1], dropout_type[1], activation, kernel_initializer, unet_type)
    
    skip0_4 = Concatenate(axis=-1)([x0_3, x0_2, x0_1, skip1_0])
    x0_4 = upsampling_block(x1_3, skip0_4, filters[0], dropout_rate[0], dropout_type[0], activation, kernel_initializer, unet_type)
    
    
    output_1 = Conv2D(num_classes, kernel_size=1, kernel_initializer=kernel_initializer)(x0_1)
    output_1 = Activation('softmax', name='output_1', dtype='float32')(output_1)
    
    output_2 = Conv2D(num_classes, kernel_size=1, kernel_initializer=kernel_initializer)(x0_2)
    output_2 = Activation('softmax', name='output_2', dtype='float32')(output_2)
    
    output_3 = Conv2D(num_classes, kernel_size=1, kernel_initializer=kernel_initializer)(x0_3)
    output_3 = Activation('softmax', name='output_3', dtype='float32')(output_3)
    
    output_4 = Conv2D(num_classes, kernel_size=1, kernel_initializer=kernel_initializer)(x0_4)
    output_4 = Activation('softmax', name='output_4', dtype='float32')(output_4)
    
    if deep_supervision:
        model = Model(inputs=x0_0, outputs=[output_1,
                                            output_2,
                                            output_3,
                                            output_4] , name='Unet_2plus')
    else:
        model = Model(inputs=x0_0, outputs=output_4, name='Unet_2plus')
    
    return model


def Unet_3plus(input_shape: tuple,
               filters: tuple, 
               num_classes: int, 
               dropout_rate: float, 
               dropout_type: str,
               activation: str,
               scale_dropout = True,
               dropout_offset = 0.01,
               kernel_initializer = HeNormal(42),
               attention: bool = False,
               backbone_name: str = None,
               freeze_backbone: bool = True,
               deep_supervision: bool = False,
               interpolation: str = 'bilinear'):    
    
    depth = len(filters)
    
    assert depth==5, f'Expected a list of filters with length : 5. Instead got length : {depth}.'
    
    if attention:
        unet_type = 'attention'
    else:
        unet_type = 'normal'
    
    offset = dropout_offset if scale_dropout else 0  
    
    if isinstance(dropout_rate, (list, tuple)):
        assert len(dropout_rate)== depth, 'dropout_rate length does not match model depth. The length of dropout_rate needs to match the length of filters if passed as a list or tuple'
    else:
        offset = dropout_offset if scale_dropout else 0
        dropout_rate = [dropout_rate + offset*i for i in range(depth)]
        
    if isinstance(dropout_type, (list, tuple)):
        assert len(dropout_type)==depth, 'dropout_type length does not match model depth. The length of dropout_type needs to match the length of filters if passed as a list or tuple'
    else:
        dropout_type = [dropout_type] * depth  
        
    map_filters = filters[0]
    decode_filters = 5*map_filters
    
    
    # -----------------------------------------------------------------------------------------------------------------------------------------
    x0 = tf.keras.Input(shape=input_shape)
    
    if backbone_name is None:
        # ENCODER
        x1_en, skip1 = downsampling_block(x0, filters[0], dropout_rate[0], dropout_type[0], activation, kernel_initializer, unet_type)
        x2_en, skip2 = downsampling_block(x1_en, filters[1], dropout_rate[1], dropout_type[1], activation, kernel_initializer, unet_type)
        x3_en, skip3 = downsampling_block(x2_en, filters[2], dropout_rate[2], dropout_type[1], activation, kernel_initializer, unet_type)
        x4_en, skip4 = downsampling_block(x3_en, filters[3], dropout_rate[3], dropout_type[1], activation, kernel_initializer, unet_type)
        
        x5_en = conv_block(x4_en, filters[4], dropout_rate[4], dropout_type[4], activation, kernel_initializer, unet_type)
    
    else:
        backbone = get_backbone(backbone_name, input_tensor=x0, freeze_backbone=freeze_backbone, depth=depth)
        skip1, skip2, skip3, skip4, x5_en = backbone(x0, training=False)    
        depth = depth + 1

    # DECODER
    # -------------------------------------------------------------------------------------------------------------------------
    x_de= []
    x_de.append(x5_en)
    
    down4_1 = MaxPooling2D(pool_size=8, strides=8)(skip1)
    down4_1 = Conv2D(map_filters, kernel_size=3, padding='same', kernel_initializer=kernel_initializer)(down4_1)
    down4_2 = MaxPooling2D(pool_size=4, strides=4)(skip2)
    down4_2 = Conv2D(map_filters, kernel_size=3, padding='same', kernel_initializer=kernel_initializer)(down4_2)
    down4_3 = MaxPooling2D(pool_size=2, strides=2)(skip3)
    down4_3 = Conv2D(map_filters, kernel_size=3, padding='same', kernel_initializer=kernel_initializer)(down4_3)
    skip4 = Conv2D(map_filters, kernel_size=3, padding='same', kernel_initializer=kernel_initializer)(skip4)
    up4_5 = UpSampling2D(2, interpolation=interpolation)(x5_en)
    up4_5 = Conv2D(map_filters, kernel_size=3, padding='same', kernel_initializer=kernel_initializer)(up4_5)
    
    skip4 = Concatenate(axis=-1)([down4_1, down4_2, down4_3, skip4, up4_5])
    x4_de = Conv2D(decode_filters, kernel_size=3, padding='same', kernel_initializer=kernel_initializer)(skip4)
    x4_de = BatchNormalization()(x4_de)
    x4_de = Activation(activation)(x4_de)
    x4_de = dropout_layer(x4_de, dropout_type[depth-2], dropout_rate[depth-2])
    x_de.append(x4_de)
    
    # -------------------------------------------------------------------------------------------------------------------------
    down3_1 = MaxPooling2D(pool_size=4, strides=4)(skip1)
    down3_1 = Conv2D(map_filters, kernel_size=3, padding='same', kernel_initializer=kernel_initializer)(down3_1)
    down3_2 = MaxPooling2D(pool_size=2, strides=2)(skip2)
    down3_2 = Conv2D(map_filters, kernel_size=3, padding='same', kernel_initializer=kernel_initializer)(down3_2)
    skip3 =Conv2D(map_filters, kernel_size=3, padding='same', kernel_initializer=kernel_initializer)(skip3)
    up3_4 = UpSampling2D(2, interpolation=interpolation)(x4_de)
    up3_4 = Conv2D(map_filters, kernel_size=3, padding='same', kernel_initializer=kernel_initializer)(up3_4)
    up3_5 = UpSampling2D(4, interpolation=interpolation)(x5_en)
    up3_5 = Conv2D(map_filters, kernel_size=3, padding='same', kernel_initializer=kernel_initializer)(up3_5)
    
    skip3 = Concatenate(axis=-1)([down3_1, down3_2, skip3, up3_4, up3_5])
    x3_de = Conv2D(decode_filters, kernel_size=3, padding='same', kernel_initializer=kernel_initializer)(skip3)
    x3_de = BatchNormalization()(x3_de)
    x3_de = Activation(activation)(x3_de)
    x3_de = dropout_layer(x3_de, dropout_type[depth-3], dropout_rate[depth-3]) 
    x_de.append(x3_de)
    
    # -------------------------------------------------------------------------------------------------------------------------
    down2_1 = MaxPooling2D(pool_size=2, strides=2)(skip1)
    down2_1 = Conv2D(map_filters, kernel_size=3, padding='same', kernel_initializer=kernel_initializer)(down2_1)
    skip2 = Conv2D(map_filters, kernel_size=3, padding='same', kernel_initializer=kernel_initializer)(skip2)
    up2_3 = UpSampling2D(2, interpolation=interpolation)(x3_de)
    up2_3 = Conv2D(map_filters, kernel_size=3, padding='same', kernel_initializer=kernel_initializer)(up2_3)
    up2_4 = UpSampling2D(4, interpolation=interpolation)(x4_de)
    up2_4 = Conv2D(map_filters, kernel_size=3, padding='same', kernel_initializer=kernel_initializer)(up2_4)
    up2_5 = UpSampling2D(8, interpolation=interpolation)(x5_en)
    up2_5 = Conv2D(map_filters, kernel_size=3, padding='same', kernel_initializer=kernel_initializer)(up2_5)
    
    skip2 = Concatenate(axis=-1)([down2_1, skip2, up2_3, up2_4, up2_5])
    x2_de = Conv2D(decode_filters, kernel_size=3, padding='same', kernel_initializer=kernel_initializer)(skip2)
    x2_de = BatchNormalization()(x2_de)
    x2_de = Activation(activation)(x2_de)
    x2_de = dropout_layer(x2_de, dropout_type[depth-4], dropout_rate[depth-4])
    x_de.append(x2_de)
    
    # -------------------------------------------------------------------------------------------------------------------------
    skip1 = Conv2D(map_filters, kernel_size=3, padding='same', kernel_initializer=kernel_initializer)(skip1)
    up1_2 = UpSampling2D(2, interpolation=interpolation)(x2_de)
    up1_2 = Conv2D(map_filters, kernel_size=3, padding='same', kernel_initializer=kernel_initializer)(up1_2)
    up1_3 = UpSampling2D(4, interpolation=interpolation)(x3_de)
    up1_3 = Conv2D(map_filters, kernel_size=3, padding='same', kernel_initializer=kernel_initializer)(up1_3)
    up1_4 = UpSampling2D(8, interpolation=interpolation)(x4_de)
    up1_4 = Conv2D(map_filters, kernel_size=3, padding='same', kernel_initializer=kernel_initializer)(up1_4)
    up1_5 = UpSampling2D(16, interpolation=interpolation)(x5_en)
    up1_5 = Conv2D(map_filters, kernel_size=3, padding='same', kernel_initializer=kernel_initializer)(up1_5)
    
    skip1 = Concatenate(axis=-1)([skip1, up1_2, up1_3, up1_4, up1_5])
    x1_de = Conv2D(decode_filters, kernel_size=3, padding='same', kernel_initializer=kernel_initializer)(skip1)
    x1_de = BatchNormalization()(x1_de)
    x1_de = Activation(activation)(x1_de)
    x1_de = dropout_layer(x1_de, dropout_type[depth-5], dropout_rate[depth-5])
    x_de.append(x1_de)
    
    # add another upsampling operation if backbone is used
    if backbone_name is not None:
        x0_de = UpSampling2D(2, interpolation=interpolation)(x1_de)
        x0_de = Conv2D(decode_filters, kernel_size=3, padding='same', kernel_initializer=kernel_initializer)(x0_de)
        x0_de = dropout_layer(x0_de, dropout_type[depth-6], dropout_rate[depth-6])
        x_de.append(x0_de)

    
    if deep_supervision:
        x_de = x_de[::-1]
        output = []
        for i in range(depth):
            out = Conv2D(num_classes, kernel_size=1, kernel_initializer=kernel_initializer)(x_de[i])
            out = Activation('softmax', name=f'output_{i+1}', dtype='float32')(out)
            output.append(out)
    else:
        output = Conv2D(num_classes, kernel_size=1, kernel_initializer=kernel_initializer)(x_de[-1])
        output = Activation('softmax', name='output', dtype='float32')(output)
   
    model = Model(inputs=x0, outputs=output, name='Unet_3plus')
    
    return model