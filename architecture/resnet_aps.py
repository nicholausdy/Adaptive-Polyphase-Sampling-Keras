# Reference: https://towardsdatascience.com/building-a-resnet-in-keras-e8f1322a49ba


import tensorflow as tf
from tensorflow.keras import layers
from architecture.aps import APSLayer, APSDownsampleGivenPolyIndices
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, GlobalAveragePooling2D, Dense, Reshape
from tensorflow.keras.models import Model


class CircularPad(layers.Layer):
 
    def __init__(self, padding=(1, 1, 1, 1)):
 
        super(CircularPad, self).__init__()
        
        self.pad_sizes = padding

    def call(self, x):
 
        top_pad, bottom_pad, left_pad, right_pad = self.pad_sizes
        height_pad = tf.concat([x[:, -bottom_pad:], x, x[:, :top_pad]], axis=1)
 
        return tf.concat([height_pad[:, :, -right_pad:], height_pad, height_pad[:, :, :left_pad]], axis=2)


def non_downsampling_residual_block(x, filters, kernel_size, lyr = None):

    y = Conv2D(kernel_size = kernel_size, strides = 1, filters = filters)(CircularPad(padding = (1,1,1,1))(x))
    y = BatchNormalization()(y)
    y = layers.Activation('relu')(y)

    y = Conv2D(kernel_size = kernel_size, strides = 1, filters = filters)(CircularPad(padding = (1,1,1,1))(y))
    y = BatchNormalization()(y)
    y = layers.Activation('relu')(y)

    out = Add()([x, y])
    out = BatchNormalization()(out)
    out = layers.Activation('relu')(out)
    
    return out


def downsampling_residual_block(x, filters, kernel_size, stride=2, order=2, lyr = None):
    
    y = Conv2D(kernel_size = kernel_size, strides = 1, filters = filters)(CircularPad(padding = (1,1,1,1))(x))
    
    y, max_norm_index = APSLayer(stride=stride, order=order)(y)
    y = BatchNormalization()(y)
    y = layers.Activation('relu')(y)

    y = Conv2D(kernel_size = kernel_size, strides = 1, filters = filters)(CircularPad(padding = (1,1,1,1))(y))

    x = APSDownsampleGivenPolyIndices(stride=stride)(x, max_norm_index)
    x = Conv2D(kernel_size = kernel_size, strides = 1, filters = filters)(CircularPad(padding = (1,1,1,1))(x))

    out = Add()([x, y])
    out = BatchNormalization()(out)
    out = layers.Activation('relu')(out)
    
    return out


def build_resnet_20_aps(input_shape):

    num_filters = 16
    kernel_size = 3
    num_residual_block = 6

    inputs = Input(shape= input_shape)

    t = Conv2D(kernel_size = kernel_size, strides = 1, filters = num_filters)(CircularPad(padding = (1,1,1,1))(inputs))
    t = BatchNormalization()(t)
    t = layers.Activation('relu')(t)

    for i in range(num_residual_block):
    
        if  (i > 1) and (i % 2 == 0):
    
            t = downsampling_residual_block(
                t,
                filters = num_filters,
                kernel_size = kernel_size,
                stride = 2,
                order = 2, lyr = i
            ) 
     
        else:
     
            t = non_downsampling_residual_block(
                t,
                filters = num_filters,
                kernel_size = kernel_size, lyr = i
            )
     
        if (i % 2 != 0):
     
            num_filters *= 2

    t = GlobalAveragePooling2D()(t)
    outputs = Dense(10, activation='softmax')(t)

    model = Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate= 0.01)
    model.compile(
        optimizer = optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model