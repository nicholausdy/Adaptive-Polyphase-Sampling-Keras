# Reference: https://towardsdatascience.com/building-a-resnet-in-keras-e8f1322a49ba

import tensorflow as tf
from tensorflow import keras
from architecture.aps import APSLayer, APSDownsampleGivenPolyIndices
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization,\
                                    Add, GlobalAveragePooling2D, Dense, Reshape
from tensorflow.keras.models import Model

def relu_bn_layer(inputs):
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn

def non_downsampling_residual_block(x, filters, kernel_size):
    y = Conv2D(
        kernel_size = kernel_size,
        strides = 1,
        filters = filters,
        padding="same"
    )(x)
    y = relu_bn_layer(y)
    y = keras.layers.Conv2D(
        kernel_size = kernel_size,
        strides = 1,
        filters = filters,
        padding="same"
    )(y)
    out = Add()([x,y])
    out = relu_bn_layer(out)
    return out

def downsampling_residual_block(x, filters, kernel_size, stride=2, order=2):
    y = Conv2D(
        kernel_size = kernel_size,
        strides = 1,
        filters = filters,
        padding = "same"
    )(x)
    aps_downsample = APSLayer(stride=stride, order=order)
    y, max_norm_index = aps_downsample(y)
    y = relu_bn_layer(y)
    y = Conv2D(
        kernel_size = kernel_size,
        strides = 1,
        filters = filters,
        padding = "same"
    )(y)
    aps_downsample_poly_indices = APSDownsampleGivenPolyIndices(stride=stride)
    x = aps_downsample_poly_indices(x, max_norm_index)
    x = Conv2D(
        kernel_size = 1,
        strides = 1,
        filters = filters,
        padding = "same"
    )(x)
    out = Add()([x,y])
    out = relu_bn_layer(out)
    return out

def build_resnet_20_aps(input_shape):
    inputs = Input(shape= input_shape)
    num_filters = 16
    kernel_size = 3
    t = BatchNormalization()(inputs)
    t = Conv2D(
        kernel_size = kernel_size,
        strides = 1,
        filters = num_filters,
        padding="same"
    )(t)
    t = relu_bn_layer(t)
    num_residual_block = 6
    for i in range(num_residual_block):
        if  (i > 1) and (i % 2 == 0):
            t = downsampling_residual_block(
                t,
                filters = num_filters,
                kernel_size = kernel_size,
                stride = 2,
                order = 2
            ) 
        else:
            t = non_downsampling_residual_block(
                t,
                filters = num_filters,
                kernel_size = kernel_size
            )
        if (i % 2 != 0):
            num_filters *= 2

    t = GlobalAveragePooling2D()(t)
    outputs = Dense(10, activation='softmax')(t)

    model = Model(inputs, outputs)
    optimizer = keras.optimizers.Adam(learning_rate= 0.01)
    model.compile(
        optimizer = optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model
