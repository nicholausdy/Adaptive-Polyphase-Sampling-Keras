import numpy as np
import tensorflow as tf
from tensorflow import keras

class APSLayer(keras.layers.Layer):
    def __init__(self, stride = 2, order = 2, name=None, **kwargs):
        super(APSLayer, self).__init__(name=name)
        self.stride = stride
        self.order = order
        super(APSLayer, self).__init__(**kwargs)

    def call(self, inputs):
        downsampled, max_norm_index = self.downsample(inputs, self.stride, self.order)
        return downsampled, max_norm_index

    def get_config(self):
        config = super().get_config()
        config.update({
            "stride": self.stride,
            "order": self.order
        })
        return config

    @tf.function
    def downsample(self, matrix, stride=2, order=2):
        # gather polyphase components
        polyphase_components = tf.TensorArray(tf.float32, size = 0, dynamic_size=True, clear_after_read=False)
        input_shape = matrix.shape.as_list()
        num_row = input_shape[1]
        num_column = input_shape[2]
        num_channel = input_shape[3]
        arr_index = 0
        for i in range(stride):
            for j in range(stride):
                strided_matrix = tf.strided_slice(
                    matrix, 
                    begin=[0,i,j,0], 
                    end=[0, num_row, num_column, num_channel], 
                    strides=[1, self.stride, self.stride, 1],
                    begin_mask=9,
                    end_mask=9
                )
                # self.polyphase_components[polyphase_indices] = strided_matrix
                strided_matrix = tf.cast(strided_matrix, dtype=np.float32)
                polyphase_components = polyphase_components.write(arr_index, strided_matrix)
                arr_index += 1

        # find polyphase element with maximum norm
        norms_array = tf.TensorArray(tf.float32, size = 0, dynamic_size=True, clear_after_read=False)
        arr_index = 0
        for i in tf.range(self.stride*2):
            comp = polyphase_components.read(i)
            norm = tf.norm(tensor= comp, ord= order)
            norms_array = norms_array.write(arr_index, norm)
            arr_index += 1
        norms_array = norms_array.stack()
        max_norm_index = tf.math.argmax(norms_array)
        max_norm_index = tf.cast(max_norm_index, dtype=np.int32)
        return polyphase_components.read(max_norm_index), max_norm_index

class APSDownsampleGivenPolyIndices(keras.layers.Layer):  
    def __init__(self, stride = 2, name=None, **kwargs):
        super(APSDownsampleGivenPolyIndices, self).__init__(name=name)
        self.stride = stride
        super(APSDownsampleGivenPolyIndices, self).__init__(**kwargs)

    def call(self, inputs, max_poly_indices):        
        strided_matrix = self.downsample(inputs, max_poly_indices)
        return strided_matrix

    @tf.function
    def downsample(self, inputs, max_poly_indices):
        lookup = tf.TensorArray(tf.int32, size = 0, dynamic_size=True, clear_after_read=False)
        arr_index = 0
        for i in range(self.stride):
            for j in range(self.stride):
                elem = tf.constant([0,i,j,0], dtype=tf.int32)
                lookup = lookup.write(arr_index, elem)
                arr_index += 1
        input_shape = inputs.shape.as_list()
        num_row = input_shape[1]
        num_column = input_shape[2]
        num_channel = input_shape[3]
        max_poly_indices = lookup.read(max_poly_indices)
        strided_matrix = tf.strided_slice(
            inputs,
            begin= max_poly_indices,
            end = tf.constant([0, num_row, num_column, num_channel], dtype=tf.int32),
            strides = tf.constant([1, self.stride, self.stride, 1], dtype=tf.int32),
            begin_mask=9,
            end_mask=9
        )
        return tf.cast(strided_matrix, dtype=np.float32)


    def get_config(self):
        config = super().get_config()
        config.update({
            "stride": self.stride,
        })
        return config
