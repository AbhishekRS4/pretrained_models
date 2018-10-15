# @author : Abhishek R S

import os
import numpy as np
import h5py
import tensorflow as tf

'''

Xception model

# Reference
- [Xception: Deep Learning with Depthwise Separable Convolutions](
   https://arxiv.org/abs/1610.02357)

# Pretrained model weights
- [Download pretrained xception model]
  (https://github.com/fchollet/deep-learning-models/releases/)

'''

class Xception:

    # initialize network parameters
    def __init__(self, data_format = 'channels_first', xception_path = 'xception_weights_tf_dim_ordering_tf_kernels_notop.h5'):
        self._weights_h5 = h5py.File(xception_path, 'r')
        self._data_format = data_format
        self._encoder_data_format = None
        self._pool_kernel = None
        self._pool_strides = None
        self._padding = 'SAME'
        self._conv_strides = [1, 1, 1, 1]
        self._reduction_rate = 0.5
        self._feature_map_axis = None

        if self._data_format == 'channels_first':
            self._encoder_data_format = 'NCHW'
            self._feature_map_axis = 1
            self._pool_kernel = [1, 1, 3, 3]
            self._pool_strides = [1, 1, 2, 2]
        else: 
            self._encoder_data_format = 'NHWC'
            self._feature_map_axis = -1
            self._pool_kernel = [1, 3, 3, 1]
            self._pool_strides = [1, 2, 2, 1]

    # build xception encoder
    def xception_encoder(self, features):

        # input : RGB format
        # input = input / 127.5
        # input = input - 1.

        # Stage 1
        self.stage1 = self._conv_layer(features, name = 'block1_conv1', weights_key = 'convolution2d_1', strides = self._pool_strides)
        self.stage1 = self._batchnorm_layer(self.stage1, name = 'block1_conv1_bn', weights_key = 'batchnormalization_1')
        self.stage1 = tf.nn.relu(self.stage1, name = 'block1_conv1_relu')
        
        self.stage1 = self._conv_layer(self.stage1, name = 'block1_conv2', weights_key = 'convolution2d_2')
        self.stage1 = self._batchnorm_layer(self.stage1, name = 'block1_conv2_bn', weights_key = 'batchnormalization_2')
        self.stage1 = tf.nn.relu(self.stage1, name = 'block1_conv2_relu')

        # Stage 2
        self.stage2_residual = self._conv_layer(self.stage1, name = 'block2_conv1', weights_key = 'convolution2d_3', strides = self._pool_strides)
        self.stage2_residual = self._batchnorm_layer(self.stage2_residual, name = 'block2_conv1_bn', weights_key = 'batchnormalization_3')

        self.stage2 = self._separable_conv_layer(self.stage1, name = 'block2_sepconv1', weights_key = 'separableconvolution2d_1')
        self.stage2 = self._batchnorm_layer(self.stage2, name = 'block2_sepconv1_bn', weights_key = 'batchnormalization_4')
        self.stage2 = tf.nn.relu(self.stage2, name = 'block2_sepconv1_relu')

        self.stage2 = self._separable_conv_layer(self.stage2, name = 'block2_sepconv2', weights_key = 'separableconvolution2d_2')
        self.stage2 = self._batchnorm_layer(self.stage2, name = 'block2_sepconv2_bn', weights_key = 'batchnormalization_5')

        self.stage2 = tf.nn.max_pool(self.stage2, ksize = self._pool_kernel, strides = self._pool_strides, padding = self._padding, data_format = self._encoder_data_format, name = 'block2_pool')
        self.stage2 = tf.add(self.stage2_residual, self.stage2, name = 'block2_add')

        # Stage 3
        self.stage3_residual = self._conv_layer(self.stage2, name = 'block3_conv1', weights_key = 'convolution2d_4', strides = self._pool_strides)
        self.stage3_residual = self._batchnorm_layer(self.stage3_residual, name = 'block3_conv1_bn', weights_key = 'batchnormalization_6')

        self.stage3 = self._separable_conv_layer(self.stage2, name = 'block3_sepconv1', weights_key = 'separableconvolution2d_3')
        self.stage3 = self._batchnorm_layer(self.stage3, name = 'block3_sepconv1_bn', weights_key = 'batchnormalization_7')
        self.stage3 = tf.nn.relu(self.stage3, name = 'block3_sepconv1_relu')

        self.stage3 = self._separable_conv_layer(self.stage3, name = 'block3_sepconv2', weights_key = 'separableconvolution2d_4')
        self.stage3 = self._batchnorm_layer(self.stage3, name = 'block3_sepconv2_bn', weights_key = 'batchnormalization_8')

        self.stage3 = tf.nn.max_pool(self.stage3, ksize = self._pool_kernel, strides = self._pool_strides, padding = self._padding, data_format = self._encoder_data_format, name = 'block3_pool')
        self.stage3 = tf.add(self.stage3_residual, self.stage3, name = 'block3_add')

        # Stage 4
        self.stage4_residual = self._conv_layer(self.stage3, name = 'block4_conv1', weights_key = 'convolution2d_5', strides = self._pool_strides)
        self.stage4_residual = self._batchnorm_layer(self.stage4_residual, name = 'block4_conv1_bn', weights_key = 'batchnormalization_9')

        self.stage4 = self._separable_conv_layer(self.stage3, name = 'block4_sepconv1', weights_key = 'separableconvolution2d_5')
        self.stage4 = self._batchnorm_layer(self.stage4, name = 'block4_sepconv1_bn', weights_key = 'batchnormalization_10')
        self.stage4 = tf.nn.relu(self.stage4, name = 'block4_sepconv1_relu')

        self.stage4 = self._separable_conv_layer(self.stage4, name = 'block4_sepconv2', weights_key = 'separableconvolution2d_6')
        self.stage4 = self._batchnorm_layer(self.stage4, name = 'block4_sepconv2_bn', weights_key = 'batchnormalization_11')

        self.stage4 = tf.nn.max_pool(self.stage4, ksize = self._pool_kernel, strides = self._pool_strides, padding = self._padding, data_format = self._encoder_data_format, name = 'block4_pool')
        self.stage4 = tf.add(self.stage4_residual, self.stage4, name = 'block4_add')
        self.temp_layer = self.stage4

        # Stage 5
        for i in range(8):
            self.temp_residual_layer = self.temp_layer
            prefix = 'block' + str(i + 5) + '_'

            self.temp_layer = tf.nn.relu(self.temp_layer, name = prefix + 'sepconv1_relu')
            self.temp_layer = self._separable_conv_layer(self.temp_layer, name = prefix + 'sepconv1', weights_key = 'separableconvolution2d_' + str(7 + 3 * i))
            self.temp_layer = self._batchnorm_layer(self.temp_layer, name = prefix + 'sepconv1_bn', weights_key = 'batchnormalization_' + str(12 + 3 * i))

            self.temp_layer = tf.nn.relu(self.temp_layer, name = prefix + 'sepconv2_relu')
            self.temp_layer = self._separable_conv_layer(self.temp_layer, name = prefix + 'sepconv2', weights_key = 'separableconvolution2d_' + str(8 + 3 * i))
            self.temp_layer = self._batchnorm_layer(self.temp_layer, name = prefix + 'sepconv2_bn', weights_key = 'batchnormalization_' + str(13 + 3 * i))

            self.temp_layer = tf.nn.relu(self.temp_layer, name = prefix + 'sepconv3_relu')
            self.temp_layer = self._separable_conv_layer(self.temp_layer, name = prefix + 'sepconv3', weights_key = 'separableconvolution2d_' + str(9 + 3 * i))
            self.temp_layer = self._batchnorm_layer(self.temp_layer, name = prefix + 'sepconv3_bn', weights_key = 'batchnormalization_' + str(14 + 3 * i))

            self.temp_layer = tf.add(self.temp_residual_layer, self.temp_layer, name = prefix + 'add')

        # Stage 6
        self.stage13_residual = self._conv_layer(self.temp_layer, name = 'block13_conv1', weights_key = 'convolution2d_6', strides = self._pool_strides)
        self.stage13_residual = self._batchnorm_layer(self.stage13_residual, name = 'block13_conv1_bn',  weights_key = 'batchnormalization_36')

        self.stage13 = self._separable_conv_layer(self.temp_layer, name = 'block13_sepconv1', weights_key = 'separableconvolution2d_31')
        self.stage13 = self._batchnorm_layer(self.stage13, name = 'block13_sepconv1_bn', weights_key = 'batchnormalization_37')
        self.stage13 = tf.nn.relu(self.stage13, name = 'block13_sepconv1_relu')

        self.stage13 = self._separable_conv_layer(self.stage13, name = 'block13_sepconv2', weights_key = 'separableconvolution2d_32')
        self.stage13 = self._batchnorm_layer(self.stage13, name = 'block13_sepconv2_bn', weights_key = 'batchnormalization_38')

        self.stage13 = tf.nn.max_pool(self.stage13, ksize = self._pool_kernel, strides = self._pool_strides, padding = self._padding, data_format = self._encoder_data_format, name = 'block13_pool')
        self.stage13 = tf.add(self.stage13_residual, self.stage13, name = 'block13_add')

        # Stage 7
        self.stage14 = self._separable_conv_layer(self.stage13, name = 'block14_sepconv1', weights_key = 'separableconvolution2d_33')
        self.stage14 = self._batchnorm_layer(self.stage14, name = 'block14_sepconv1_bn', weights_key = 'batchnormalization_39')
        self.stage14 = tf.nn.relu(self.stage14, name = 'block14_sepconv1_relu')

        self.stage14 = self._separable_conv_layer(self.stage14, name = 'block14_sepconv2', weights_key = 'separableconvolution2d_34')
        self.stage14 = self._batchnorm_layer(self.stage14, name = 'block14_sepconv2_bn', weights_key = 'batchnormalization_40')
        self.stage14 = tf.nn.relu(self.stage14, name = 'block14_sepconv2_relu')

    #---------------------------------------#
    # pretrained xception encoder functions #
    #---------------------------------------#
    #-----------------------#
    # convolution2d layer   #
    #-----------------------#
    def _conv_layer(self, input_layer, name, weights_key, strides = [1, 1, 1, 1]):
        W = tf.constant(self._weights_h5[weights_key][weights_key + '_W:0'])

        x = tf.nn.conv2d(input_layer, filter = W, strides = strides, padding = self._padding, data_format = self._encoder_data_format, name = name)

        return x

    #---------------------------------#
    # separable convolution2d layer   #
    #---------------------------------#
    def _separable_conv_layer(self, input_layer, name, weights_key, strides = [1, 1, 1, 1]):
        W_depthwise = tf.constant(self._weights_h5[weights_key][weights_key + '_depthwise_kernel:0'])
        W_pointwise = tf.constant(self._weights_h5[weights_key][weights_key + '_pointwise_kernel:0'])

        x = tf.nn.separable_conv2d(input_layer, depthwise_filter = W_depthwise, pointwise_filter = W_pointwise, strides = strides, padding = self._padding, data_format = self._encoder_data_format, name = name)

        return x

    #-----------------------#
    # batchnorm layer       #
    #-----------------------#
    def _batchnorm_layer(self, input_layer, name, weights_key):
        if self._encoder_data_format == 'NCHW':
            input_layer = tf.transpose(input_layer, perm = [0, 2, 3, 1])

        mean = tf.constant(self._weights_h5[weights_key][weights_key + '_running_mean:0'])
        std = tf.constant(self._weights_h5[weights_key][weights_key + '_running_std:0'])
        beta = tf.constant(self._weights_h5[weights_key][weights_key + '_beta:0'])
        gamma = tf.constant(self._weights_h5[weights_key][weights_key + '_gamma:0'])

        bn = tf.nn.batch_normalization(input_layer, mean = mean, variance = std, offset = beta, scale = gamma, variance_epsilon = 1e-12, name = name)

        if self._encoder_data_format == 'NCHW':
            bn = tf.transpose(bn, perm = [0, 3, 1, 2])

        return bn
