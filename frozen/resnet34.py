# @author : Abhishek R S

import os
import numpy as np
import h5py
import tensorflow as tf

'''

ResNet-34

# Reference
- [Deep Residual Learning for Image Recognition]
  (https://arxiv.org/abs/1512.03385)

# Pretrained model weights
- [Download pretrained resnet-34 model]
  (https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet34_imagenet_1000_no_top.h5)

'''

class ResNet34:

    # initialize network parameters
    def __init__(self, data_format = 'channels_first', resnet_path = 'resnet34_imagenet_1000_no_top.h5'):
        self._weights_h5 = h5py.File(resnet_path, 'r')
        self._data_format = data_format
        self._encoder_data_format = None
        self._padding = 'SAME'
        self._pool_kernel = None
        self._pool_strides = None
        self._res_conv_strides = [1, 1, 1, 1]

        if self._data_format == 'channels_first':
            self._encoder_data_format = 'NCHW'
            self._pool_kernel = [1, 1, 3, 3]
            self._pool_strides = [1, 1, 2, 2]
        else: 
            self._encoder_data_format = 'NHWC'
            self._pool_kernel = [1, 3, 3, 1]
            self._pool_strides = [1, 2, 2, 1]

    # build resnet-34 encoder
    def resnet34_encoder(self, features):

        # input : BGR format in range [0-255]

        # Stage 0
        self.stage0 = self._conv_layer(features, 'conv0', strides = self._pool_strides)
        self.stage0 = self._batchnorm_layer(self.stage0, 'bn0')
        self.stage0 = tf.nn.relu(self.stage0, name = 'relu0')
        
        # Stage 1
        self.stage1 = tf.nn.max_pool(self.stage0, ksize = self._pool_kernel, strides = self._pool_strides, padding = self._padding, data_format = self._encoder_data_format, name = 'pool1')

        # Stage 2
        self.stage2 = self._res_conv_block(input_layer = self.stage1, stage = 'stage1_unit1_', strides = self._res_conv_strides)
        self.stage2 = self._res_identity_block(input_layer = self.stage2, stage = 'stage1_unit2_')
        self.stage2 = self._res_identity_block(input_layer = self.stage2, stage = 'stage1_unit3_')

        # Stage 3
        self.stage3 = self._res_conv_block(input_layer = self.stage2, stage = 'stage2_unit1_', strides = self._pool_strides)
        self.stage3 = self._res_identity_block(input_layer = self.stage3, stage = 'stage2_unit2_')
        self.stage3 = self._res_identity_block(input_layer = self.stage3, stage = 'stage2_unit3_')
        self.stage3 = self._res_identity_block(input_layer = self.stage3, stage = 'stage2_unit4_')

        # Stage 4
        self.stage4 = self._res_conv_block(input_layer = self.stage3, stage = 'stage3_unit1_', strides = self._pool_strides)
        self.stage4 = self._res_identity_block(input_layer = self.stage4, stage = 'stage3_unit2_')
        self.stage4 = self._res_identity_block(input_layer = self.stage4, stage = 'stage3_unit3_')
        self.stage4 = self._res_identity_block(input_layer = self.stage4, stage = 'stage3_unit4_')
        self.stage4 = self._res_identity_block(input_layer = self.stage4, stage = 'stage3_unit5_')
        self.stage4 = self._res_identity_block(input_layer = self.stage4, stage = 'stage3_unit6_')

        # Stage 5
        self.stage5 = self._res_conv_block(input_layer = self.stage4, stage = 'stage4_unit1_', strides = self._pool_strides)
        self.stage5 = self._res_identity_block(input_layer = self.stage5, stage = 'stage4_unit2_')
        self.stage5 = self._res_identity_block(input_layer = self.stage5, stage = 'stage4_unit3_')
        self.stage5 = self._batchnorm_layer(self.stage5, 'bn1')
        self.stage5 = tf.nn.relu(self.stage5, name = 'relu1')

    #-------------------------------------#
    # pretrained resnet encoder functions #
    #-------------------------------------#
    #-----------------------#
    # convolution2d layer   #
    #-----------------------#
    def _conv_layer(self, input_layer, name, strides = [1, 1, 1, 1]):
        hierarchy_name = list(self._weights_h5[name])[0]
        W = tf.constant(self._weights_h5[name][hierarchy_name]['kernel:0'])
        x = tf.nn.conv2d(input_layer, filter = W, strides = strides, padding = self._padding, data_format = self._encoder_data_format, name = name + 'conv')

        return x

    #-----------------------#
    # batchnorm layer       #
    #-----------------------#
    def _batchnorm_layer(self, input_layer, name):
        if self._encoder_data_format == 'NCHW':
            input_layer = tf.transpose(input_layer, perm = [0, 2, 3, 1])

        hierarchy_name = list(self._weights_h5[name])[0]
        mean = tf.constant(self._weights_h5[name][hierarchy_name]['moving_mean:0'])
        std = tf.constant(self._weights_h5[name][hierarchy_name]['moving_variance:0'])
        beta = tf.constant(self._weights_h5[name][hierarchy_name]['beta:0'])
        gamma = tf.constant(self._weights_h5[name][hierarchy_name]['gamma:0'])

        bn = tf.nn.batch_normalization(input_layer, mean = mean, variance = std, offset = beta, scale = gamma, variance_epsilon = 1e-12, name = name)

        if self._encoder_data_format == 'NCHW':
            bn = tf.transpose(bn, perm = [0, 3, 1, 2])

        return bn

    #-----------------------#
    # convolution block     #
    #-----------------------#
    def _res_conv_block(self, input_layer, stage, strides):
        x = self._batchnorm_layer(input_layer, name = stage + 'bn1')
        x = tf.nn.relu(x, name = stage + 'relu1')

        shortcut = x
        x = self._conv_layer(x, name = stage + 'conv1', strides = strides)

        x = self._batchnorm_layer(x, name = stage + 'bn2')
        x = tf.nn.relu(x, name = stage + 'relu2')
        x = self._conv_layer(x, name = stage + 'conv2')

        shortcut = self._conv_layer(shortcut, name = stage + 'sc', strides = strides)
        x = tf.add(x, shortcut, name = stage + 'add')

        return x

    #-----------------------#
    # identity block        #
    #-----------------------#
    def _res_identity_block(self, input_layer, stage):
        x = self._batchnorm_layer(input_layer, name = stage + 'bn1')
        x = tf.nn.relu(x, name = stage + 'relu1')
        x = self._conv_layer(x, name = stage + 'conv1')

        x = self._batchnorm_layer(x, name = stage + 'bn2')
        x = tf.nn.relu(x, name = stage + 'relu2')
        x = self._conv_layer(x, name = stage + 'conv2')

        x = tf.add(x, input_layer, name = stage + 'add')

        return x
