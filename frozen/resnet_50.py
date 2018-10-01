# @author : Abhishek R S

import os
import numpy as np
import h5py
import tensorflow as tf

class ResNet50:

    # initialize network parameters
    def __init__(self, data_format = 'channels_first', resnet_path = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'):
        self._weights_h5 = h5py.File(resnet_path, 'r')
        self._data_format = data_format
        self._res_data_format = None
        self._pool_kernel = None
        self._pool_strides = None
        self._res_conv_strides = [1, 1, 1, 1]

        if self._data_format == 'channels_first':
            self._res_data_format = 'NCHW'
            self._pool_kernel = [1, 1, 3, 3]
            self._pool_strides = [1, 1, 2, 2]
        else: 
            self._res_data_format = 'NHWC'
            self._pool_kernel = [1, 3, 3, 1]
            self._pool_strides = [1, 2, 2, 1]

    # build resnet encoder
    def resnet50_encoder(self, features):

        # input : BGR format with image_net mean subtracted
        # bgr mean : [103.939, 116.779, 123.68]

        # build the resnet-50 encoder
        # Stage 0
        self.stage0 = self._res_conv_layer(features, 'conv1', strides = self._pool_strides)
        self.stage0 = self._res_batch_norm_layer(self.stage0, 'bn_conv1')
        self.stage0 = tf.nn.relu(self.stage0, name = 'relu1')
        
        # Stage 1
        self.stage1 = tf.nn.max_pool(self.stage0, ksize = self._pool_kernel, strides = self._pool_strides, padding = 'SAME', data_format = self._res_data_format, name = 'pool_conv1')

        # Stage 2
        self.stage2 = self._res_conv_block(input_layer = self.stage1, stage = '2a', strides = self._res_conv_strides)
        self.stage2 = self._res_identity_block(input_layer = self.stage2, stage = '2b')
        self.stage2 = self._res_identity_block(input_layer = self.stage2, stage = '2c')

        # Stage 3
        self.stage3 = self._res_conv_block(input_layer = self.stage2, stage = '3a', strides = self._pool_strides)
        self.stage3 = self._res_identity_block(input_layer = self.stage3, stage = '3b')
        self.stage3 = self._res_identity_block(input_layer = self.stage3, stage = '3c')
        self.stage3 = self._res_identity_block(input_layer = self.stage3, stage = '3d')

        # Stage 4
        self.stage4 = self._res_conv_block(input_layer = self.stage3, stage = '4a', strides = self._pool_strides)
        self.stage4 = self._res_identity_block(input_layer = self.stage4, stage = '4b')
        self.stage4 = self._res_identity_block(input_layer = self.stage4, stage = '4c')
        self.stage4 = self._res_identity_block(input_layer = self.stage4, stage = '4d')
        self.stage4 = self._res_identity_block(input_layer = self.stage4, stage = '4e')
        self.stage4 = self._res_identity_block(input_layer = self.stage4, stage = '4f')

        # Stage 5
        self.stage5 = self._res_conv_block(input_layer = self.stage4, stage = '5a', strides = self._pool_strides)
        self.stage5 = self._res_identity_block(input_layer = self.stage5, stage = '5b')
        self.stage5 = self._res_identity_block(input_layer = self.stage5, stage = '5c')

    #-------------------------------------#
    # pretrained resnet encoder functions #
    #-------------------------------------#

    #-----------------------#
    # convolution2d layer   #
    #-----------------------#
    def _res_conv_layer(self, input_layer, layer_name, strides = [1, 1, 1, 1], padding = 'SAME'):
        W = tf.constant(self._weights_h5[layer_name][layer_name + '_W_1:0'])
        b = self._weights_h5[layer_name][layer_name + '_b_1:0']
        b = tf.constant(np.reshape(b, (b.shape[0])))
        x = tf.nn.conv2d(input_layer, filter = W, strides = strides, padding = padding, data_format = self._res_data_format, name = layer_name + 'conv')
        x = tf.nn.bias_add(x, b, data_format = self._res_data_format, name = layer_name + 'bias')

        return x

    #-----------------------#
    # batch norm layer
    #-----------------------#
    def _res_batch_norm_layer(self, input_layer, layer_name):
        if self._res_data_format == 'NCHW':
            input_layer = tf.transpose(input_layer, perm = [0, 2, 3, 1])

        mean = tf.constant(self._weights_h5[layer_name][layer_name + '_running_mean_1:0'])
        std = tf.constant(self._weights_h5[layer_name][layer_name + '_running_std_1:0'])
        beta = tf.constant(self._weights_h5[layer_name][layer_name + '_beta_1:0'])
        gamma = tf.constant(self._weights_h5[layer_name][layer_name + '_gamma_1:0'])

        bn = tf.nn.batch_normalization(input_layer, mean = mean, variance = std, offset = beta, scale = gamma, variance_epsilon = 1e-12, name = layer_name)

        if self._res_data_format == 'NCHW':
            bn = tf.transpose(bn, perm = [0, 3, 1, 2])

        return bn

    #-----------------------#
    # convolution block     #
    #-----------------------#
    def _res_conv_block(self, input_layer, stage, strides):
        x = self._res_conv_layer(input_layer, layer_name = 'res' + stage + '_branch2a', strides = strides)
        x = self._res_batch_norm_layer(x, layer_name = 'bn' + stage + '_branch2a')
        x = tf.nn.relu(x)

        x = self._res_conv_layer(x, layer_name = 'res' + stage + '_branch2b')
        x = self._res_batch_norm_layer(x, layer_name = 'bn' + stage + '_branch2b')
        x = tf.nn.relu(x)

        x = self._res_conv_layer(x, layer_name = 'res' + stage + '_branch2c')
        x = self._res_batch_norm_layer(x, layer_name = 'bn' + stage + '_branch2c')

        shortcut = self._res_conv_layer(input_layer, layer_name = 'res' + stage + '_branch1', strides = strides)
        shortcut = self._res_batch_norm_layer(shortcut, layer_name = 'bn' + stage + '_branch1')

        x = tf.add(x, shortcut, name = 'add' + stage)
        x = tf.nn.relu(x, name = 'relu' + stage)

        return x

    #-----------------------#
    # identity block        #
    #-----------------------#
    def _res_identity_block(self, input_layer, stage):
        x = self._res_conv_layer(input_layer, layer_name = 'res' + stage + '_branch2a')
        x = self._res_batch_norm_layer(x, layer_name = 'bn' + stage + '_branch2a')
        x = tf.nn.relu(x)

        x = self._res_conv_layer(x, layer_name = 'res' + stage + '_branch2b')
        x = self._res_batch_norm_layer(x, layer_name = 'bn' + stage + '_branch2b')
        x = tf.nn.relu(x)

        x = self._res_conv_layer(x, layer_name = 'res' + stage + '_branch2c')
        x = self._res_batch_norm_layer(x, layer_name = 'bn' + stage + '_branch2c')

        x = tf.add(x, input_layer, name = 'add' + stage)
        x = tf.nn.relu(x, name = 'relu' + stage)

        return x
