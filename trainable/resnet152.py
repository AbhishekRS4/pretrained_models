# @author : Abhishek R S

import os
import numpy as np
import h5py
import tensorflow as tf

'''

ResNet-152

# Reference
- [Deep Residual Learning for Image Recognition](
   https://arxiv.org/abs/1512.03385)

# Pretrained model weights
- [Download pretrained resnet-152 model]
  (https://drive.google.com/file/d/0Byy2AcGyEVxfeXExMzNNOHpEODg/view)

'''

class ResNet152:

    # initialize network parameters
    def __init__(self, data_format = 'channels_first', training = True, resnet_path = 'resnet152_weights_tf.h5'):
        self._training = training
        self._weights_h5 = h5py.File(resnet_path, 'r')
        self._data_format = data_format
        self._encoder_data_format = None
        self._pool_kernel = None
        self._pool_strides = None
        self._feature_map_axis = None
        self._res_conv_strides = [1, 1, 1, 1]

        if self._data_format == 'channels_first':
            self._feature_map_axis = 1
            self._encoder_data_format = 'NCHW'
            self._pool_kernel = [1, 1, 3, 3]
            self._pool_strides = [1, 1, 2, 2]
        else: 
            self._feature_map_axis = -1
            self._encoder_data_format = 'NHWC'
            self._pool_kernel = [1, 3, 3, 1]
            self._pool_strides = [1, 2, 2, 1]

    # build resnet-152 encoder
    def resnet152_encoder(self, features):

        # input : BGR format with image_net mean subtracted
        # bgr mean : [103.939, 116.779, 123.68]

        # Stage 0
        self.stage0 = self._conv_layer(features, 'conv1', strides = self._pool_strides)
        self.stage0 = self._batchnorm_layer(self.stage0, 'bn_conv1')
        self.stage0 = tf.nn.relu(self.stage0, name = 'relu1')
        
        # Stage 1
        self.stage1 = tf.nn.max_pool(self.stage0, ksize = self._pool_kernel, strides = self._pool_strides, padding = 'SAME', data_format = self._encoder_data_format, name = 'pool1')

        # Stage 2
        self.stage2 = self._res_conv_block(input_layer = self.stage1, stage = '2a', strides = self._res_conv_strides)
        self.stage2 = self._res_identity_block(input_layer = self.stage2, stage = '2b')
        self.stage2 = self._res_identity_block(input_layer = self.stage2, stage = '2c')

        # Stage 3
        self.stage3 = self._res_conv_block(input_layer = self.stage2, stage = '3a', strides = self._pool_strides)
        for i in range(1, 8):
            self.stage3 = self._res_identity_block(input_layer = self.stage3, stage = '3b' + str(i))

        # Stage 4
        self.stage4 = self._res_conv_block(input_layer = self.stage3, stage = '4a', strides = self._pool_strides)
        for i in range(1, 36):
            self.stage4 = self._res_identity_block(input_layer = self.stage4, stage = '4b' + str(i))

        # Stage 5
        self.stage5 = self._res_conv_block(input_layer = self.stage4, stage = '5a', strides = self._pool_strides)
        self.stage5 = self._res_identity_block(input_layer = self.stage5, stage = '5b')
        self.stage5 = self._res_identity_block(input_layer = self.stage5, stage = '5c')

    #-----------------------#
    # batchnorm layer       #
    #-----------------------#
    def _batchnorm_layer(self, input_layer, name = 'bn'):
        return tf.layers.batch_normalization(input_layer, axis = self._feature_map_axis, training = self._training, name = name)

    #-------------------------------------#
    # pretrained resnet encoder functions #
    #-------------------------------------#
    #-----------------------#
    # convolution2d layer   #
    #-----------------------#
    def _conv_layer(self, input_layer, name, strides = [1, 1, 1, 1], padding = 'SAME'):
        W_init_value = np.array(self._weights_h5[name][name + '_W_1:0'], dtype = np.float32)
        W = tf.get_variable(name = name + '_kernel', shape = W_init_value.shape, initializer = tf.constant_initializer(W_init_value), dtype = tf.float32)
        x = tf.nn.conv2d(input_layer, filter = W, strides = strides, padding = padding, data_format = self._encoder_data_format, name = name + 'conv')

        return x

    #-----------------------#
    # convolution block     #
    #-----------------------#
    def _res_conv_block(self, input_layer, stage, strides):
        x = self._conv_layer(input_layer, name = 'res' + stage + '_branch2a', strides = strides)
        x = self._batchnorm_layer(x, name = 'bn' + stage + '_branch2a')
        x = tf.nn.relu(x, name = 'relu' + stage + '_branch2a')

        x = self._conv_layer(x, name = 'res' + stage + '_branch2b')
        x = self._batchnorm_layer(x, name = 'bn' + stage + '_branch2b')
        x = tf.nn.relu(x, name = 'relu' + stage + '_branch2b')

        x = self._conv_layer(x, name = 'res' + stage + '_branch2c')
        x = self._batchnorm_layer(x, name = 'bn' + stage + '_branch2c')

        shortcut = self._conv_layer(input_layer, name = 'res' + stage + '_branch1', strides = strides)
        shortcut = self._batchnorm_layer(shortcut, name = 'bn' + stage + '_branch1')

        x = tf.add(x, shortcut, name = 'add' + stage)
        x = tf.nn.relu(x, name = 'relu' + stage)

        return x

    #-----------------------#
    # identity block        #
    #-----------------------#
    def _res_identity_block(self, input_layer, stage):
        x = self._conv_layer(input_layer, name = 'res' + stage + '_branch2a')
        x = self._batchnorm_layer(x, name = 'bn' + stage + '_branch2a')
        x = tf.nn.relu(x, name = 'relu' + stage + '_branch2a')

        x = self._conv_layer(x, name = 'res' + stage + '_branch2b')
        x = self._batchnorm_layer(x, name = 'bn' + stage + '_branch2b')
        x = tf.nn.relu(x, name = 'relu' + stage + '_branch2b')

        x = self._conv_layer(x, name = 'res' + stage + '_branch2c')
        x = self._batchnorm_layer(x, name = 'bn' + stage + '_branch2c')

        x = tf.add(x, input_layer, name = 'add' + stage)
        x = tf.nn.relu(x, name = 'relu' + stage)

        return x
