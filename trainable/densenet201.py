# @author : Abhishek R S

import os
import numpy as np
import h5py
import tensorflow as tf

'''

DenseNet-201

# Reference
- [Densely Connected Convolutional Networks]
  (https://arxiv.org/abs/1608.06993)

# Pretrained model weights
- [Download pretrained densenet-201 model]
  (https://github.com/fchollet/deep-learning-models/releases/)

'''

class DenseNet201:

    # initialize network parameters
    def __init__(self, data_format = 'channels_first', training = True, densenet_path = 'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5'):
        self._training = training
        self._weights_h5 = h5py.File(densenet_path, 'r')
        self._data_format = data_format
        self._encoder_data_format = None
        self._pool_kernel = None
        self._pool_strides = None
        self._padding = 'SAME'
        self._conv_strides = [1, 1, 1, 1]
        self._blocks = [6, 12, 48, 32]
        self._growth_rate = 32
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

    # build densenet-201 encoder
    def densenet201_encoder(self, features):

        # input : RGB format normalized with mean and std
        # x = x / 255.
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        # x = (x - mean) / std

        # Stage 0
        self.stage0 = self._conv_layer(features, 'conv1_conv', strides = self._pool_strides)
        self.stage0 = self._batchnorm_layer(self.stage0, 'conv1_bn')
        self.stage0 = tf.nn.relu(self.stage0, name = 'conv1_relu')
        
        # Stage 1
        self.stage1 = tf.nn.max_pool(self.stage0, ksize = self._pool_kernel, strides = self._pool_strides, padding = self._padding, data_format = self._encoder_data_format, name = 'pool1')

        # Stage 2
        self.stage2_dense = self._dense_block(self.stage1, self._blocks[0], name = 'conv2') 
        self.stage2_transition = self._transition_block(self.stage2_dense, self._reduction_rate, name = 'pool2')

        # Stage 3
        self.stage3_dense = self._dense_block(self.stage2_transition, self._blocks[1], name = 'conv3') 
        self.stage3_transition = self._transition_block(self.stage3_dense, self._reduction_rate, name = 'pool3')

        # Stage 4
        self.stage4_dense = self._dense_block(self.stage3_transition, self._blocks[2], name = 'conv4') 
        self.stage4_transition = self._transition_block(self.stage4_dense, self._reduction_rate, name = 'pool4')

        # Stage 5
        self.stage5_dense = self._dense_block(self.stage4_transition, self._blocks[3], name = 'conv5') 

    #-----------------------#
    # batchnorm layer       #
    #-----------------------#
    def _batchnorm_layer(self, input_layer, name = 'bn'):
        return tf.layers.batch_normalization(input_layer, axis = self._feature_map_axis, training = self._training, name = name)

    #---------------------------------------#
    # pretrained densenet encoder functions #
    #---------------------------------------#
    #-----------------------#
    # dense block           #
    #-----------------------#
    def _dense_block(self, input_layer, blocks, name):
        x = input_layer

        for i in range(1, blocks + 1):
            x = self._conv_block(x, self._growth_rate, name = name + '_block' + str(i))

        return x

    #-----------------------#
    # convolution block     #
    #-----------------------#
    def _conv_block(self, input_layer, growth_rate, name):
        x = self._batchnorm_layer(input_layer, name = name + '_0_bn')
        x = tf.nn.relu(x, name = name + '_0_relu')
        x = self._conv_layer(x, name = name + '_1_conv')

        x = self._batchnorm_layer(x, name = name + '_1_bn')
        x = tf.nn.relu(x, name = name + '_1_relu')
        x = self._conv_layer(x, name = name + '_2_conv')

        x = tf.concat([input_layer, x], axis = self._feature_map_axis, name = name + '_concat')

        return x

    #-----------------------#
    # transition block      #
    #-----------------------#
    def _transition_block(self, input_layer, reduction_rate, name):
        x = self._batchnorm_layer(input_layer, name = name + '_bn')
        x = tf.nn.relu(x, name = name + '_relu')
        x = self._conv_layer(x, name = name + '_conv')

        x = self._avgpool_layer(x, [2, 2], [2, 2], name = name + '_pool')
        
        return x

    #-----------------------#
    # avgpool layer         #
    #-----------------------#
    def _avgpool_layer(self, input_layer, pool_size = [2, 2], strides = [2, 2], name = 'avg_pool'): 
        return tf.layers.average_pooling2d(input_layer, pool_size = pool_size, strides = strides, padding = self._padding, data_format = self._data_format, name = name)

    #-----------------------#
    # convolution2d layer   #
    #-----------------------#
    def _conv_layer(self, input_layer, name, strides = [1, 1, 1, 1]):
        weights_key = name.split('_')

        if len(weights_key) == 4:
            last_key = list(self._weights_h5[weights_key[0]][weights_key[1]][weights_key[2]][weights_key[3]][weights_key[0]][weights_key[1]][weights_key[2]])
            weights_hierarchy = self._weights_h5[weights_key[0]][weights_key[1]][weights_key[2]][weights_key[3]][weights_key[0]][weights_key[1]][weights_key[2]][last_key[0]]
        else:
            last_key = list(self._weights_h5[weights_key[0]][weights_key[1]][weights_key[0]])
            weights_hierarchy = self._weights_h5[weights_key[0]][weights_key[1]][weights_key[0]][last_key[0]]

        W_init_value = np.array(weights_hierarchy['kernel:0'], dtype = np.float32)
        W = tf.get_variable(name = name + '_kernel', shape = W_init_value.shape, initializer = tf.constant_initializer(W_init_value), dtype = tf.float32)
        x = tf.nn.conv2d(input_layer, filter = W, strides = strides, padding = self._padding, data_format = self._encoder_data_format, name = name)

        return x
