# @author : Abhishek R S

import os
import numpy as np
import h5py
import tensorflow as tf

'''

VGG-19

# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](
   https://arxiv.org/abs/1409.1556)

# Pretrained model weights
- [Download pretrained vgg-19 model]
  (https://github.com/fchollet/deep-learning-models/releases/)

'''

class VGG19:

    # initialize network parameters
    def __init__(self, data_format = 'channels_first', training = True, vgg_path = 'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'):
        self._training = training
        self._weights_h5 = h5py.File(vgg_path, 'r')
        self._data_format = data_format
        self._encoder_data_format = None
        self._pool_kernel = None
        self._pool_strides = None
        self._encoder_conv_strides = [1, 1, 1, 1]
        self._padding = 'SAME'

        if self._data_format == 'channels_first':
            self._encoder_data_format = 'NCHW'
            self._pool_kernel = [1, 1, 2, 2]
            self._pool_strides = [1, 1, 2, 2]
        else: 
            self._encoder_data_format = 'NHWC'
            self._pool_kernel = [1, 2, 2, 1]
            self._pool_strides = [1, 2, 2, 1]

    # build vgg-19 encoder
    def vgg19_encoder(self, features):

        # input : BGR format with image_net mean subtracted
        # bgr mean : [103.939, 116.779, 123.68]

        # Stage 1
        self.conv1_1 = self._conv_block(features, 'block1_conv1')
        self.conv1_2 = self._conv_block(self.conv1_1, 'block1_conv2')
        self.pool1 = self._maxpool_layer(self.conv1_2, name = 'pool1')

        # Stage 2
        self.conv2_1 = self._conv_block(self.pool1, 'block2_conv1')
        self.conv2_2 = self._conv_block(self.conv2_1, 'block2_conv2')
        self.pool2 = self._maxpool_layer(self.conv2_2, name = 'pool2')

        # Stage 3
        self.conv3_1 = self._conv_block(self.pool2, 'block3_conv1')
        self.conv3_2 = self._conv_block(self.conv3_1, 'block3_conv2')
        self.conv3_3 = self._conv_block(self.conv3_2, 'block3_conv3')
        self.conv3_4 = self._conv_block(self.conv3_3, 'block3_conv4')
        self.pool3 = self._maxpool_layer(self.conv3_4, name = 'pool3')

        # Stage 4
        self.conv4_1 = self._conv_block(self.pool3, 'block4_conv1')
        self.conv4_2 = self._conv_block(self.conv4_1, 'block4_conv2')
        self.conv4_3 = self._conv_block(self.conv4_2, 'block4_conv3')
        self.conv4_4 = self._conv_block(self.conv4_3, 'block4_conv4')
        self.pool4 = self._maxpool_layer(self.conv4_4, name = 'pool4')

        # Stage 5
        self.conv5_1 = self._conv_block(self.pool4, 'block5_conv1')
        self.conv5_2 = self._conv_block(self.conv5_1, 'block5_conv2')
        self.conv5_3 = self._conv_block(self.conv5_2, 'block5_conv3')
        self.conv5_4 = self._conv_block(self.conv5_3, 'block5_conv4')
        self.pool5 = self._maxpool_layer(self.conv5_4, name = 'pool5')

    #-------------------------------------#
    # pretrained vgg-19 encoder functions #
    #-------------------------------------#
    #-----------------------#
    # convolution2d layer   #
    #-----------------------#
    def _conv_block(self, input_layer, name):
        W_init_value = np.array(self._weights_h5[name][name + '_W_1:0'], dtype = np.float32)
        b_init_value = np.array(self._weights_h5[name][name + '_b_1:0'], dtype = np.float32)

        W = tf.get_variable(name = name + '_kernel', shape = W_init_value.shape, initializer = tf.constant_initializer(W_init_value), dtype = tf.float32)
        b = tf.get_variable(name = name + '_bias', shape = b_init_value.shape, initializer = tf.constant_initializer(b_init_value), dtype = tf.float32)

        x = tf.nn.conv2d(input_layer, filter = W, strides = self._encoder_conv_strides, padding = self._padding, data_format = self._encoder_data_format, name = name + '_conv')
        x = tf.nn.bias_add(x, b, data_format = self._encoder_data_format, name = name + '_bias')
        x = tf.nn.relu(x, name = name + '_relu') 

        return x

    #-----------------------#
    # maxpool2d layer       #
    #-----------------------#
    def _maxpool_layer(self, input_layer, name):
        pool = tf.nn.max_pool(input_layer, ksize = self._pool_kernel, strides = self._pool_strides, padding = self._padding, data_format = self._encoder_data_format, name = name)

        return pool
