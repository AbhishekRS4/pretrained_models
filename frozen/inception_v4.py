# @author : Abhishek R S

import os
import numpy as np
import h5py
import tensorflow as tf

'''

Inception_v4, Inception_Resnet_v2

# Reference
- [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning]
  (https://arxiv.org/abs/1602.07261) 

# Pretrained model weights
- [Download pretrained Inception_Resnet_v2 model]
  (https://github.com/fchollet/deep-learning-models/releases/download/v0.7/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5)

'''

class InceptionV4:

    # initialize network parameters
    def __init__(self, data_format = 'channels_first', inception_path = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'):
        self._weights_h5 = h5py.File(inception_path, 'r')
        self._data_format = data_format
        self._encoder_data_format = None
        self._pool_kernel = None
        self._pool_strides = None
        self._feature_map_axis = None
        self._encoder_conv_strides = [1, 1, 1, 1]
        self._padding = 'SAME'
        self._counter = 1

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

    # build inception_v4 encoder
    def inception_v4_encoder(self, features):

        # input : RGB format in range [-1, 1]
        # input = input / 127.5
        # input = input - 1.

        # Stage 1
        self.stage1 = self._conv_block(features, strides = self._pool_strides, name = 'Conv2d_1a_3x3')
        self.stage1 = self._conv_block(self.stage1, name = 'Conv2d_2a_3x3')
        self.stage1 = self._conv_block(self.stage1, name = 'Conv2d_2b_3x3')
        self.pool1 = self._maxpool_layer(self.stage1, name = 'Block1_Maxpool')

        # Stage 2
        self.stage2 = self._conv_block(self.pool1, name = 'Conv2d_3b_1x1')
        self.stage2 = self._conv_block(self.stage2, name = 'Conv2d_4a_3x3')
        self.pool2 = self._maxpool_layer(self.stage2, name = 'Block2_Maxpool')

        # Stage 3
        # Mixed 5b
        self.stage3_branch0 = self._conv_block(self.pool2, name = 'Mixed_5b_Branch_0_Conv2d_1x1')
        self.stage3_branch1 = self._conv_block(self.pool2, name = 'Mixed_5b_Branch_1_Conv2d_0a_1x1')
        self.stage3_branch1 = self._conv_block(self.stage3_branch1, name = 'Mixed_5b_Branch_1_Conv2d_0b_5x5')
        self.stage3_branch2 = self._conv_block(self.pool2, name = 'Mixed_5b_Branch_2_Conv2d_0a_1x1')
        self.stage3_branch2 = self._conv_block(self.stage3_branch2, name = 'Mixed_5b_Branch_2_Conv2d_0b_3x3')
        self.stage3_branch2 = self._conv_block(self.stage3_branch2, name = 'Mixed_5b_Branch_2_Conv2d_0c_3x3')
        self.stage3_branch_pool = self._avgpool_layer(self.pool2, name = 'Mixed_5b_Avgpool') 
        self.stage3_branch_pool = self._conv_block(self.stage3_branch_pool, name = 'Mixed_5b_Branch_3_Conv2d_0b_1x1')
        self.stage3_x = tf.concat([self.stage3_branch0, self.stage3_branch1, self.stage3_branch2, self.stage3_branch_pool], axis = self._feature_map_axis, name = 'Mixed_5b_Concat')
        
        for i in range(1, 11):
            self.stage3_x = self._inception_block_a(self.stage3_x, scale = 0.17, name = 'Block35_' + str(i) + '_')

        # stage 4
        # Mixed 6a
        self.stage4_branch0 = self._conv_block(self.stage3_x, name = 'Mixed_6a_Branch_0_Conv2d_1a_3x3', strides = self._pool_strides)
        self.stage4_branch1 = self._conv_block(self.stage3_x, name = 'Mixed_6a_Branch_1_Conv2d_0a_1x1')
        self.stage4_branch1 = self._conv_block(self.stage4_branch1, name = 'Mixed_6a_Branch_1_Conv2d_0b_3x3')
        self.stage4_branch1 = self._conv_block(self.stage4_branch1, name = 'Mixed_6a_Branch_1_Conv2d_1a_3x3', strides = self._pool_strides)
        self.stage4_branch_pool = self._maxpool_layer(self.stage3_x, name = 'Mixed_6a_Maxpool')
        self.stage4_x = tf.concat([self.stage4_branch0, self.stage4_branch1, self.stage4_branch_pool], axis = self._feature_map_axis, name = 'Mixed_6a_Concat')

        for i in range(1, 21):
            self.stage4_x = self._inception_block_b(self.stage4_x, scale = 0.1, name = 'Block17_' + str(i) + '_')

        # Stage 5
        # Mixed 7a
        self.stage5_branch0 = self._conv_block(self.stage4_x, name = 'Mixed_7a_Branch_0_Conv2d_0a_1x1')
        self.stage5_branch0 = self._conv_block(self.stage5_branch0, name = 'Mixed_7a_Branch_0_Conv2d_1a_3x3', strides = self._pool_strides)
        self.stage5_branch1 = self._conv_block(self.stage4_x, name = 'Mixed_7a_Branch_1_Conv2d_0a_1x1')
        self.stage5_branch1 = self._conv_block(self.stage5_branch1, name = 'Mixed_7a_Branch_1_Conv2d_1a_3x3', strides = self._pool_strides)
        self.stage5_branch2 = self._conv_block(self.stage4_x, name = 'Mixed_7a_Branch_2_Conv2d_0a_1x1')
        self.stage5_branch2 = self._conv_block(self.stage5_branch2, name = 'Mixed_7a_Branch_2_Conv2d_0b_3x3')
        self.stage5_branch2 = self._conv_block(self.stage5_branch2, name = 'Mixed_7a_Branch_2_Conv2d_1a_3x3', strides = self._pool_strides)
        self.stage5_branch_pool = self._maxpool_layer(self.stage4_x, name = 'Mixed_7a_Maxpool')
        self.stage5_x = tf.concat([self.stage5_branch0, self.stage5_branch1, self.stage5_branch2, self.stage5_branch_pool], axis = self._feature_map_axis, name = 'Mixed_7a_Concat')

        for i in range(1, 10):
            self.stage5_x = self._inception_block_c(self.stage5_x, scale = 0.2, name = 'Block8_' + str(i) + '_')

        self.stage5_x = self._inception_block_c(self.stage5_x, scale = 1., activation = False, name = 'Block8_' + '10_')

        # Stage 6
        self.stage6 = self._conv_block(self.stage5_x, name = 'Conv2d_7b_1x1')

    #--------------------------------------------------#
    # pretrained inception_resnet_v2 encoder functions #
    #--------------------------------------------------#
    def _inception_block_a(self, x, scale, name):
        branch0 = self._conv_block(x, name = name + 'Branch_0_Conv2d_1x1')

        branch1 = self._conv_block(x, name = name + 'Branch_1_Conv2d_0a_1x1')
        branch1 = self._conv_block(branch1, name = name + 'Branch_1_Conv2d_0b_3x3')

        branch2 = self._conv_block(x, name = name + 'Branch_2_Conv2d_0a_1x1')
        branch2 = self._conv_block(branch2, name = name + 'Branch_2_Conv2d_0b_3x3')
        branch2 = self._conv_block(branch2, name = name + 'Branch_2_Conv2d_0c_3x3')

        concat = tf.concat([branch0, branch1, branch2], axis = self._feature_map_axis, name = name + 'Concat')

        # still not correct
        up = self._conv_block(concat, batchnorm = False, activation = False, name = name + 'Conv2d_1x1')
        x = x + up * scale
        x = tf.nn.relu(x, name = name + 'Relu')

        return x

    def _inception_block_b(self, x, scale, name):
        branch0 = self._conv_block(x, name = name + 'Branch_0_Conv2d_1x1')

        branch1 = self._conv_block(x, name = name + 'Branch_1_Conv2d_0a_1x1')
        branch1 = self._conv_block(branch1, name = name + 'Branch_1_Conv2d_0b_1x7')
        branch1 = self._conv_block(branch1, name = name + 'Branch_1_Conv2d_0c_7x1')

        concat = tf.concat([branch0, branch1], axis = self._feature_map_axis, name = name + 'Concat')

        # still not correct
        up = self._conv_block(concat, batchnorm = False, activation = False, name = name + 'Conv2d_1x1')
        x = x + up * scale
        x = tf.nn.relu(x, name = name + 'Relu')

        return x

    def _inception_block_c(self, x, scale, activation = True, name):
        branch0 = self._conv_block(x, name = name + 'Branch_0_Conv2d_1x1')

        branch1 = self._conv_block(x, name = name + 'Branch_1_Conv2d_0a_1x1')
        branch1 = self._conv_block(branch1, name = name + 'Branch_1_Conv2d_0b_1x3')
        branch1 = self._conv_block(branch1, name = name + 'Branch_1_Conv2d_0c_3x1')

        concat = tf.concat([branch0, branch1], axis = self._feature_map_axis, name = name + 'Concat')

        # still not correct
        up = self._conv_block(concat, batchnorm = False, activation = False, name = name + 'Conv2d_1x1')
        x = x + up * scale

        if activation:
            x = tf.nn.relu(x, name = name + 'Relu')

        return x

    #-----------------------#
    # convolution block     #
    #-----------------------#
    def _conv_block(self, x, name, strides = None, padding = None, batchnorm = True, activation = True):
        use_bias = not(batchnorm)
        if strides is None:
            strides = self._encoder_conv_strides

        if padding is None:
            padding = self._padding

        x = self._conv_layer(x, strides = strides, padding = padding, use_bias = use_bias, name = name) 

        if batchnorm:
            x = self._batchnorm_layer(x, name = name + '_BatchNorm')

        if activation:
            x = tf.nn.relu(x, name = name + '_Relu')

        return x

    #-----------------------#
    # convolution2d layer   #
    #-----------------------#
    def _conv_layer(self, input_layer, strides, padding, use_bias, name):
        W = tf.constant(self._weights_h5[name][name]['kernel:0'])
        x = tf.nn.conv2d(input_layer, filter = W, strides = strides, padding = padding, data_format = self._encoder_data_format, name = name + '_Conv')

        if use_bias:
            b = self._weights_h5[name][name]['bias:0']
            b = tf.constant(np.reshape(b, b.shape[0]))
            x = tf.nn.bias_add(x, b, data_format = self._encoder_data_format, name = name + '_Bias')

        return x

    #-----------------------#
    # batchnorm layer       #
    #-----------------------#
    def _batchnorm_layer(self, input_layer, name):
        if self._encoder_data_format == 'NCHW':
            input_layer = tf.transpose(input_layer, perm = [0, 2, 3, 1])

        gamma_shape = self._weights_h5[name][name]['beta:0'].shape  

        mean = tf.constant(self._weights_h5[name][name]['moving_mean:0'])
        std = tf.constant(self._weights_h5[name][name]['moving_variance:0'])
        beta = tf.constant(self._weights_h5[name][name]['beta:0'])
        gamma = tf.constant(np.ones(shape = gamma_shape), dtype = np.float32)

        bn = tf.nn.batch_normalization(input_layer, mean = mean, variance = std, offset = beta, scale = gamma, variance_epsilon = 1e-12, name = name)

        if self._encoder_data_format == 'NCHW':
            bn = tf.transpose(bn, perm = [0, 3, 1, 2])

        return bn

    #-----------------------#
    # avgpool layer         #
    #-----------------------#
    def _avgpool_layer(self, input_layer, name, pool_size = [3, 3], strides = [1, 1]):
        return tf.layers.average_pooling2d(input_layer, pool_size = pool_size, strides = strides, padding = self._padding, data_format = self._data_format, name = name)

    #-----------------------#
    # maxpool2d layer       #
    #-----------------------#
    def _maxpool_layer(self, input_layer, name):
        pool = tf.nn.max_pool(input_layer, ksize = self._pool_kernel, strides = self._pool_strides, padding = self._padding, data_format = self._encoder_data_format, name = name)

        return pool
