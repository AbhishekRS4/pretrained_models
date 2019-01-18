# @author : Abhishek R S

import os
import numpy as np
import h5py
import tensorflow as tf

'''

Inception_v3

# Reference
- [Rethinking the Inception Architecture for Computer Vision]
  (http://arxiv.org/abs/1512.00567)

# Pretrained model weights
- [Download pretrained Inception_v3 model]
  (https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5)

'''

class InceptionV3:

    # initialize network parameters
    def __init__(self, data_format = 'channels_first', training = True, inception_path = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'):
        self._weights_h5 = h5py.File(inception_path, 'r')
        self._data_format = data_format
        self._training = training
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

    # build inception_v3 encoder
    def inception_v3_encoder(self, features):

        # input : RGB format in range [-1, 1]
        # input = input / 127.5
        # input = input - 1.

        # Stage 1
        self.stage1 = self._conv_block(features, strides = self._pool_strides, name = 'block1_conv1')
        self.stage1 = self._conv_block(self.stage1, name = 'block1_conv2')
        self.stage1 = self._conv_block(self.stage1, name = 'block1_conv3')
        self.pool1 = self._maxpool_layer(self.stage1, name = 'block1_maxpool')

        # Stage 2
        self.stage2 = self._conv_block(self.pool1, name = 'block2_conv1')
        self.stage2 = self._conv_block(self.stage2, name = 'block2_conv2')
        self.pool2 = self._maxpool_layer(self.stage2, name = 'block2_maxpool')

        # Stage 3
        self.stage3 = self._inception_block_a(self.pool2, 'block_a_mixed1_')
        self.stage3 = self._inception_block_a(self.stage3, 'block_a_mixed2_')
        self.stage3 = self._inception_block_a(self.stage3, 'block_a_mixed3_')

        # stage 4
        self.stage4 = self._inception_block_b(self.stage3, 'block_b_mixed1_')

        # Stage 5
        self.stage5 = self._inception_block_c(self.stage4, 'block_c_mixed1_')
        self.stage5 = self._inception_block_c(self.stage5, 'block_c_mixed2_')
        self.stage5 = self._inception_block_c(self.stage5, 'block_c_mixed3_')
        self.stage5 = self._inception_block_c(self.stage5, 'block_c_mixed4_')

        # Stage 6
        self.stage6 = self._inception_block_d(self.stage5, 'block_d_mixed1_')
        
        # Stage 7 
        self.stage7 = self._inception_block_e(self.stage6, 'block_e_mixed1_')
        self.stage7 = self._inception_block_e(self.stage7, 'block_e_mixed2_')

    #-------------------------------------------#
    # pretrained inception_v3 encoder functions #
    #-------------------------------------------#
    def _inception_block_a(self, input_layer, name):
        branch1x1 = self._conv_block(input_layer, name = name + 'branch1x1')

        branch5x5 = self._conv_block(input_layer, name = name + 'branch5x5_1')
        branch5x5 = self._conv_block(branch5x5, name = name + 'branch5x5_2')

        branch3x3 = self._conv_block(input_layer, name = name + 'branch3x3_1')
        branch3x3 = self._conv_block(branch3x3, name = name + 'branch3x3_2')
        branch3x3 = self._conv_block(branch3x3, name = name + 'branch3x3_3')

        branch_pool = self._avgpool_layer(input_layer, name = name + 'branch_avgpool') 
        branch_pool = self._conv_block(branch_pool, name = name + 'branch_avgpool1x1')

        out = tf.concat([branch1x1, branch5x5, branch3x3, branch_pool], axis = self._feature_map_axis, name = name + 'concat')

        return out

    def _inception_block_b(self, input_layer, name):
        branch3x3 = self._conv_block(input_layer, strides = self._pool_strides, name = name + 'branch3x3_1')

        branch3x3dbl = self._conv_block(input_layer, name = name + 'branch3x3_2')
        branch3x3dbl = self._conv_block(branch3x3dbl, name = name + 'branch3x3_3')
        branch3x3dbl = self._conv_block(branch3x3dbl, strides = self._pool_strides, name = name + 'branch3x3_4')

        branch_pool = self._maxpool_layer(input_layer, name = name + 'maxpool')

        out = tf.concat([branch3x3, branch3x3dbl, branch_pool], axis = self._feature_map_axis, name = name + 'concat')

        return out

    def _inception_block_c(self, input_layer, name):
        branch1x1 = self._conv_block(input_layer, name = name + 'branch1x1')

        branch7x7 = self._conv_block(input_layer, name = name + 'branch7x7_1')
        branch7x7 = self._conv_block(branch7x7, name = name + 'branch7x7_2')
        branch7x7 = self._conv_block(branch7x7, name = name + 'branch7x7_3')

        branch7x7dbl = self._conv_block(input_layer, name = name + 'branch7x7_4')
        branch7x7dbl = self._conv_block(branch7x7dbl, name = name + 'branch7x7_5')
        branch7x7dbl = self._conv_block(branch7x7dbl, name = name + 'branch7x7_6')
        branch7x7dbl = self._conv_block(branch7x7dbl, name = name + 'branch7x7_7')
        branch7x7dbl = self._conv_block(branch7x7dbl, name = name + 'branch7x7_8')

        branch_pool = self._avgpool_layer(input_layer, name = name + 'branch_avgpool') 
        branch_pool = self._conv_block(branch_pool, name = name + 'branch_avgpool1x1')

        out = tf.concat([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis = self._feature_map_axis, name = name + 'concat')

        return out

    def _inception_block_d(self, input_layer, name):
        branch3x3 = self._conv_block(input_layer, name = name + 'branch3x3_1')
        branch3x3 = self._conv_block(branch3x3, strides = self._pool_strides, name = name + 'branch3x3_2')

        branch7x7 = self._conv_block(input_layer, name = name + 'branch7x7_1')
        branch7x7 = self._conv_block(branch7x7, name = name + 'branch7x7_2')
        branch7x7 = self._conv_block(branch7x7, name = name + 'branch7x7_3')
        branch7x7 = self._conv_block(branch7x7, strides = self._pool_strides, name = name + 'branch7x7_4')

        branch_pool = self._maxpool_layer(input_layer, name = name + 'maxpool')

        out = tf.concat([branch3x3, branch7x7, branch_pool], axis = self._feature_map_axis, name = name + 'concat')

        return out

    def _inception_block_e(self, input_layer, name):
        branch1x1 = self._conv_block(input_layer, name = name + 'branch1x1')

        branch3x3_1 = self._conv_block(input_layer, name = name + 'branch3x3_1')
        branch3x3_2 = self._conv_block(branch3x3_1, name = name + 'branch3x3_2')
        branch3x3_3 = self._conv_block(branch3x3_1, name = name + 'branch3x3_3')
        branch3x3_concat1 = tf.concat([branch3x3_2, branch3x3_3], axis = self._feature_map_axis, name = name + 'concat1')

        branch3x3_4 = self._conv_block(input_layer, name = name + 'branch3x3_4')
        branch3x3_5 = self._conv_block(branch3x3_4, name = name + 'branch3x3_5')
        branch3x3_6 = self._conv_block(branch3x3_5, name = name + 'branch3x3_6')
        branch3x3_7 = self._conv_block(branch3x3_5, name = name + 'branch3x3_7')
        branch3x3_concat2 = tf.concat([branch3x3_6, branch3x3_7], axis = self._feature_map_axis, name = name + 'concat2')

        branch_pool = self._avgpool_layer(input_layer, name = name + 'branch_avgpool') 
        branch_pool = self._conv_block(branch_pool, name = name + 'branch_avgpool1x1')

        out = tf.concat([branch1x1, branch3x3_concat1, branch3x3_concat2, branch_pool], axis = self._feature_map_axis, name = name + 'concat3')

        return out

    #-----------------------#
    # convolution block     #
    #-----------------------#
    def _conv_block(self, input_layer, strides = None, padding = None, name = 'conv_block'):
        if strides is None:
            strides = self._encoder_conv_strides

        if padding is None:
            padding = self._padding

        conv = self._conv_layer(input_layer, strides, padding, name = name + '_conv') 
        bn = self._batchnorm_layer(conv, name = name + '_bn')
        relu = tf.nn.relu(bn, name = name + '_relu')

        return relu

    #-----------------------#
    # convolution2d layer   #
    #-----------------------#
    def _conv_layer(self, input_layer, strides, padding, name, key = 'conv2d_'):
        W_init_value = np.array(self._weights_h5[key + str(self._counter)][key + str(self._counter)]['kernel:0']).astype(np.float32)
        W = tf.get_variable(name = name + '_W', shape = W_init_value.shape, initializer = tf.constant_initializer(W_init_value), dtype = tf.float32)
        x = tf.nn.conv2d(input_layer, filter = W, strides = strides, padding = padding, data_format = self._encoder_data_format, name = name)

        self._counter += 1

        return x

    #-----------------------#
    # avgpool layer         #
    #-----------------------#
    def _avgpool_layer(self, input_layer, pool_size = [3, 3], strides = [1, 1], name = 'avg_pool'):
        return tf.layers.average_pooling2d(input_layer, pool_size = pool_size, strides = strides, padding = self._padding, data_format = self._data_format, name = name)

    #-----------------------#
    # maxpool2d layer       #
    #-----------------------#
    def _maxpool_layer(self, input_layer, name):
        pool = tf.nn.max_pool(input_layer, ksize = self._pool_kernel, strides = self._pool_strides, padding = self._padding, data_format = self._encoder_data_format, name = name)

        return pool

    #-----------------------#
    # batchnorm layer       #
    #-----------------------#
    def _batchnorm_layer(self, input_layer, name):
        return tf.layers.batch_normalization(input_layer, axis = self._feature_map_axis, training = self._training, name = name)
