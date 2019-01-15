import os
import numpy as np
import tensorflow as tf

'''
#Parse functions for tf dataset api

* Use parse_fn_res34 for resnet-34
* Use parse_fn_caffe for vgg, resnets except for resnet-34
* Use parse_fn_torch for densenet
* Use parse_fn_tf for xception, inception_v3, inception_resnet_v2
* The resulting images will be in NCHW format
'''

imagenet_caffe_mean = np.array(
    [103.939, 116.779, 123.68], dtype=np.float32).reshape(1, 3)

imagenet_torch_mean = np.array(
    [0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3)
imagenet_torch_std = np.array(
    [0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3)


def parse_fn_res34(img_name, lbl):
    # read
    img_string = tf.read_file(img_name)

    # decode
    img = tf.image.decode_jpeg(img_string, channels=3)

    # change image to float32 format with range [0, 255]
    img = tf.cast(img, tf.float32)

    # CHW format
    img = tf.transpose(img, perm=[2, 0, 1])

    # change image to int32 format
    lbl = tf.cast(lbl, tf.int32)

    return img, lbl


def parse_fn_caffe(img_name, lbl):
    # read
    img_string = tf.read_file(img_name)

    # decode
    img = tf.image.decode_jpeg(img_string, channels=3)

    # change image to BGR and subtract imagenet mean, as float32 format
    img_r, img_g, img_b = tf.split(value=img, axis=2, num_or_size_splits=3)
    img = tf.concat(values=[img_b, img_g, img_r], axis=2)
    img = img - imagenet_caffe_mean

    # CHW format
    img = tf.transpose(img, perm=[2, 0, 1])

    # change image to int32 format
    lbl = tf.cast(lbl, tf.int32)

    return img, lbl


def parse_fn_torch(img_name, lbl):
    # read
    img_string = tf.read_file(img_name)

    # decode
    img = tf.image.decode_jpeg(img_string, channels=3)

    # change image to range [0, 1] and normalize imagenet mean and std, as float32 format
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img - imagenet_torch_mean
    img = img / imagenet_torch_std

    # CHW format
    img = tf.transpose(img, perm=[2, 0, 1])

    # change image to int32 format
    lbl = tf.cast(lbl, tf.int32)

    return img, lbl


def parse_fn_tf(img_name, lbl):
    # read
    img_string = tf.read_file(img_name)

    # decode
    img = tf.image.decode_jpeg(img_string, channels=3)

    # change image range to [-1, 1] as float32 format
    img = tf.cast(img, tf.float32)
    img = img / 127.5
    img = img - 1.

    # CHW format
    img = tf.transpose(img, perm=[2, 0, 1])

    # change image to int32 format
    lbl = tf.cast(lbl, tf.int32)

    return img, lbl


'''
* creates tf dataset with input list of images and labels
* while running inference remove shuffle, set both num_epochs and batch_size to 1
'''
def create_tf_dataset(images_list, labels_list, num_epochs, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((images_list, labels_list))
    dataset = dataset.shuffle(1000)
    dataset = dataset.map(parse_fn_tf, num_parallel_calls=8)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(batch_size)

    return dataset
