# Code for pretrained\_models in tensorflow

## Imagenet pretrained models for transfer learning
- [x] VGG-16 - [VGG-16](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5)
- [x] VGG-19 - [VGG-19](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5)
- [ ] Resnet-34 - [ResNet-34]()
- [x] Resnet-50 - [ResNet-50](https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5)
- [x] Resnet-101 - [ResNet-101](https://drive.google.com/file/d/0Byy2AcGyEVxfTmRRVmpGWDczaXM/view)
- [x] Resnet-152 - [ResNet-152](https://drive.google.com/file/d/0Byy2AcGyEVxfeXExMzNNOHpEODg/view)
- [x] Xception - [Xception](https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5)
- [x] Densenet-121 - [DenseNet-121](https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5)
- [x] Densenet-169 - [DenseNet-169](https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5)
- [x] Densenet-201 - [DenseNet-201](https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5)
- [ ] Inception\_v3 - [Inception\_v3]()
- [ ] InceptionResnet\_v2 - [InceptionResNet\_v2]()
- [ ] Mobilenet\_v2 - [MobileNet\_v2]() 
- [ ] Nasnet\_large - [NasNet\_large]() 
- [ ] Nasnet\_mobile - [NasNet\_mobile]()

## Frozen Models
* For convolution and batchnorm layers, pretrained weights are loaded which cannnot be trained

## Trainable Models
* For convolution layers, pretrained weights are loaded
* For batchnorm layers, the parameters are reinitialized by tf
* And both layers are trainable
