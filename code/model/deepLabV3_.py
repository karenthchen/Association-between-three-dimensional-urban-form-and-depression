# -*- coding: utf-8 -*-

""" Deeplabv3+ model for Keras.
This model is based on:
https://github.com/bonlime/keras-deeplab-v3-plus


# Reference
- [Encoder-Decoder with Atrous Separable Convolution
    for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf)
- [Xception: Deep Learning with Depthwise Separable Convolutions]
    (https://arxiv.org/abs/1610.02357)
- [Inverted Residuals and Linear Bottlenecks: Mobile Networks for
    Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from keras.models import Model
from keras import layers
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Add
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import DepthwiseConv2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.engine import Layer
from keras.engine import InputSpec
from keras.engine.topology import get_source_inputs
from keras import backend as K
from keras.applications import imagenet_utils
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
import cbam
from keras.initializers import Constant
from keras import backend as K
import sys
sys.path.insert(0, '/home/qiu/CodeSummary/MTMS4urban/dataPrepare')
import utli
import tensorflow as tf


#######
def customMetrics_l1(y_true, y_pred):
    print('!!!!!!!!!!!!!!!!!!', y_pred.shape)
    return y_pred[0]
def customMetrics_l2(y_true, y_pred):
    return y_pred[1]
def customMetrics_v1(y_true, y_pred):
    return y_pred[2]
def customMetrics_v2(y_true, y_pred):
    return y_pred[3]


# Custom loss layer
class CustomMultiLossLayer(Layer):
    def __init__(self, nb_outputs=2, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        super(CustomMultiLossLayer, self).__init__(**kwargs)

    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        # for i in range(self.nb_outputs):
        #     self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
        #                                       initializer=Constant(0.), trainable=True)]

        self.log_vars += [self.add_weight(name='log_var' + str(0), shape=(1,),
                                          initializer=Constant(0.), trainable=True)]
        self.log_vars += [self.add_weight(name='log_var' + str(1), shape=(1,),
                                          initializer=Constant(0.), trainable=True)]

        super(CustomMultiLossLayer, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        #loss = 0

        # for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
        #     precision = K.exp(-log_var[0])
        #     loss += K.sum(precision * (y_true - y_pred)**2. + log_var[0], -1)

        precision1 = K.exp(-self.log_vars[0][0])
        #print(ys_true[0].shape, ys_pred[0].shape, ys_true[1].shape, ys_pred[1].shape)
        loss1 = K.mean( tf.keras.metrics.binary_crossentropy(ys_true[0], ys_pred[0]) )
        loss = K.sum( precision1*loss1 + self.log_vars[0][0]/2.0)

        precision2 = K.exp(-self.log_vars[1][0])
        loss2 = utli.masked_loss_function(ys_true[1], ys_pred[1])
        loss = loss + K.sum( precision2*loss2 + self.log_vars[1][0]/2.0)

        #print('!!!!!!!!!!!!!!!!!!', loss, loss1, loss2, self.log_vars[0][0], self.log_vars[1][0])

        return loss, loss1, loss2, self.log_vars[0][0], self.log_vars[1][0]#K.mean(loss)

    def call(self, inputs):
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss, loss1, loss2, var1, var2 = self.multi_loss(ys_true, ys_pred)
        #self.add_loss([loss,  loss1, loss2], inputs=inputs)
        self.add_loss(loss)
        # self.add_loss(loss1)
        # self.add_loss(loss2)
        # We won't actually use the output.

        out = tf.convert_to_tensor(tf.stack([loss1, loss2, var1, var2], -1))
        print(out)
        return out##tf.stack([loss1, loss2, var1, var2], -1)# ys_true[0]#K.concatenate(inputs, -1)

    def compute_output_shape(self, inputs):
        return (4, )

class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = K.image_data_format()
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(
                output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(
                upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return K.tf.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                       inputs.shape[2] * self.upsampling[1]),
                                              align_corners=True)
        else:
            return K.tf.image.resize_bilinear(inputs, (self.output_size[0],
                                                       self.output_size[1]),
                                              align_corners=True)

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation('relu')(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x


def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    """Implements right 'same' padding for even kernel sizes
        Without this there is a 1 pixel drift when stride = 2
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
    """
    if stride == 1:
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)


def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                    rate=1, depth_activation=False, return_skip=False):
    """ Basic building block of modified Xception network
        Args:
            inputs: input tensor
            depth_list: number of filters in each SepConv layer. len(depth_list) == 3
            prefix: prefix before name
            skip_connection_type: one of {'conv','sum','none'}
            stride: stride at last depthwise conv
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & pointwise convs
            return_skip: flag to return additional tensor after 2 SepConvs for decoder
            """
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              prefix + '_separable_conv{}'.format(i + 1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                kernel_size=1,
                                stride=stride)
        shortcut = BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs


def relu6(x):
    return K.relu(x, max_value=6)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
    in_channels = inputs._keras_shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'expanded_conv_{}_'.format(block_id)
    if block_id:
        # Expand

        x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                   use_bias=False, activation=None,
                   name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'expand_BN')(x)
        x = Activation(relu6, name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'
    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                        use_bias=False, padding='same', dilation_rate=(rate, rate),
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'depthwise_BN')(x)

    x = Activation(relu6, name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_filters,
               kernel_size=1, padding='same', use_bias=False, activation=None,
               name=prefix + 'project')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'project_BN')(x)

    if skip_connection:
        return Add(name=prefix + 'add')([inputs, x])

    # if in_channels == pointwise_filters and stride == 1:
    #    return Add(name='res_connect_' + str(block_id))([inputs, x])

    return x


def deepLabV3_out(img_input, out_shape=(512, 512), classes=7, atrous_rates = (6, 8, 12), entry_block3_stride = (1,2,2), flow =0, midBlocks=4, taskAttation=0, middle_block_rate = 2, exit_block_rates = (2, 4)):
    """ Instantiates the Deeplabv3+ architecture

    Optionally loads weights pre-trained
    on PASCAL VOC. This model is available for TensorFlow only,
    and can only be used with inputs following the TensorFlow
    data format `(width, height, channels)`.
    # Arguments
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: shape of input image. format HxWxC
            PASCAL VOC model was trained on (512,512,3) images
        out_shape: shape of output image. format HxWxclasses
        classes: number of desired classes. If classes != 21,
            last layer is initialized randomly
        flow: 0 for task hse, 1 for task lcz, 2 for multi-task


    # Returns
        A Keras model instance.

    # Raises
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
        ValueError: in case of invalid argument for `weights` or `backbone`

    """

    if K.backend() != 'tensorflow':
        raise RuntimeError('The Deeplabv3+ model is only available with '
                           'the TensorFlow backend.')

    inc_rate=2
    dim=16#32

    #This is based on the website code, but this is not mentioned in the paper.

    x = Conv2D(dim, (3, 3), strides=1,#, strides=(2, 2),
               name='entry_flow_conv1_1', use_bias=False, padding='same')(img_input)
    x = BatchNormalization(name='entry_flow_conv1_1_BN')(x)
    x = Activation('relu')(x)

    dim=dim*inc_rate
    x = _conv2d_same(x, dim, 'entry_flow_conv1_2', kernel_size=3, stride=1)
    x = BatchNormalization(name='entry_flow_conv1_2_BN')(x)
    x = Activation('relu')(x)

    dim=dim*inc_rate
    x = _xception_block(x, [dim, dim, dim], 'entry_flow_block1',
                        skip_connection_type='conv', stride=entry_block3_stride[0],
                        depth_activation=False)
    dim=dim*inc_rate
    x = _xception_block(x, [dim, dim, dim], 'entry_flow_block2',
                               skip_connection_type='conv', stride=entry_block3_stride[1],
                               depth_activation=False)
    #skip1 = x ##size 1/2
    dim=dim*inc_rate
    x = _xception_block(x, [dim, dim, dim], 'entry_flow_block3',
                        skip_connection_type='conv', stride=entry_block3_stride[2],
                        depth_activation=False)#, return_skip=True
    skip1 = x ##size 1/4


    # x0 = cbam.attach_attention_module(x,'cbam_block')
    # x1 = cbam.attach_attention_module(x,'cbam_block')


    for i in range(midBlocks):
        x = _xception_block(x, [dim, dim, dim], 'middle_flow_unit_{}'.format(i + 1),
                            skip_connection_type='sum', stride=1, rate=middle_block_rate,
                            depth_activation=False)

        if i==0:
            x0 = cbam.attach_attention_module(x,'cbam_block')
            x1 = cbam.attach_attention_module(x,'cbam_block')

        else:
            x0 = Concatenate()([x0, x])# attention module for tasks
            x1 = Concatenate()([x1, x])

            x0 = Conv2D(dim, (1, 1), padding='same', use_bias=False)(x0)
            x1 = Conv2D(dim, (1, 1), padding='same', use_bias=False)(x1)

            x0 = cbam.attach_attention_module(x0,'cbam_block')
            x1 = cbam.attach_attention_module(x1,'cbam_block')


    x = _xception_block(x, [dim, dim, dim], 'exit_flow_block1',
                        skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
                        depth_activation=False)

    # # attention module for tasks
    # x0 = cbam.attach_attention_module(x,'cbam_block')+x0
    # x1 = cbam.attach_attention_module(x,'cbam_block')+x1

    x = _xception_block(x, [dim, dim, dim], 'exit_flow_block2',
                        skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                        depth_activation=True)

    # # attention module for tasks
    # x0_ = cbam.attach_attention_module(x,'cbam_block')
    # x1_ = cbam.attach_attention_module(x,'cbam_block')
    # x0 = Add()([x0, x0_])
    # x1 = Add()([x1, x1_])

    # end of feature extractor
    # branching for Atrous Spatial Pyramid Pooling

    # Image Feature branch
    b4 = AveragePooling2D(pool_size=(int(x.shape[1] ), int(x.shape[2])))(x)

    b4 = Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
    b4 = BilinearUpsampling(int(x.shape[1]), int(x.shape[2]))(b4)

    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation('relu', name='aspp0_activation')(b0)

    #atrous_rates 3+(3-1)x(12-1)=25
    b1 = SepConv_BN(x, 256, 'aspp1',
                    rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    b2 = SepConv_BN(x, 256, 'aspp2',
                    rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    b3 = SepConv_BN(x, 256, 'aspp3',
                    rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)
    x_common = Concatenate()([b4, b0, b1, b2, b3])


    dec_skip1 = Conv2D(48, (1, 1), padding='same',
                       use_bias=False, name='feature_projection0')(skip1)
    dec_skip1 = BatchNormalization(
        name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = Activation('relu')(dec_skip1)


    if taskAttation==1:
        x0 = Concatenate()([x0, x_common])# attention module for tasks
        x1 = Concatenate()([x1, x_common])

        x0 = Conv2D(x_common.shape[-1].value, (1, 1), padding='same', use_bias=False)(x0)
        x1 = Conv2D(x_common.shape[-1].value, (1, 1), padding='same', use_bias=False)(x1)

        x0 = cbam.attach_attention_module(x0,'cbam_block')
        x1 = cbam.attach_attention_module(x1,'cbam_block')
    else:
        x0 = x_common
        x1 = x_common

    if flow !=1:
        # x0_ = cbam.attach_attention_module(x_common,'cbam_block')
        # x0 = Concatenate()([x0, x0_])

        x = Conv2D(256, (1, 1), padding='same',
                   use_bias=False, name='concat_projection')(x0)
        x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
        x = Activation('relu')(x)
        x = Dropout(0.1)(x)

        x = Concatenate()([x, dec_skip1])
        x = SepConv_BN(x, 256, 'decoder_conv0',
                       depth_activation=True, epsilon=1e-5)
        x = SepConv_BN(x, 256, 'decoder_conv1',
                       depth_activation=True, epsilon=1e-5)

        x = Conv2D(classes, (1, 1), padding='same')(x)
        if out_shape[0] != x.shape[1]:
            print('BilinearUpsampling needed:')
            x = BilinearUpsampling(output_size=(out_shape[0], out_shape[1]))(x)
        o0 = Activation('softmax', name="pre")(x)#!!!!!!!!!!!!!!!!!!!
    #
    if flow ==0:
        return o0


def deepLabV3_(input_tensor=None, input_shape=(512, 512, 3), out_shape=(512, 512), classes=7, atrous_rates = (6, 8, 12), entry_block3_stride = (1,2,2), flow =0, midBlocks=8, taskAttation=0, middle_block_rate = 1, exit_block_rates = (1, 1)):
    """ Instantiates the Deeplabv3+ architecture

    Optionally loads weights pre-trained
    on PASCAL VOC. This model is available for TensorFlow only,
    and can only be used with inputs following the TensorFlow
    data format `(width, height, channels)`.
    # Arguments
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: shape of input image. format HxWxC
            PASCAL VOC model was trained on (512,512,3) images
        out_shape: shape of output image. format HxWxclasses
        classes: number of desired classes. If classes != 21,
            last layer is initialized randomly
        flow: 0 for task hse, 1 for task lcz, 2 for multi-task


    # Returns
        A Keras model instance.

    # Raises
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
        ValueError: in case of invalid argument for `weights` or `backbone`

    """

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = Input(shape=input_shape)

    # if flow ==2:
    #     o0, o1 = deepLabV3_out(inputs, out_shape=out_shape, classes=classes, atrous_rates = atrous_rates, entry_block3_stride = entry_block3_stride, flow = flow, midBlocks = midBlocks, taskAttation=taskAttation, middle_block_rate = middle_block_rate, exit_block_rates = exit_block_rates)
    #     model = Model(inputs, [o0,o1], name='mtsNN')
    # if flow ==1:
    #     o1 = deepLabV3_out(inputs, out_shape=out_shape, classes=classes, atrous_rates = atrous_rates, entry_block3_stride = entry_block3_stride, flow = flow, midBlocks = midBlocks, taskAttation=taskAttation, middle_block_rate = middle_block_rate, exit_block_rates = exit_block_rates)
    #     model = Model(inputs, o1, name='lczNN')
    if flow ==0:
        o0 = deepLabV3_out(inputs, out_shape=out_shape, classes=classes, atrous_rates = atrous_rates, entry_block3_stride = entry_block3_stride, flow = flow, midBlocks = midBlocks, taskAttation=taskAttation, middle_block_rate = middle_block_rate, exit_block_rates = exit_block_rates)
        print(o0.shape)
        model = Model(inputs, o0, name='Prediction')
    return model
