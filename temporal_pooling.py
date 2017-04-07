# -*- coding: utf-8 -*-

"""
    Temporal Pooling Layers
    
    2017/04/07  ZYD

    This file is the implementation of temporal pooling layers.
    These layers take a 5D tensor as input (B x T x H x W x C)
    and output tensor of size B x H' x W' x C

    [Reference]
        $KERAS_ROOT/keras/layers/pooling.py

    [TODO]
        1.  Temporal average pooling            [NOT_IMPLEMENTED]
        2.  Temporal max pooling                [NOT_IMPLEMENTED]
        3.  Temporal global average pooling     [NOT_IMPLEMENTED]
        4.  Temporal global max pooling         [NOT_IMPLEMENTED]
"""

from __future__ import absolute_import

import keras.backend as K
from keras.engine import Layer, InputSpec


class _TemporalPooling2D(Layer):
    """ Abstract class for different temporal pooling 2D layers """
    def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
                 dim_ordering='default', **kwargs):
        super(_TemporalPooling2D, self).__init__(**kwargs)

        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        if dim_ordering not in {'tf', 'th'}:
            raise ValueError('`dim_ordering` must be in {tf, th}.')

        self.pool_size = tuple(pool_size)
        if strides is None:
            strides = self.pool_size
        self.strides = tuple(strides)

        if border_mode not in {'valid', 'same'}:
            raise ValueError('`border_mode` must be in {valid, same}.')
        self.border_mode = border_mode

        self.input_spec = [InputSpec(ndim=4)]

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.dim_ordering == 'tf':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            raise ValueError('Invalid dim_ordering:', self.dim_ordering)

        if self.dim_ordering == 'th':
            return (input_shape[0], input_shape[1], rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, input_shape[3])

