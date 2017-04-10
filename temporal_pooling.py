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
        1.  Temporal average pooling            [✓]
        2.  Temporal max pooling                [✓]
        3.  Temporal global average pooling     [NOT_IMPLEMENTED]
        4.  Temporal global max pooling         [NOT_IMPLEMENTED]
"""

from __future__ import absolute_import

import keras.backend as K
from keras.engine import Layer, InputSpec
from kera.utils.np_utils import import conv_output_length


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
            channels = input_shape[2]
            rows = input_shape[3]
            cols = input_shape[4]
        elif self.dim_ordering == 'tf':
            rows = input_shape[2]
            cols = input_shape[3]
            channels = input_shape[4]
        else:
            raise ValueError('Invalid dim_ordering:', self.dim_ordering)

       rows = conv_output_length(rows, self.pool_size[0], 
                                 self.border_mode, self.stride[0])
       cols = conv_output_length(cols, self.pool_size[0], 
                                 self.border_mode, self.stride[0])

        if self.dim_ordering == 'th':
            return (input_shape[0], channels, rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, channels)

    def _pooling_function(self, inputs, pool_size, strides, border_mode,
                          dim_ordering):
        raise NotImplementedError

    def call(self, x, mask=None):
        output = self._pooling_function(inputs=x, pool_size=self.pool_size,
                                        strides=self.strides,
                                        border_mode=self.border_mode,
                                        dim_ordering=self.dim_ordering)
        return output

    def get_config(self):
        config = {
            'pool_size': self.pool_size,
            'border_mode': self.border_mode,
            'strides': self.strides,
            'dim_ordering': self.dim_ordering
        }
        base_config = super(_TemporalPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TemporalAveragePooling2D(_TemporalPooling2D):
    """
        Temporal average pooling operation for spatial-temporal data.
        This layer accept 5D input and will compute average pooling on feature
        maps of the same channel at different time step.
        The output spatial size is determined by pool_size, strides and 
        border_mode. 
        This is different from native keras AveragePooling3D in that the native
        one will not average the corresponding channel of feature maps at each
        time step.

        E.g. Input is:
            t = 1           t = 2       ...     t = N
            1|2|3           1|2|3               1|2|3
        At each time step, the number of channel of input feature map is 3 
        (1|2|3) and there are N steps.
        Ouput will be:
            1'|2'|3'
        where x' = avg(x@t1 + x@t2 + ... + x@t_N)

        # Arguments
            pool_size:      Tuple of 2 integers. Factors by which to downscale 
                            (horizontal, vertical). For example, (2, 2) will 
                            harve the spatial size.
            strides:        Tuple of 2 integers, or None. Strides values. If
                            None, it will default to `pool_size`.
            border_mode:    'valid' or 'same'
            dim_ordering:   'th' ot 'tf'.
			spatial_pool_
			mode:			which kind of spatial pooling to use

        # Input shape
            5D tensor with shape:
                'th'    B x T x C x H x W   |   'tf'    B x T x H x W x C
        
        # Output shape
            4D tensor with shape:
                'th'    B x C x H' x W'       |   'tf'    B x H' x W' x C
            where H' = (H - PH + 2P) / SH + 1, W' = (W - PW + 2P) / SW + 1
    """

    def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
                 dim_ordering='default', spatial_pool_mode = 'avg', **kwargs):
		self.spatial_pool_mode = spatial_pool_mode
        super(TemporalAveragePooling2D, self).__init__(pool_size, strides,
                                                       border_mode, dim_ordering,
                                                       **kwargs)

    def _pooling_function(self, inputs, pool_size, strides, 
                          border_mode, dim_ordering):
		# Averaging over temporal dimension
		avg = K.mean(inputs, axis=[1])
		# And apply spatial pooling
		output = K.pool2d(avg, pool_size, stride, border_mode,
						  dim_ordering, pool_mode=self.spatial_pool_mode)
		return output


class TemporalMaxPooling2D(_TemporalPooling2D):
    """
        Temporal max pooling operation for spatial-temporal data.
        This layer accept 5D input and will compute max pooling on feature
        maps of the same channel at different time step.
        The output spatial size is determined by pool_size, strides and 
        border_mode. 
        This is different from native keras MaxPooling3D in that the native
        one will not maximize the corresponding channel of feature maps at each
        time step.

        E.g. Input is:
            t = 1           t = 2       ...     t = N
            1|2|3           1|2|3               1|2|3
        At each time step, the number of channel of input feature map is 3 
        (1|2|3) and there are N steps.
        Ouput will be:
            1'|2'|3'
        where x' = max(x@t1, x@t2, ..., x@t_N)

        # Arguments
            pool_size:      Tuple of 2 integers. Factors by which to downscale 
                            (horizontal, vertical). For example, (2, 2) will 
                            harve the spatial size.
            strides:        Tuple of 2 integers, or None. Strides values. If
                            None, it will default to `pool_size`.
            border_mode:    'valid' or 'same'
            dim_ordering:   'th' ot 'tf'.
			spatial_pool_
			mode:			which kind of spatial pooling to use

        # Input shape
            5D tensor with shape:
                'th'    B x T x C x H x W   |   'tf'    B x T x H x W x C
        
        # Output shape
            4D tensor with shape:
                'th'    B x C x H' x W'       |   'tf'    B x H' x W' x C
            where H' = (H - PH + 2P) / SH + 1, W' = (W - PW + 2P) / SW + 1
    """

    def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
                 dim_ordering='default', spatial_pool_mode = 'avg', **kwargs):
		self.spatial_pool_mode = spatial_pool_mode
        super(TemporalMaxPooling2D, self).__init__(pool_size, strides,
                                                       border_mode, dim_ordering,
                                                       **kwargs)

    def _pooling_function(self, inputs, pool_size, strides, 
                          border_mode, dim_ordering):
		# Maximize over temporal dimension
		m = K.max(inputs, axis=[1])
		# And apply spatial pooling
		output = K.pool2d(m, pool_size, stride, border_mode,
						  dim_ordering, pool_mode=self.spatial_pool_mode)
		return output


