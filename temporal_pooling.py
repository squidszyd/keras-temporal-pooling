"""
	Temporal Pooling Layers

	2017/04/07  ZYD

	This file is the implementation of temporal pooling layers.
	These layers take a 5D tensor as input (B x T x H x W x C)
	and output tensor of size B x H' x W' x C

	[Reference]
		$KERAS_ROOT/keras/layers/pooling.py
"""

from __future__ import absolute_import

import keras.backend as K
from keras.engine import Layer, InputSpec
from keras.utils.np_utils import conv_output_length


class _TemporalPooling2D(Layer):
	""" Abstract class for different temporal pooling 2D layers """
	def __init__(self, spatial_pool=False, pool_size=(2, 2), strides=None, border_mode='valid',
			dim_ordering='default', **kwargs):
		super(_TemporalPooling2D, self).__init__(**kwargs)

		self.spatial_pool = spatial_pool

		if dim_ordering == 'default':
			self.dim_ordering = K.image_dim_ordering()
		elif dim_ordering not in {'tf', 'th'}:
			raise ValueError('`dim_ordering` must be in {tf, th}.')
		else:
			self.dim_ordering = dim_ordering

		self.pool_size = tuple(pool_size)
		if strides is None:
			strides = self.pool_size
		self.strides = tuple(strides)

		if border_mode not in {'valid', 'same'}:
			raise ValueError('`border_mode` must be in {valid, same}.')
		self.border_mode = border_mode

		self.input_spec = [InputSpec(ndim=5)]

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

		if self.spatial_pool is True:
			rows = conv_output_length(rows, self.pool_size[0], self.border_mode, self.strides[0])
			cols = conv_output_length(cols, self.pool_size[0], self.border_mode, self.strides[0])

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
			spatial_pool    Whether to apply spatial pooling
			pool_size:      Tuple of 2 integers. Factors by which to downscale
							(horizontal, vertical). For example, (2, 2) will
							harve the spatial size.
			strides:        Tuple of 2 integers, or None. Strides values. If
							None, it will default to `pool_size`.
			border_mode:    'valid' or 'same'
			dim_ordering:   'th' ot 'tf'.
						spatial_pool_
						mode:                   which kind of spatial pooling to use

		# Input shape
			5D tensor with shape:
				'th'    B x T x C x H x W   |   'tf'    B x T x H x W x C

		# Output shape
			4D tensor with shape:
				'th'    B x C x H' x W'       |   'tf'    B x H' x W' x C
			where H' = (H - PH + 2P) / SH + 1, W' = (W - PW + 2P) / SW + 1
	"""

	def __init__(self, spatial_pool=False, pool_size=(2, 2), strides=None, border_mode='valid',
			dim_ordering='default', spatial_pool_mode = 'avg', **kwargs):
		self.spatial_pool_mode = spatial_pool_mode
		self.spatial_pool = spatial_pool
		super(TemporalAveragePooling2D, self).__init__(spatial_pool, pool_size, strides,
				border_mode, dim_ordering, **kwargs)

	def _pooling_function(self, inputs, pool_size, strides, border_mode,
			dim_ordering):
		# Averaging over temporal dimension
		avg = K.mean(inputs, axis=[1])
		# And apply spatial pooling
		if self.spatial_pool is True:
			output = K.pool2d(avg, pool_size, strides, border_mode,
					dim_ordering, pool_mode=self.spatial_pool_mode)
			return output
		else:
			return avg


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
			spatial_pool    Whether to apply spatial pooling
			pool_size:      Tuple of 2 integers. Factors by which to downscale
							(horizontal, vertical). For example, (2, 2) will
							harve the spatial size.
			strides:        Tuple of 2 integers, or None. Strides values. If
							None, it will default to `pool_size`.
			border_mode:    'valid' or 'same'
			dim_ordering:   'th' ot 'tf'.
						spatial_pool_
						mode:                   which kind of spatial pooling to use

		# Input shape
			5D tensor with shape:
				'th'    B x T x C x H x W   |   'tf'    B x T x H x W x C

		# Output shape
			4D tensor with shape:
				'th'    B x C x H' x W'       |   'tf'    B x H' x W' x C
			where H' = (H - PH + 2P) / SH + 1, W' = (W - PW + 2P) / SW + 1
	"""

	def __init__(self, spatial_pool=False, pool_size=(2, 2), strides=None, border_mode='valid',
			dim_ordering='default', spatial_pool_mode = 'avg', **kwargs):
		self.spatial_pool_mode = spatial_pool_mode
		self.spatial_pool = spatial_pool
		super(TemporalAveragePooling2D, self).__init__(spatial_pool, pool_size, strides,
				border_mode, dim_ordering, **kwargs)

	def _pooling_function(self, inputs, pool_size, strides, border_mode,
			dim_ordering):
		# Maximize over temporal dimension
		m = K.max(inputs, axis=[1])
		if self.spatial_pool is True:
			# And apply spatial pooling
			output = K.pool2d(m, pool_size, strides, border_mode,
					dim_ordering, pool_mode=self.spatial_pool_mode)
			return output
		else:
			return m


class TemporalAverageGlobalPooling2D(_TemporalPooling2D):
	"""
		Temporal average global pooling operation for spatial-temporal data.
		This layer accept 5D input and will compute average pooling on feature
		maps of the same channel at different time step and apply global spatial
		pooling.

		E.g. Input is:
			t = 1           t = 2       ...     t = N
			1|2|3           1|2|3               1|2|3
		At each time step, the number of channel of input feature map is 3
		(1|2|3) and there are N steps.
		Ouput will be:
			1'|2'|3'
		where x' = avg(x@t1 + x@t2 + ... + x@t_N)

		# Arguments
			dim_ordering:   'th' ot 'tf'.
			keep_dims:       boolean, default False
			spatial_pool_
			mode:           which kind of spatial pooling to use

		# Input shape
			5D tensor with shape:
				'th'    B x T x C x H x W   |   'tf'    B x T x H x W x C

		# Output shape
			If keep_dims is True:
				5D tensor with shape B x C x 1 x 1 (th) or B x 1 x 1 x C (tf)
			If keep_dims is False:
				3D tensor with shape B x C
	"""

	def __init__(self, spatial_pool_mode = 'avg', dim_ordering='default',
			keep_dims = False, **kwargs):
		self.spatial_pool_mode = spatial_pool_mode
		self.keep_dims = keep_dims
		super(TemporalAverageGlobalPooling2D, self).__init__(dim_ordering=dim_ordering, **kwargs)

	def _pooling_function(self, inputs, pool_size, strides,
			border_mode, dim_ordering):
		# Averaging over temporal dimension
		avg = K.mean(inputs, axis=[1])
		# And apply spatial pooling
		output = None
		if self.spatial_pool_mode == 'avg':
			if self.dim_ordering == 'tf':
				output = K.mean(avg, axis=[1, 2], keep_dims=self.keep_dims)
			else:
				output = K.mean(avg, axis=[2, 3], keep_dims=self.keep_dims)
		elif self.spatial_pool_mode == 'max':
			if self.dim_ordering == 'tf':
				output = K.max(avg, axis=[1, 2], keep_dims=self.keep_dims)
			else:
				output = K.max(avg, axis=[2, 3], keep_dims=self.keep_dims)
		else:
			raise NoImplementedError
		return output

	def get_output_shape_for(self, input_shape):
		if self.dim_ordering == 'th':
			return (input_shape[0], input_shape[2])
		elif self.dim_ordering == 'tf':
			return (input_shape[0], input_shape[4])
		else:
			raise ValueError('Invalid dim_ordering:', self.dim_ordering)


class TemporalMaxGlobalPooling2D(_TemporalPooling2D):
	"""
		Temporal max global pooling operation for spatial-temporal data.
		This layer accept 5D input and will compute maximize pooling on feature
		maps of the same channel at different time step and apply global spatial
		pooling.

		E.g. Input is:
			t = 1           t = 2       ...     t = N
			1|2|3           1|2|3               1|2|3
		At each time step, the number of channel of input feature map is 3
		(1|2|3) and there are N steps.
		Ouput will be:
			1'|2'|3'
		where x' = max(x@t1, x@t2, ..., x@t_N)

		# Arguments
			dim_ordering:   'th' ot 'tf'.
			keep_dims:       boolean, default False
			spatial_pool_
			mode:           which kind of spatial pooling to use

		# Input shape
			5D tensor with shape:
				'th'    B x T x C x H x W   |   'tf'    B x T x H x W x C

		# Output shape
			If keep_dims is True:
				5D tensor with shape B x C x 1 x 1 (th) or B x 1 x 1 x C (tf)
			If keep_dims is False:
				3D tensor with shape B x C
	"""

	def __init__(self, spatial_pool_mode = 'avg', dim_ordering='default',
			keep_dims = False, **kwargs):
		self.spatial_pool_mode = spatial_pool_mode
		self.keep_dims = keep_dims
		super(TemporalMaxGlobalPooling2D, self).__init__(dim_ordering=dim_ordering, **kwargs)

	def _pooling_function(self, inputs, pool_size, strides, border_mode,
			dim_ordering):
		# Maximize over temporal dimension
		m = K.max(inputs, axis=[1])
		# And apply spatial pooling
		output = None
		if self.spatial_pool_mode == 'avg':
			if self.dim_ordering == 'tf':
				output = K.mean(m, axis=[1, 2], keep_dims=self.keep_dims)
			else:
				output = K.mean(m, axis=[2, 3], keep_dims=self.keep_dims)
		elif self.spatial_pool_mode == 'max':
			if self.dim_ordering == 'tf':
				output = K.max(m, axis=[1, 2], keep_dims=self.keep_dims)
			else:
				output = K.max(m, axis=[2, 3], keep_dims=self.keep_dims)
		else:
			raise NoImplementedError
		return output

	def get_output_shape_for(self, input_shape):
		if self.dim_ordering == 'th':
			return (input_shape[0], input_shape[2])
		elif self.dim_ordering == 'tf':
			return (input_shape[0], input_shape[4])
		else:
			raise ValueError('Invalid dim_ordering:', self.dim_ordering)
