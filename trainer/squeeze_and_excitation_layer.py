"""
Squeeze and Excitation Module
*****************************
Collection of squeeze and excitation classes where each can be inserted as a block into a neural network architechture
    1. `Channel Squeeze and Excitation <https://arxiv.org/abs/1709.01507>`_
    2. `Spatial Squeeze and Excitation <https://arxiv.org/abs/1803.02579>`_
    3. `Channel and Spatial Squeeze and Excitation <https://arxiv.org/abs/1803.02579>`_
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys

#import numpy as np

import tensorflow as tf

###
class ChannelSELayer(tf.keras.layers.Layer):
  """
  Re-implementation of Squeeze-and-Excitation (SE) block described in:
    *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
  r choice:
    https://towardsdatascience.com/review-senet-squeeze-and-excitation-network-winner-of-ilsvrc-2017-image-classification-a887b98b2883
  """

  def __init__(self,
    #
    num_in_channels,
    reduction_ratio=16,
    #
    data_format='channels_last',
    #
    use_bias=True,
    ##
    kernel_initializer_ec='he_normal',
    kernel_regularizer_ec=None,
    kernel_constraint_ec=None,
    ##
    kernel_initializer_dc='glorot_uniform',
    kernel_regularizer_dc=None,
    kernel_constraint_dc=None,
    ##
    bias_initializer='zeros',
    bias_regularizer=None,
    bias_constraint=None,
    #
    dtype='float32', #tf.float32,
    debug_print=False,
    #
    *args,
    **kwargs
    ):

    # num_in_channels: Number of input channels
    # reduction_ratio: r, By how much should the num_channels should be reduced, default: 16

    super().__init__(dtype=dtype, *args, **kwargs)
    self.input_spec = [
      tf.keras.layers.InputSpec(ndim=4)
    ]
    self.LayerName = self.name

    self.num_in_channels = num_in_channels
    self.reduction_ratio = reduction_ratio
    self.data_format = data_format
    
    if self.data_format == 'channels_first':
      #self.channel_axis = tf.constant(1, dtype=tf.int32)
      self.channel_axis = 1
      self.spatial_axes = [2, 3]
    else:
      #self.channel_axis = tf.constant(-1, dtype=tf.int32)
      self.channel_axis = -1
      self.spatial_axes = [1, 2]

    #
    self.kernel_initializer_ec = tf.keras.initializers.get(kernel_initializer_ec)
    self.kernel_regularizer_ec = tf.keras.regularizers.get(kernel_regularizer_ec)
    self.kernel_constraint_ec = tf.keras.constraints.get(kernel_constraint_ec)
    
    self.kernel_initializer_dc = tf.keras.initializers.get(kernel_initializer_dc)
    self.kernel_regularizer_dc = tf.keras.regularizers.get(kernel_regularizer_dc)
    self.kernel_constraint_dc = tf.keras.constraints.get(kernel_constraint_dc)
    
    self.use_bias = use_bias

    self.bias_initializer = tf.keras.initializers.get(bias_initializer)
    self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self.bias_constraint = tf.keras.constraints.get(bias_constraint)

    #
    #self.dtype = dtype #'float32', #tf.float32
    self.debug_print = debug_print #False

    ##
    self.num_squeezed_channels = int(float(num_in_channels) / float(reduction_ratio))
    if (self.num_squeezed_channels <= 1):
      self.num_squeezed_channels = 1
    
    #print("num_squeezed_channels", self.num_squeezed_channels)
    #print("kernel_initializer_ec", self.kernel_initializer_ec)
    #print("kernel_regularizer_dc", self.kernel_regularizer_dc)
    
    if (self.num_squeezed_channels >= 1):
      ## constants
      self.ec_activation = tf.keras.activations.relu # max(x, 0)
      self.dc_activation = tf.keras.activations.sigmoid
      
      #
      self.ec = tf.keras.layers.Dense(
        units=self.num_squeezed_channels,
        activation=self.ec_activation, # relu
        use_bias=self.use_bias,
        kernel_initializer=self.kernel_initializer_ec, #'he_normal',
        kernel_regularizer=self.kernel_regularizer_ec,
        kernel_constraint=self.kernel_constraint_ec,
        bias_initializer=self.bias_initializer, #'zeros',
        bias_regularizer=self.bias_regularizer, #None,
        bias_constraint=self.bias_constraint, #None,
        activity_regularizer=None
        )
      self.dc = tf.keras.layers.Dense(
        units=self.num_in_channels,
        activation=self.dc_activation,  # relu
        use_bias=self.use_bias,
        kernel_initializer=self.kernel_initializer_dc,  # 'he_normal',
        kernel_regularizer=self.kernel_regularizer_dc,
        kernel_constraint=self.kernel_constraint_dc,
        bias_initializer=self.bias_initializer,  # 'zeros',
        bias_regularizer=self.bias_regularizer,  # None,
        bias_constraint=self.bias_constraint,  # None,
        activity_regularizer=None
      )


  def build(self, input_shape):
    #print("cSE Build input_shape: ", input_shape)
  
    if (type(input_shape) == tf.TensorShape):
      input_shape = tuple(input_shape.as_list())
    if (type(input_shape) == list):
      ii_ts = 0
      for i_ts in input_shape:
        if (type(i_ts) == tf.TensorShape):
          input_shape[ii_ts] = tuple(i_ts.as_list())
        ii_ts = ii_ts + 1
    #print("cSE Build input_shape (after): ", input_shape)
  
    self.built = True
    
  def get_config(self):
    config = super().get_config()
    config.update({
      "num_in_channels": self.num_in_channels,
      "reduction_ratio": self.reduction_ratio,
      "data_format": self.data_format,
      "use_bias": self.use_bias,
      'kernel_initializer_ec': tf.keras.initializers.serialize(self.kernel_initializer_ec),
      'kernel_regularizer_ec': tf.keras.regularizers.serialize(self.kernel_regularizer_ec),
      'kernel_constraint_ec': tf.keras.constraints.serialize(self.kernel_constraint_ec),
      'kernel_initializer_dc': tf.keras.initializers.serialize(self.kernel_initializer_dc),
      'kernel_regularizer_dc': tf.keras.regularizers.serialize(self.kernel_regularizer_dc),
      'kernel_constraint_dc': tf.keras.constraints.serialize(self.kernel_constraint_dc),
      'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
      'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
      'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
      "dtype": self._dtype_policy, #self.dtype,
      "debug_print": self.debug_print
    })
    return config

  def compute_output_shape(self, input_shape):
    output_shape = input_shape
    return tf.TensorShape(output_shape)
  
  # @staticmethod
  # def cast_input(input, dtype):
  #   return tf.cast(input, dtype=dtype)

  # @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, None, None]),
  #                               tf.TensorSpec(shape=[])
  #                               )
  #              )
  #@tf.function(experimental_relax_shapes=True)
  @tf.function
  def call(self, inputs, training=True):
    # [B, H, W, C] or [B, C, H, W]
    images = inputs
    
    # [B, H, W, C] or [B, C, H, W]
    #images = self.cast_input(images, self.dtype)
    images = tf.cast(images, dtype=self._compute_dtype)

    if (self.num_squeezed_channels >= 1):
      # Average along the channel dimension
      # [B, C]
      ec_input = tf.math.reduce_mean(images, axis=self.spatial_axes)
      #print("ec_input", ec_input)
  
      # ec (FC + RELU)
      # [B, C/r]
      ec_output = self.ec(ec_input)
      #print("ec_output", ec_output)
  
      # dc (FC + SIGMOID)
      # [B, C]
      dc_output = self.dc(ec_output)
      #print("dc_output", dc_output)
  
      # [B, 1, 1, C] or [B, C, 1, 1]
      # recalibration_weights = tf.keras.backend.switch(
      #   tf.math.equal(self.channel_axis, tf.constant(1, dtype=tf.int32)),
      #   lambda: tf.reshape(dc_output, shape=[-1, self.num_in_channels, 1, 1]),
      #   lambda: tf.reshape(dc_output, shape=[-1, 1, 1, self.num_in_channels])
      # )
  
      if self.channel_axis == 1:
       recalibration_weights = tf.reshape(dc_output, shape=[-1, self.num_in_channels, 1, 1])
      else:
       recalibration_weights = tf.reshape(dc_output, shape=[-1, 1, 1, self.num_in_channels])
      #recalibration_weights = tf.reshape(dc_output, shape=[-1, 1, 1, self.num_in_channels])
      #print("recalibration_weights", recalibration_weights)
  
      # [B, H, W, C] or [B, C, H, W]
      recalibrated = tf.multiply(images, recalibration_weights)
      #print("recalibrated", recalibrated)
    else:
      #recalibrated = tf.multiply(images, tf.constant(0.5, dtype=tf.float32))
      recalibrated = tf.multiply(images, tf.constant(0.5, dtype=self._compute_dtype))
      #print("recalibrated (every pixel * 0.5, i.e., simulate sigmoid(0.0))", recalibrated)
    
    return recalibrated
  
###
class SpatialSELayer(tf.keras.layers.Layer):
  """
  Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
    *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
  """
  def __init__(self,
    #
    num_in_channels,
    #
    data_format='channels_last',
    #
    use_bias=True,
    ##
    kernel_initializer='glorot_uniform',
    kernel_regularizer=None,
    kernel_constraint=None,
    ##
    bias_initializer='zeros',
    bias_regularizer=None,
    bias_constraint=None,
    #
    dtype='float32', #tf.float32,
    debug_print=False,
    #
    *args,
    **kwargs
    ):

    # num_in_channels: Number of input channels
    
    super().__init__(dtype=dtype, *args, **kwargs)
    self.input_spec = [
      tf.keras.layers.InputSpec(ndim=4)
    ]
    self.LayerName = self.name

    self.num_in_channels = num_in_channels
    self.data_format = data_format
    
    if self.data_format == 'channels_first':
      self.channel_axis = tf.constant(1, dtype=tf.int32)
      #self.spatial_axes = [2, 3]
    else:
      self.channel_axis = tf.constant(-1, dtype=tf.int32)
      #self.spatial_axes = [1, 2]
      
    #
    self.use_bias = use_bias
    
    #
    self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)

    self.bias_initializer = tf.keras.initializers.get(bias_initializer)
    self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self.bias_constraint = tf.keras.constraints.get(bias_constraint)

    #
    #self.dtype = dtype #'float32', #tf.float32
    self.debug_print = debug_print #False

    #
    self.activation = tf.keras.activations.sigmoid
    
    #
    self.conv = tf.keras.layers.Conv2D(
      filters=1,
      kernel_size=(1,1),
      strides=(1,1),
      padding='same',
      data_format=self.data_format,
      dilation_rate=(1,1),
      activation=self.activation,
      use_bias=self.use_bias,
      kernel_initializer=self.kernel_initializer,
      kernel_regularizer=self.kernel_regularizer,
      kernel_constraint=self.kernel_constraint,
      bias_initializer=self.bias_initializer,
      bias_regularizer=self.bias_regularizer,
      bias_constraint=self.bias_constraint,
      activity_regularizer=None
    )

  def build(self, input_shape):
    #print("sSE Build input_shape: ", input_shape)
  
    if (type(input_shape) == tf.TensorShape):
      input_shape = tuple(input_shape.as_list())
    if (type(input_shape) == list):
      ii_ts = 0
      for i_ts in input_shape:
        if (type(i_ts) == tf.TensorShape):
          input_shape[ii_ts] = tuple(i_ts.as_list())
        ii_ts = ii_ts + 1
    #print("sSE Build input_shape (after): ", input_shape)
  
    self.built = True

  def get_config(self):
    config = super().get_config()
    config.update({
      "num_in_channels": self.num_in_channels,
      "data_format": self.data_format,
      "use_bias": self.use_bias,
      'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
      'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
      'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
      'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
      'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
      'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
      "dtype": self._dtype_policy, #self.dtype,
      "debug_print": self.debug_print
    })
    return config

  def compute_output_shape(self, input_shape):
    output_shape = input_shape
    return tf.TensorShape(output_shape)

  # @staticmethod
  # def cast_input(input, dtype):
  #   return tf.cast(input, dtype=dtype)

  # @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, None, None]),
  #                               tf.TensorSpec(shape=[])
  #                               )
  #              )
  #@tf.function(experimental_relax_shapes=True)
  @tf.function
  def call(self, inputs, training=True):
    # [B, H, W, C] or [B, C, H, W]
    images = inputs
    
    # [B, H, W, C] or [B, C, H, W]
    #images = self.cast_input(images, self.dtype)
    images = tf.cast(images, dtype=self._compute_dtype)
    
    # [B, H, W, 1] or [B, 1, H, W]
    q = self.conv(images)
    
    # [B, H, W, C] or [B, C, H, W]
    recalibrated = tf.multiply(images, q)
    
    return recalibrated
  
###
class ChannelSpatialSELayer(tf.keras.layers.Layer):
  """
  Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
    *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
  """
  def __init__(self,
    
    #
    num_in_channels,
    reduction_ratio = 16,
    #
    data_format = 'channels_last',
    #
    use_bias = True,
    ##
    kernel_initializer_ec = 'he_normal',
    kernel_regularizer_ec = None,
    kernel_constraint_ec = None,
    ##
    kernel_initializer_dc = 'glorot_uniform',
    kernel_regularizer_dc = None,
    kernel_constraint_dc = None,
    ##
    kernel_initializer='glorot_uniform',
    kernel_regularizer=None,
    kernel_constraint=None,
    ##
    bias_initializer = 'zeros',
    bias_regularizer = None,
    bias_constraint = None,
    #
    dtype = 'float32',  # tf.float32,
    debug_print = False,
    #
    *args,
    ** kwargs
    ):
    
    # num_in_channels: Number of input channels
    # reduction_ratio: r, By how much should the num_channels should be reduced, default: 16

    super().__init__(dtype=dtype, *args, **kwargs)
    self.input_spec = [
      tf.keras.layers.InputSpec(ndim=4)
    ]
    self.LayerName = self.name
    
    #
    self.num_in_channels = num_in_channels
    self.reduction_ratio = reduction_ratio
    #
    self.data_format = data_format
    #
    self.use_bias = use_bias
    ##
    self.kernel_initializer_ec = tf.keras.initializers.get(kernel_initializer_ec)
    self.kernel_regularizer_ec = tf.keras.regularizers.get(kernel_regularizer_ec)
    self.kernel_constraint_ec = tf.keras.constraints.get(kernel_constraint_ec)

    self.kernel_initializer_dc = tf.keras.initializers.get(kernel_initializer_dc)
    self.kernel_regularizer_dc = tf.keras.regularizers.get(kernel_regularizer_dc)
    self.kernel_constraint_dc = tf.keras.constraints.get(kernel_constraint_dc)
    
    self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)

    self.bias_initializer = tf.keras.initializers.get(bias_initializer)
    self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self.bias_constraint = tf.keras.constraints.get(bias_constraint)

    #
    #self.dtype = dtype
    self.debug_print = debug_print

    #
    self.cSELyr = ChannelSELayer(
      self.num_in_channels,
      reduction_ratio=self.reduction_ratio,
      data_format=self.data_format, #'channels_last',
      use_bias=self.use_bias, #True,
      kernel_initializer_ec=self.kernel_initializer_ec, #'he_normal',
      kernel_regularizer_ec=self.kernel_regularizer_ec, #None,
      kernel_constraint_ec=self.kernel_constraint_ec, #None,
      kernel_initializer_dc=self.kernel_initializer_dc, #'glorot_uniform',
      kernel_regularizer_dc=self.kernel_regularizer_dc, #None,
      kernel_constraint_dc=self.kernel_constraint_dc, #None,
      bias_initializer=self.bias_initializer, #'zeros',
      bias_regularizer=self.bias_regularizer, #None,
      bias_constraint=self.bias_constraint, #None,
      dtype=self._dtype_policy, #self.dtype, #'float32',  # tf.float32,
      debug_print=self.debug_print #False,
      )
    
    self.sSELyr = SpatialSELayer(
      self.num_in_channels,
      data_format=self.data_format, #'channels_last',
      use_bias=self.use_bias, #True,
      kernel_initializer=self.kernel_initializer, #'glorot_uniform',
      kernel_regularizer=self.kernel_regularizer, #None,
      kernel_constraint=self.kernel_constraint, #None,
      bias_initializer=self.bias_initializer, #'zeros',
      bias_regularizer=self.bias_regularizer, #None,
      bias_constraint=self.bias_constraint, #None,
      dtype=self._dtype_policy, #self.dtype, #'float32', #tf.float32,
      debug_print=self.debug_print #False,
      )

  def build(self, input_shape):
    #print("csSE Build input_shape: ", input_shape)
  
    if (type(input_shape) == tf.TensorShape):
      input_shape = tuple(input_shape.as_list())
    if (type(input_shape) == list):
      ii_ts = 0
      for i_ts in input_shape:
        if (type(i_ts) == tf.TensorShape):
          input_shape[ii_ts] = tuple(i_ts.as_list())
        ii_ts = ii_ts + 1
    #print("csSE Build input_shape (after): ", input_shape)
  
    self.built = True

  def get_config(self):
    config = super().get_config()
    config.update({
      "num_in_channels": self.num_in_channels,
      "reduction_ratio": self.reduction_ratio,
      "data_format": self.data_format,
      "use_bias": self.use_bias,
      'kernel_initializer_ec': tf.keras.initializers.serialize(self.kernel_initializer_ec),
      'kernel_regularizer_ec': tf.keras.regularizers.serialize(self.kernel_regularizer_ec),
      'kernel_constraint_ec': tf.keras.constraints.serialize(self.kernel_constraint_ec),
      'kernel_initializer_dc': tf.keras.initializers.serialize(self.kernel_initializer_dc),
      'kernel_regularizer_dc': tf.keras.regularizers.serialize(self.kernel_regularizer_dc),
      'kernel_constraint_dc': tf.keras.constraints.serialize(self.kernel_constraint_dc),
      'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
      'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
      'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
      'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
      'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
      'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
      "dtype": self._dtype_policy, #self.dtype,
      "debug_print": self.debug_print
    })
    return config

  def compute_output_shape(self, input_shape):
    output_shape = input_shape
    return tf.TensorShape(output_shape)

  # @staticmethod
  # def cast_input(input, dtype):
  #   return tf.cast(input, dtype=dtype)

  # @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, None, None]),
  #                               tf.TensorSpec(shape=[])
  #                               )
  #              )
  #@tf.function(experimental_relax_shapes=True)
  @tf.function
  def call(self, inputs, training=True):
    # [B, H, W, C] or [B, C, H, W]
    images = inputs
    
    # [B, H, W, C] or [B, C, H, W]
    #images = self.cast_input(images, self.dtype)
    images = tf.cast(images, dtype=self._compute_dtype)
    
    # [B, H, W, C] or [B, C, H, W]
    cSE_images = self.cSELyr(images)
    
    # [B, H, W, C] or [B, C, H, W]
    sSE_images = self.sSELyr(images)
    
    csSE_images = tf.add(cSE_images, sSE_images)
    
    return csSE_images
  
    
    