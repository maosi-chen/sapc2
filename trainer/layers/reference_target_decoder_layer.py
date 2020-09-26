from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys

import tensorflow as tf
#import tensorflow.keras as K


from tensorflow.python.keras.utils import tf_utils
from tensorflow.keras.layers import BatchNormalization

from .merge_layer import MergeLayer


class DecodeLayer(tf.keras.layers.Layer):

  def __init__(
    self,
    USF,
    #
    Merge_filters,
    Merge_kernel_size,
    Merge_data_format='channels_last',
    Merge_activation=None,
    Merge_kernel_initializer='he_normal', #tf.keras.initializers.glorot_uniform,
    Merge_kernel_regularizer=None,
    Merge_use_batch_normalization=True,
    Merge_use_bias=False,
    #
    use_BP_MSK=True,
    UMLoss_weight=1.0,
    #
    dtype=tf.float32,
    debug_print=False,     
    #
    *args,
    **kwargs
    ):

    super().__init__(dtype=dtype, *args, **kwargs)

    self.USF = USF
    self.interpolation_UpSmpl = 'bilinear' #'nearest'

    self.Merge_filters = Merge_filters
    self.Merge_kernel_size = Merge_kernel_size

    self.Merge_data_format = Merge_data_format
    if self.Merge_data_format == 'channels_first':
      self.channel_axis = 1
    else:
      self.channel_axis = -1
    self.Merge_activation = tf.keras.activations.get(Merge_activation)
    self.Merge_kernel_initializer = tf.keras.initializers.get(Merge_kernel_initializer)
    self.Merge_kernel_regularizer = tf.keras.regularizers.get(Merge_kernel_regularizer)
    self.Merge_use_batch_normalization = Merge_use_batch_normalization
    self.Merge_use_bias = Merge_use_bias

    self.use_BP_MSK = use_BP_MSK
    self.UMLoss_weight = UMLoss_weight
    self.UMLoss_weight_ts = tf.convert_to_tensor(self.UMLoss_weight, dtype=self._compute_dtype)

    self.debug_print = debug_print

    #
    self.UpSmpl2DLyr = tf.keras.layers.UpSampling2D(
      size=self.USF,
      data_format=self.Merge_data_format,
      interpolation=self.interpolation_UpSmpl,
      dtype=self._dtype_policy,
      )

    self.MergeLyr = MergeLayer(
      self.Merge_filters,
      self.Merge_kernel_size,
      data_format=self.Merge_data_format,
      activation=None, #self.Merge_activation,
      kernel_initializer=self.Merge_kernel_initializer, #tf.keras.initializers.glorot_uniform,
      kernel_regularizer = self.Merge_kernel_regularizer,
      use_batch_normalization=False, #self.Merge_use_batch_normalization,
      use_bias=self.Merge_use_bias,
      #
      dtype=self._dtype_policy, #.dtype,
      debug_print=self.debug_print
      )

    self.AddResLyr = tf.keras.layers.Add()

    self.epsilon = 6e-8 # minimum of float16
    self.epsilon_ts = tf.convert_to_tensor(self.epsilon, dtype=self._compute_dtype)

    #
    #Batch normalization layer
    if self.Merge_use_batch_normalization:
      self.batch_normalization = BatchNormalization(
        trainable=True, epsilon = self.epsilon,
        axis=self.channel_axis,
        dtype=self._dtype_policy #dtype=self.dtype #self._dtype_policy
        )

    self.ActivationLyr = self.Merge_activation


  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    # if (type(input_shape) == tf.TensorShape):
    #   input_shape = tuple(input_shape.as_list())
    #
    # if (type(input_shape) == list):
    #   ii_ts = 0
    #   for i_ts in input_shape:
    #     if (type(i_ts) == tf.TensorShape):
    #       input_shape[ii_ts] = tuple(i_ts.as_list())
    #       ii_ts = ii_ts + 1

    self.built = True

  ## implement this method as required by keras.
  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    #Takes the same input shape as build, and calculates the output shape based on that
    #B H and W are the same as either input image. 
    #However Outfeatures is specified in the constructor, so we utilize that 
    
    #Set the output and leave B, H and W alone
    output_shape = input_shape[1]
    
    #Change the number of channels to n_outputs
    output_shape[self.channel_axis] = self.OutFeatures
    return output_shape

  def call(self, inputs, training = None): #True

    InLowResImg = inputs[0]
    MrgPConvImg = inputs[1]
    if self.use_BP_MSK:
      BP_Img = inputs[2]
      MSK_Img = inputs[3]

    InLowResImg = tf.cast(InLowResImg, dtype=self._compute_dtype)
    MrgPConvImg = tf.cast(MrgPConvImg, dtype=self._compute_dtype)
    if self.use_BP_MSK:
      BP_Img = tf.cast(BP_Img, dtype=self._compute_dtype)
      MSK_Img = tf.cast(MSK_Img, dtype=self._compute_dtype)

    UpsmpledInImg = self.UpSmpl2DLyr(InLowResImg)

    DecodedImage = self.MergeLyr([MrgPConvImg, UpsmpledInImg], training=training)

    if self.Merge_use_batch_normalization:
      DecodedImage = self.batch_normalization(DecodedImage, training=training)

    if self.ActivationLyr is not None:
      DecodedImage = self.ActivationLyr(DecodedImage)

    if self.use_BP_MSK:
      ## calculate the unmasked MSE between BP and PSConLyr images
      unmaskedSSE = tf.math.reduce_sum(
        tf.math.square((DecodedImage - BP_Img) * MSK_Img)
      )
      unmaskedCnt = tf.math.reduce_sum(MSK_Img)
      unmaskedMSE = unmaskedSSE / unmaskedCnt
      weighted_unmaskedMSE = unmaskedMSE * self.UMLoss_weight_ts

      ## add to loss
      self.add_loss(weighted_unmaskedMSE)

    return DecodedImage

  def get_config(self):
    config = super().get_config()
    config.update({
      'USF': self.USF,
      'Merge_filters': self.Merge_filters,
      'Merge_kernel_size': self.Merge_kernel_size,
      'Merge_data_format': self.Merge_data_format,
      'Merge_activation': tf.keras.activations.serialize(self.Merge_activation),
      'Merge_kernel_initializer': tf.keras.initializers.serialize(self.Merge_kernel_initializer),
      'Merge_kernel_regularizer': tf.keras.regularizers.serialize(self.Merge_kernel_regularizer),
      'Merge_use_batch_normalization': self.Merge_use_batch_normalization,
      'Merge_use_bias': self.Merge_use_bias,
      'use_BP_MSK': self.use_BP_MSK,
      'UMLoss_weight': self.UMLoss_weight,
      'dtype': self._dtype_policy, #self.dtype,
      'debug_print': self.debug_print
      })
    return config
    
