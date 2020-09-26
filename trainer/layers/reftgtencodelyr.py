from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys
import numpy as np

import tensorflow as tf
import tensorflow.keras as K

from tensorflow.python.keras.utils import tf_utils

from .merge_layer import MergeLayer
from .partial_merge_layer import PartialMergeLayer
#from .gated_partial_merge import GatedPartialMergeLayer
from .partial_convolution_layer import PConv2D

"""## Reference-Target Encoder Layer (RefTgtEncodeLyr)"""
class RefTgtEncodeLyr(tf.keras.layers.Layer):

  def __init__(
    self, 
    # ---------------------------
    # args (common)
    # ---------------------------
    # args (Partial Convolution)
    PConv_filters,
    PConv_kernel_size,
    # ---------------------------
    # args (Partial Merge)
    PMerge_filters,
    PMerge_kernel_size,
    # ---------------------------
    # kwargs (common)
    data_format='channels_last',
    dtype=tf.float32, 
    debug_print=False,      
    # ---------------------------
    # kwargs (Partial Convolution)
    PConv_strides=1, 
    PConv_padding="same", 
    PConv_dilation_rate=1,
    PConv_activation=None,
    PConv_use_bias=True,
    PConv_kernel_initializer='he_normal',
    PConv_kernel_regularizer=None,
    PConv_use_batch_normalization = True,
    # ---------------------------
    # kwargs (Partial Merge)
    GPMerge_PConv_activation=tf.keras.layers.PReLU(),
    PMerge_activation=None,
    #GPMerge_PConv_use_batch_normalization=True,
    PMerge_use_batch_normalization=True,
    PMerge_use_bias=True,
    #
    PMerge_kernel_initializer='he_normal', #K.initializers.glorot_uniform,
    PMerge_bias_initializer='zeros',
    PMerge_kernel_regularizer=None,
    PMerge_bias_regularizer=None,
    PMerge_activity_regularizer=None,
    PMerge_kernel_constraint=None,
    PMerge_bias_constraint=None,
    #
    compress_ratio=1.0,

    skip_residual_before_pmerge=False,

    # ---------------------------
    # args (tf.keras.layers.layer)
    *args, 
    **kwargs
    ):
    super().__init__(dtype=dtype, *args, **kwargs)
    
    # ---------------------------
    # Common
    # ---------------------------
    
    self.LayerName = self.name
    self.data_format = data_format
    if self.data_format == 'channels_first':
      self.channel_axis = 1
    else:
      self.channel_axis = -1
    self.debug_print = debug_print
    
    #
    self.input_spec = [
      tf.keras.layers.InputSpec(ndim=4), 
      tf.keras.layers.InputSpec(ndim=4), 
      tf.keras.layers.InputSpec(ndim=4)
    ]
    
    
    # ---------------------------
    # PConv2D
    # ---------------------------

    self.LayerName_PConv = 'PConv' + self.name
    
    self.PConv_filters = PConv_filters
    self.PConv_kernel_size = PConv_kernel_size
    
    self.PConv_data_format = self.data_format
    
    self.PConv_strides = PConv_strides 
    self.PConv_padding = PConv_padding
    self.PConv_dilation_rate = PConv_dilation_rate
    self.PConv_activation = tf.keras.activations.get(PConv_activation)
    self.PConv_use_bias = PConv_use_bias
    self.PConv_kernel_initializer = tf.keras.initializers.get(PMerge_kernel_initializer) #kernel_initializer = tf.keras.initializers.get(PConv_kernel_initializer)
    self.PConv_kernel_regularizer = tf.keras.regularizers.get(PConv_kernel_regularizer)
    self.PConv_use_batch_normalization = PConv_use_batch_normalization
    
    # ---------------------------
    # PConv2D
    # ---------------------------
    
    # Layer
    self.PConv2DLyr = PConv2D(
      self.PConv_filters, #10, 
      self.PConv_kernel_size, #(5,5), 
      strides = self.PConv_strides, #=2, 
      data_format = self.PConv_data_format, #="channels_last",
      activation = self.PConv_activation, #=None, #tf.keras.activations.relu
      use_bias = self.PConv_use_bias, #=True, #False, #True
      kernel_initializer = self.PConv_kernel_initializer, #=tf.keras.initializers.glorot_uniform,
      kernel_regularizer = self.PConv_kernel_regularizer,
      use_batch_normalization = self.PConv_use_batch_normalization,
      dtype = self._dtype_policy, #self.dtype,
      debug_print = self.debug_print,
      #
      name = self.LayerName_PConv
      )

    # ---------------------------
    # PartialMergeLayer (parameters)
    # ---------------------------
    #
    self.LayerName_GPMerge = 'GatedPartialMerge' + self.name
    self.LayerName_PMerge = 'PMerge' + self.name

    self.PMerge_filters = PMerge_filters
    self.PMerge_kernel_size = PMerge_kernel_size
    
    #
    self.PMerge_kernel_initializer = tf.keras.initializers.get(PMerge_kernel_initializer)
    self.PMerge_kernel_regularizer = tf.keras.regularizers.get(PMerge_kernel_regularizer)
    self.PMerge_kernel_constraint = tf.keras.constraints.get(PMerge_kernel_constraint)
    
    #
    self.PMerge_use_bias = PMerge_use_bias
    
    #
    self.PMerge_bias_initializer = tf.keras.initializers.get(PMerge_bias_initializer)
    self.PMerge_bias_regularizer = tf.keras.regularizers.get(PMerge_bias_regularizer)
    self.PMerge_bias_constraint = tf.keras.constraints.get(PMerge_bias_constraint)

    #
    self.PMerge_activity_regularizer = tf.keras.regularizers.get(PMerge_activity_regularizer)

    #
    #self.GPMerge_PConv_use_batch_normalization = GPMerge_PConv_use_batch_normalization #True
    self.PMerge_use_batch_normalization = PMerge_use_batch_normalization
    
    
    # data_format: 'channels_first' [B, C, H, W] or 
    #              'channels_last'  [B, H, W, C]
    self.PMerge_data_format = self.data_format
    
    #
    self.GPMerge_PConv_activation = tf.keras.activations.get(GPMerge_PConv_activation)
    self.PMerge_activation = tf.keras.activations.get(PMerge_activation)
    
    #
    self.compress_ratio = compress_ratio

    #
    self.skip_residual_before_pmerge = skip_residual_before_pmerge

    # ---------------------------
    # PartialMergeLayer
    # ---------------------------
    #
    self.PMergeLyr = PartialMergeLayer(
      self.PMerge_filters,
      self.PMerge_kernel_size,
      data_format=self.PConv_data_format,  # 'channels_last',
      activation=self.PMerge_activation,  # None,
      kernel_initializer=self.PMerge_kernel_initializer,  # k1_init, #tf.keras.initializers.glorot_uniform,
      # pointwise_kernel_initializer=self.PMerge_pointwise_kernel_initializer, #k2_init,
      kernel_regularizer=self.PMerge_kernel_regularizer,
      use_batch_normalization=self.PMerge_use_batch_normalization,  # True,
      use_bias=self.PMerge_use_bias,  # False
      dtype=self._dtype_policy,  # .dtype,
      debug_print=self.debug_print,
      #
      name=self.LayerName_PMerge
    )

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
    
    #img
    img_output = input_shape[0]
    img_output[self.channel_axis] = self.filters
    if self.data_format == "channels_first":
      #H and W are 2, 3
      img_output[2] = img_output[2] / self.strides
      img_output[3] = img_output[3] / self.strides
    else:
      #H and W are 1, 2
      img_output[1] = img_output[1] / self.strides
      img_output[2] = img_output[2] / self.strides

    return [img_output, img_output, img_output, img_output]
  
  def call(self, inputs, training = True):
    
    TargetImage = inputs[0] # [B, H, W, C] or [B, C, H, W]
    TargetMaskImage = inputs[1] # [B, H, W, 1 or C] or [B, 1 or C, H, W]
    RefereceImage = inputs[2] # [B, H, W, C] or [B, C, H, W]

    RefereceMaskImage = tf.ones_like(RefereceImage, dtype=self._compute_dtype, name=self.name + '_RefereceMaskImage')

    TargetImage = tf.cast(TargetImage, dtype=self._compute_dtype)
    TargetMaskImage = tf.cast(TargetMaskImage, dtype=self._compute_dtype)
    RefereceImage = tf.cast(RefereceImage, dtype=self._compute_dtype)
    
    ## corrupted reference image
    #CorruptedReferenceImage = tf.math.multiply(RefereceImage, TargetMaskImage)
    CorruptedReferenceImage = RefereceImage * TargetMaskImage

    # ---------------------------
    # PConv2D
    # ---------------------------
    # Reference Image
    PConv_Ref_img_output, _, PConv_Ref_PConv_output = self.PConv2DLyr(
      [RefereceImage, RefereceMaskImage],
      training=training
      )
    
    # Target Image
    PConv_Tgt_img_output, PConv_Tgt_mask_output, PConv_Tgt_PConv_output = self.PConv2DLyr(
      [TargetImage, TargetMaskImage],
      training=training
      )
    
    # Corrupted Reference Image
    PConv_Ref_corrupted_image_output = None
    PConv_Ref_corrupted_PConv_output = None

    merged_img_output = self.PMergeLyr(
      # [GatedDetailsRefComplete, DetailsTgtCorrupted, updated_mask_B]
      [PConv_Ref_img_output, PConv_Tgt_img_output, PConv_Tgt_mask_output],
      training=training
    )
    
    # order of output
    # 0: Activated Output (Reference) [B, H/S, W/S, OutFeatures] or [B, OutFeatures, H/S, W/S]
    # 1: Partial Merged Output [B, H/S, W/S, OutFeatures] or [B, OutFeatures, H/S, W/S]
    # 2: Updated Mask (Target) [B, H/S, W/S, OutFeatures] or [B, OutFeatures, H/S, W/S]
    # 3: Activated Output (Target) [B, H/S, W/S, OutFeatures] or [B, OutFeatures, H/S, W/S]
    return [PConv_Ref_img_output, merged_img_output, PConv_Tgt_mask_output, PConv_Tgt_img_output]
    return [PConv_Ref_img_output, merged_img_output, PConv_Tgt_mask_output, PConv_Tgt_img_output]

  def get_config(self):
    config = super().get_config()
    config.update({
      'PConv_filters': self.PConv_filters,
      'PConv_kernel_size': self.PConv_kernel_size,
      'PMerge_filters': self.PMerge_filters,
      'PMerge_kernel_size': self.PMerge_kernel_size,
      'data_format': self.data_format,
      'PConv_strides': self.PConv_strides,
      'PConv_padding': self.PConv_padding,
      'PConv_dilation_rate': self.PConv_dilation_rate,
      'PConv_activation': tf.keras.activations.serialize(self.PConv_activation),
      'PConv_use_bias': self.PConv_use_bias,
      'PConv_kernel_initializer': tf.keras.initializers.serialize(self.PConv_kernel_initializer),
      'PConv_kernel_regularizer': tf.keras.regularizers.serialize(self.PConv_kernel_regularizer),
      'PConv_use_batch_normalization': self.PConv_use_batch_normalization,
      'GPMerge_PConv_activation': tf.keras.activations.serialize(self.GPMerge_PConv_activation),
      'PMerge_activation': tf.keras.activations.serialize(self.PMerge_activation),
      'GPMerge_PConv_use_batch_normalization': self.GPMerge_PConv_use_batch_normalization,
      'PMerge_use_batch_normalization': self.PMerge_use_batch_normalization,
      'PMerge_use_bias': self.PMerge_use_bias,
      'PMerge_kernel_initializer': tf.keras.initializers.serialize(self.PMerge_kernel_initializer),
      'PMerge_bias_initializer': tf.keras.initializers.serialize(self.PMerge_bias_initializer),
      'PMerge_kernel_regularizer': tf.keras.regularizers.serialize(self.PMerge_kernel_regularizer),
      'PMerge_bias_regularizer': tf.keras.regularizers.serialize(self.PMerge_bias_regularizer),
      'PMerge_activity_regularizer': tf.keras.regularizers.serialize(self.PMerge_activity_regularizer),
      'PMerge_kernel_constraint': tf.keras.constraints.serialize(self.PMerge_kernel_constraint),
      'PMerge_bias_constraint': tf.keras.constraints.serialize(self.PMerge_bias_constraint),
      'compress_ratio': self.compress_ratio,
      'skip_residual_before_pmerge': self.skip_residual_before_pmerge,
      'dtype': self._dtype_policy, #self.dtype,
      'debug_print': self.debug_print
      })
    return config
    
