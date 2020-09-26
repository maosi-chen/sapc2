from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys

import numpy as np

import tensorflow as tf
#import tensorflow.keras as K

from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.keras.utils import tf_utils

from .partial_convolution_layer import PConv2D
from .partial_merge_layer import PartialMergeLayer

###

class GatedPartialMergeLayer(tf.keras.layers.Layer):
  
  def __init__(
    self,
    PMerge_filters,
    #
    kernel_size_time=1, #2,
    kernel_size_space=(3,3),
    #
    data_format='channels_last',
    # space conv (get details)
    kernel_initializer='he_normal',
    kernel_regularizer=None,
    kernel_constraint=None,
    use_bias=True,
    bias_initializer='zeros',
    bias_regularizer=None,
    bias_constraint=None,
    activity_regularizer=None,
    #
    PConv_use_batch_normalization=True,
    PMerge_use_batch_normalization=False,
    PConv_activation=tf.keras.layers.PReLU(),
    PMerge_activation=None,
    #
    compress_ratio=1.0,
    #
    *args,
    **kwargs
    ):
    super().__init__(*args, **kwargs)
    
    self.input_spec = [
      tf.keras.layers.InputSpec(ndim=4),  # InputImageA (ref, complete, PConv w/ act.)
      tf.keras.layers.InputSpec(ndim=4),  # InputImageB (tgt, corrupted, PConv w/ act.)
      tf.keras.layers.InputSpec(ndim=4),  # InputImageC (ref, corrupted, PConv w/ act.)
      tf.keras.layers.InputSpec(ndim=4)   # MaskImageB
    ]

    #
    self.PMerge_filters = PMerge_filters
    
    #
    self.kernel_size_time = kernel_size_time
    self.kernel_size_space = tuple(kernel_size_space)

    # data_format: 'channels_first' [B, C, H, W] or
    #              'channels_last'  [B, H, W, C]
    self.data_format = data_format
    if self.data_format == 'channels_first':
      self.channel_axis = 1
      self.spatial_dims = [2,3]
    else:
      self.channel_axis = -1
      self.spatial_dims = [1,2]
      
    #
    self.compress_ratio = compress_ratio
    
    # space conv
    self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)

    self.use_bias = use_bias

    self.bias_initializer = tf.keras.initializers.get(bias_initializer)
    self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self.bias_constraint = tf.keras.constraints.get(bias_constraint)

    self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
    
    self.PConv_activation = tf.keras.activations.get(PConv_activation)
    self.PMerge_activation = tf.keras.activations.get(PMerge_activation)
    
    self.PConv_use_batch_normalization = PConv_use_batch_normalization
    self.PMerge_use_batch_normalization = PMerge_use_batch_normalization
    
    #
    self.epsilon = 6e-8
    
    # BNs for time/space difference metrics
    self.BN_layer_TDavg = BatchNormalization(trainable=True, epsilon=self.epsilon, axis=-1, dtype=self._dtype_policy)

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    
    # input_shape should be a list of shapes for three input images:
    # [0] InputImageA
    # [1] InputImageB
    # [2] InputImageC
    # [3] MaskImageB
    self.InputImageA_shape = input_shape[0]
    self.InputImageB_shape = input_shape[1]
    self.InputImageC_shape = input_shape[2]
    self.MaskImageB_shape = input_shape[3]

    self.input_dim_ImageA = self.InputImageA_shape[self.channel_axis]
    self.input_dim_ImageB = self.InputImageB_shape[self.channel_axis]
    self.input_dim = self.input_dim_ImageA + self.input_dim_ImageB
    
    # number of compressed time difference features
    self.compressed_features = max([round(self.input_dim_ImageA * self.compress_ratio), 1])
    
    # number of final merged features
    self.numFMF = self.input_dim_ImageA
    
    # tensor of multiples (to populate the compressed time difference features)
    if self.data_format == 'channels_first':
      self.TDmultiples = tf.constant([1, 1, self.InputImageA_shape[2], self.InputImageA_shape[3]], dtype = tf.int32)
    else:
      self.TDmultiples = tf.constant([1, self.InputImageA_shape[1], self.InputImageA_shape[2], 1], dtype=tf.int32)
    
    # dense layer (for compressed time difference features)
    self.dense_lyr = tf.keras.layers.Dense(
      self.compressed_features,
      activation='sigmoid', #None,
      kernel_initializer='glorot_normal', #'#self.kernel_initializer,
      kernel_regularizer=self.kernel_regularizer,
      kernel_constraint=self.kernel_constraint,
      use_bias=self.use_bias,
      bias_initializer=self.bias_initializer,
      bias_regularizer=self.bias_regularizer,
      bias_constraint=self.bias_constraint,
      activity_regularizer = self.activity_regularizer,
    )
    
    # partial convolution to derive details of the inputs
    self.PConv2DLyr = PConv2D(
      self.compressed_features,
      self.kernel_size_space,
      strides=1,
      data_format = self.data_format,
      activation = self.PConv_activation,
      use_bias = self.use_bias,
      kernel_initializer = self.kernel_initializer, #'he_normal',
      kernel_regularizer = self.kernel_regularizer,
      bias_initializer = self.bias_initializer,
      use_batch_normalization = self.PConv_use_batch_normalization,
      dtype=self._dtype_policy,
      debug_print=False,
      #
      name="GPMerge_PConv"
      )
    
    self.PMergeLyr = PartialMergeLayer(
      self.PMerge_filters,
      self.kernel_size_space,
      data_format=self.data_format, #'channels_last',
      activation=self.PMerge_activation, #None,
      kernel_initializer=self.kernel_initializer, #k1_init, #tf.keras.initializers.glorot_uniform,
      kernel_regularizer=self.kernel_regularizer,
      use_batch_normalization=self.PMerge_use_batch_normalization, #True,
      use_bias=self.use_bias, #False
      dtype = self._dtype_policy, #.dtype,
      debug_print = False,
      #
      name="GPMerge_PMerge"
      )

    self.built = True
  
  def get_config(self):
    
    config = super().get_config()
    config.update({
      "PMerge_filters": self.PMerge_filters,
      "kernel_size_time": self.kernel_size_time, #tuple(self.kernel_size),
      "kernel_size_space": self.kernel_size_space,
      "data_format": self.data_format,
      "compress_ratio": self.compress_ratio,
      #
      'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
      'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
      'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
      'use_bias': self.use_bias,
      'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
      'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
      'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
      'activity_regularizer': tf.keras.regularizers.serialize(self.activity_regularizer),
      #
      'PConv_activation': tf.keras.activations.serialize(self.PConv_activation),
      'PMerge_activation': tf.keras.activations.serialize(self.PMerge_activation),
      'PConv_use_batch_normalization': self.PConv_use_batch_normalization,
      'PMerge_use_batch_normalization': self.PMerge_use_batch_normalization,
    })
    return config
  
  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    output_shape = input_shape
    return output_shape
  
  def call(self, inputs, training=True): #True
    # inputs should be a list of three input images
    # [0] InputImageA (ref, complete, PConv w/ act.)
    # [1] InputImageB (tgt, corrupted, PConv w/ act.)
    # [2] InputImageC (ref, corrupted, PConv w/ act.)
    # [3] MaskImageB (must have the same number of channels as [0],[1],[2])
    ## A, B, C, and Mask should all have the same number of features
    ## [B, H, W, C]
    InputImageA = inputs[0]
    InputImageB = inputs[1]
    InputImageC = inputs[2]
    MaskImageB = inputs[3]
    
    InputImageA = tf.cast(InputImageA, dtype=self._compute_dtype)
    InputImageB = tf.cast(InputImageB, dtype=self._compute_dtype)
    InputImageC = tf.cast(InputImageC, dtype=self._compute_dtype)
    MaskImageB = tf.cast(MaskImageB, dtype=self._compute_dtype)

    # ----------------------------
    MaskImageA = tf.ones_like(InputImageA, dtype=self._compute_dtype, name=self.name + '_MaskImageA')
    
    # ----------------------------
    # mask the tgt and corrupted ref with the mask (mask updated in PConv)
    # ----------------------------
    InputImageB = tf.multiply(InputImageB, MaskImageB)
    InputImageC = tf.multiply(InputImageC, MaskImageB)

    # ----------------------------
    # prepare the time difference related statistics (mu_1 and sigma_1)
    # ----------------------------
    
    ## ----- ##
    ## count the number of good pixels (N_1)
    ## [B, C]
    InputImageBC_good_cnt = tf.math.reduce_sum(MaskImageB, axis=self.spatial_dims) #, keepdims=True
    
    ## ----- ##
    # calculate the mean of AbsDiff(B,C) on the good pixels (unmasked part, where mask value == 1) (mu_1)
    ## mask the tgt and ref_corrupted images to make sure bad pixels values set to 0
    ## [B, H, W, C]
    AbsDiffBC = tf.math.abs(InputImageB - InputImageC)
    AbsDiffBC_masked = tf.multiply(AbsDiffBC, MaskImageB)
    
    ## calculate the reduce sum on the spatial dims
    ## [B, C]
    AbsDiffBC_masked_rs = tf.math.reduce_sum(AbsDiffBC_masked, axis=self.spatial_dims) #keepdims=True
    
    ## calculate the means of absolute differences
    ## [B, C]
    AbsDiffBC_masked_mean = tf.math.divide(AbsDiffBC_masked_rs, InputImageBC_good_cnt)

    # ----------------------------
    # BN of AbsDiffBC_masked_mean
    # ----------------------------
    ## [B, C]
    AbsDiffBC_masked_mean_BN = self.BN_layer_TDavg(AbsDiffBC_masked_mean, training=training)

    ## [B, C*cr]
    TimeDiffCorrCoeff_act_2D = self.dense_lyr(AbsDiffBC_masked_mean_BN)
    
    ## [B, 1, 1, C*cr] or [B, C*cr, 1, 1]
    TimeDiffCorrCoeff_act = tf.expand_dims(
      tf.expand_dims(TimeDiffCorrCoeff_act_2D, axis=self.spatial_dims[0]),
      axis=self.spatial_dims[1])
    
    # ----------------------------
    # derive details of ref (A) for partial merge
    # ----------------------------
    ## [B, H, W, C*cr] or [B, C*cr, H, W]
    DetailsRefComplete, _, _ = self.PConv2DLyr([InputImageA, MaskImageA], training=training)
    
    # ----------------------------
    # derive details of tgt (B) for partial merge
    # ----------------------------
    ## [B, H, W, C*cr] or [B, C*cr, H, W]
    DetailsTgtCorrupted, updated_mask_B, _ = self.PConv2DLyr([InputImageB, MaskImageB], training=training)
    
    # ----------------------------
    # gating the derived details of ref (A)
    # ----------------------------
    ## [B, H, W, C*cr] or [B, C*cr, H, W]
    GatedDetailsRefComplete = tf.math.multiply(TimeDiffCorrCoeff_act, DetailsRefComplete)

    # ----------------------------
    # PMerge
    # ----------------------------
    ## [B, H, W, C] or [B, C, H, W]
    merged_img_output = self.PMergeLyr([GatedDetailsRefComplete, DetailsTgtCorrupted, updated_mask_B], training=training)
    
    return merged_img_output
