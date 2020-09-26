from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys

import numpy as np
import tensorflow as tf
#import tensorflow.keras as K

from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.keras.utils import tf_utils

class PartialMergeLayer(tf.keras.layers.Layer):

  def __init__(
    self,
    #
    filters,
    kernel_size,
    #
    data_format='channels_last',
    activation=None,
    use_batch_normalization=False,
    use_bias=True,
    #
    kernel_initializer='he_normal',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    #
    dtype="float32",
    debug_print=False,
    #
    *args,
    **kwargs
    ):
    super().__init__(dtype=dtype, *args, **kwargs)

    self.input_spec = [
        tf.keras.layers.InputSpec(ndim=4),
        tf.keras.layers.InputSpec(ndim=4),
        tf.keras.layers.InputSpec(ndim=4)
    ]

    self.LayerName = self.name
    # self.trainable

    #
    self.filters = filters
    self.OutFeatures = filters

    #
    self.kernel_size = kernel_size
    self.KernelSize = kernel_size

    #
    self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)

    #
    self.use_bias = use_bias

    #
    self.bias_initializer = tf.keras.initializers.get(bias_initializer)
    self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self.bias_constraint = tf.keras.constraints.get(bias_constraint)  #

    #
    self.use_batch_normalization = use_batch_normalization

    #
    self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

    # data_format: 'channels_first' [B, C, H, W] or
    #              'channels_last'  [B, H, W, C]
    self.data_format = data_format
    if self.data_format == 'channels_first':
      self.channel_axis = 1
    else:
      self.channel_axis = -1

    #
    self.activation = tf.keras.activations.get(activation)

    #
    self.debug_print = debug_print

    # constant
    self.epsilon = 6e-8
    self.epsilon_ts = tf.convert_to_tensor(self.epsilon, dtype=self._compute_dtype)

    # Batch normalization layer
    if self.use_batch_normalization:
      # Trainable is true, adjust gamma and beta during training
      # Fused does not need to be set, it defaults to none. If fused can be used, it will be for faster operations
      # Epsilon, sets small value to avoid divide by zero.
      self.batch_normalization_layer = BatchNormalization(
        trainable=True, epsilon=self.epsilon,
        axis=self.channel_axis,
        dtype=self._dtype_policy)

  @tf_utils.shape_type_conversion
  def build(self, input_shape):

    # input_shape should be a list of shapes for three input images:
    # [0] InputImageA
    # [1] InputImageB
    # [2] MaskImageB
    self.InputImageA_shape = input_shape[0]  # tf.keras.backend.shape(InputImageA)
    self.InputImageB_shape = input_shape[1]  # tf.keras.backend.shape(InputImageB)
    self.MaskImageB_shape = input_shape[2]  # tf.keras.backend.shape(MaskImageB)

    self.input_dim_ImageA = self.InputImageA_shape[self.channel_axis]
    self.input_dim_ImageB = self.InputImageB_shape[self.channel_axis]
    self.input_dim = self.input_dim_ImageA + self.input_dim_ImageB

    # Calculate padding size to achieve zero-padding
    self.pconv_padding = (
      (int((self.kernel_size[0] - 1) / 2), int((self.kernel_size[0] - 1) / 2)),
      (int((self.kernel_size[0] - 1) / 2), int((self.kernel_size[0] - 1) / 2)),
    )

    dw_kn_shp = tuple(self.kernel_size) + tuple((self.input_dim, 1))
    print("dw_kn_shp (partial_merge_layer)", dw_kn_shp)
    self.depthwise_kernel = self.add_weight(
      shape=dw_kn_shp,
      initializer=self.kernel_initializer,
      name=self.name + '_depthwise_kernel',
      regularizer=self.kernel_regularizer,
      constraint=self.kernel_constraint
    )
    pw_kn_shp = (1, 1, self.input_dim, self.filters)
    self.pointwise_kernel = self.add_weight(
      shape=pw_kn_shp,
      initializer=self.kernel_initializer,
      name=self.name + '_pointwise_kernel',
      regularizer=self.kernel_regularizer,
      constraint=self.kernel_constraint
    )

    # bias
    if self.use_bias:
      self.bias = self.add_weight(
        shape=(self.filters,),
        #dtype=self._compute_dtype, #self.dtype,
        initializer=self.bias_initializer,
        name=self.name + '_merge_bias',
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint,
        #experimental_autocast=False
      )
    else:
      self.bias = None

    # Window size - used for normalization
    self.window_size = self.kernel_size[0] * self.kernel_size[1] * self.input_dim

    ## get the dynamic shape of self.stacked_masks
    InputImageA_shape = list(input_shape[0])
    MaskImageB_shape = list(input_shape[2])
    stacked_masks_shape = list(input_shape[0])
    if self.data_format == "channels_first":
      stacked_masks_shape[1] = int(InputImageA_shape[1] + MaskImageB_shape[1])
    else:
      stacked_masks_shape[3] = int(InputImageA_shape[3] + MaskImageB_shape[3])
      
    stacked_masks_padded_shape = stacked_masks_shape
    if self.data_format == "channels_first":
      stacked_masks_padded_shape[2] = int(stacked_masks_shape[2]) + int(self.kernel_size[0]-1)
      stacked_masks_padded_shape[3] = int(stacked_masks_shape[3]) + int(self.kernel_size[0]-1)
    else:
      stacked_masks_padded_shape[1] = int(stacked_masks_shape[1]) + int(self.kernel_size[0]-1)
      stacked_masks_padded_shape[2] = int(stacked_masks_shape[2]) + int(self.kernel_size[0]-1)
    
    stacked_masks_padded_dyn_shp = tf.TensorShape((None,)).concatenate(tf.TensorShape(stacked_masks_padded_shape[1:]))
    stacked_masks_padded_TSpec = tf.TensorSpec(shape=stacked_masks_padded_dyn_shp, dtype=self._compute_dtype)
    self.calc_merge_weight_ins = self.calc_merge_weight.get_concrete_function(stacked_masks_padded_TSpec)

    dw_kn_TSpec = tf.TensorSpec(shape=tf.TensorShape(dw_kn_shp), dtype=self._compute_dtype)
    pw_kn_TSpec = tf.TensorSpec(shape=tf.TensorShape(pw_kn_shp), dtype=self._compute_dtype)
    self.calculate_partial_weighted_mask_ratio_ins = self.calculate_partial_weighted_mask_ratio.get_concrete_function(
      stacked_masks_padded_TSpec, stacked_masks_padded_TSpec,
      dw_kn_TSpec, pw_kn_TSpec
    )

    self.built = True

  def get_config(self):

    config = super().get_config()
    config.update({
      "filters": self.filters,
      "kernel_size": tuple(self.kernel_size),
      "data_format": self.data_format,
      "activation": tf.keras.activations.serialize(self.activation),
      "use_batch_normalization": self.use_batch_normalization,
      "use_bias": self.use_bias,
      'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
      'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
      'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
      'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
      'activity_regularizer': tf.keras.regularizers.serialize(self.activity_regularizer),
      'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
      'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
      "dtype": self._dtype_policy, #self.dtype,
      "debug_print": self.debug_print
    })
    return config


  ## implement this method as required by keras.
  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    # Takes the same input shape as build, and calculates the output shape based on that
    # B H and W are the same as either input image.
    # However Outfeatures is specified in the constructor, so we utilize that

    # Set the output and leave B, H and W alone
    output_shape = input_shape[0]

    # Change the number of channels to n_outputs
    output_shape[self.channel_axis] = self.OutFeatures

    return output_shape

  @tf.function
  def calculate_partial_weighted_mask_ratio(self, stacked_masks_padded, MergeWeight_padded,
                                            in_depthwise_kernel, in_pointwise_kernel):

    # ----------------------------
    # energy of the entire window

    ## self.depthwise_kernel: [k_x, k_y, fin, 1]
    ## depthwise_kernel_abs: [k_x, k_y, fin, 1]
    depthwise_kernel_abs = tf.cast(tf.math.abs(in_depthwise_kernel), dtype=self._compute_dtype)
    ## self.pointwise_kernel: [1, 1, fin, fout]
    ## pointwise_kernel_abs: [1, 1, fin, fout]
    pointwise_kernel_abs = tf.cast(tf.math.abs(in_pointwise_kernel), dtype=self._compute_dtype)

    # padded_ones_1: an all-one matrix with shape of stacked_masks_padded
    padded_ones_1 = tf.ones_like(stacked_masks_padded, dtype=self._compute_dtype)

    ## [B, H, W, fout] or [B, fout, H, W]
    kernel_energy_window_sum = tf.keras.backend.separable_conv2d(
      padded_ones_1, #self.padded_ones_1,
      # / self.input_dim, # no need to "/ self.input_dim" b/c MergeWeight already * self.input_dim
      depthwise_kernel_abs,
      pointwise_kernel_abs,
      strides=(1, 1),
      padding='valid',
      data_format=self.data_format,
      dilation_rate=(1, 1)
    )

    # ----------------------------
    # energy of the masked part

    ## [B, H+pad, W+pad, fin] or [B, fin, H+pad, W+pad]
    energy_weighted_inputs = tf.math.multiply(x=stacked_masks_padded, y=MergeWeight_padded)
    
    ## [B, H, W, fout] or [B, fout, H, W]
    kernel_energy_weighted_Cnt2 = tf.keras.backend.separable_conv2d(
      energy_weighted_inputs,
      depthwise_kernel_abs,
      pointwise_kernel_abs,
      strides=(1, 1),
      padding='valid', #'same',
      data_format=self.data_format,
      dilation_rate=(1, 1)
    )

    # ----------------------------
    # Calculate the mask ratio on each pixel in the output mask
    ## [B, H, W, fout] or [B, fout, H, W]
    # flip the numerator and denominator of weighted_mask_ratio to avoid overflow under fp16.
    weighted_mask_ratio = tf.math.truediv(kernel_energy_weighted_Cnt2, kernel_energy_window_sum)

    return weighted_mask_ratio

  # MergeWeight is the relative importance of the f^th feature compared to other features at the same B, H, W
  # its values have been rescaled so that the value of 1.0 means the feature has the average importance
  @tf.function
  def calc_merge_weight(self, stacked_masks_padded):
    ## stacked_masks: [B, H, W, CA+CB] or [B, CA+CB, H, W]
    # Cnt1 = stacked_masks

    # Tot_Cnt1
    # ## [B, H, W, 1] or [B, 1, H, W]
    ## [B, H+pad, W+pad, 1] or [B, 1, H+pad, W+pad]
    Tot_Cnt1 = tf.keras.backend.sum(
      stacked_masks_padded, #stacked_masks,
      axis=self.channel_axis,
      keepdims=True
    )

    # MergeWeight
    ## [B, H+pad, W+pad, CA+CB] or [B, CA+CB, H+pad, W+pad]
    MergeWeight = tf.raw_ops.DivNoNan(
      x=stacked_masks_padded,
      y=Tot_Cnt1,
      name="MergeWeight_scale_to_1"
    )

    MergeWeight = tf.math.multiply(MergeWeight, tf.cast(self.input_dim, self._compute_dtype), name="MergeWeight")

    return MergeWeight

  def call(self, inputs, training=True):
    # inputs should be a list of three input images
    # [0] InputImageA
    # [1] InputImageB
    # [2] MaskImageB
    InputImageA = inputs[0]
    InputImageB = inputs[1]
    MaskImageB = inputs[2]

    InputImageA = tf.cast(InputImageA, dtype=self._compute_dtype)
    InputImageB = tf.cast(InputImageB, dtype=self._compute_dtype)
    MaskImageB = tf.cast(MaskImageB, dtype=self._compute_dtype)

    # create the MaskImageA, same size as of InputImageA, all ones
    MaskImageA = tf.ones_like(InputImageA, dtype=self._compute_dtype, name=self.name + '_MaskImageA')

    # stack the two masks
    stacked_masks = tf.keras.backend.concatenate([MaskImageA, MaskImageB], axis=self.channel_axis)
    stacked_masks_padded = tf.keras.backend.spatial_2d_padding(
      stacked_masks, self.pconv_padding, self.data_format
    )

    # stack the two images
    stacked_images = tf.keras.backend.concatenate([InputImageA, InputImageB], axis=self.channel_axis)
    stacked_images_padded = tf.keras.backend.spatial_2d_padding(
      stacked_images, self.pconv_padding, self.data_format
    )

    # ----------------------------
    # count 1
    # ----------------------------
    # MergeWeight (padded)
    ## [B, H+pad, W+pad, CA+CB] or [B, CA+CB, H+pad, W+pad]
    MergeWeight_padded = self.calc_merge_weight_ins(stacked_masks_padded)

    # ----------------------------
    # count 2
    # ----------------------------
    # the correction ratio:
    #      SUM(|W|.t.M)
    # r = ------------
    #      SUM(|W|.t.1)
    ## [B, H, W, OutFeatures] or [B, OutFeatures, H, W]
    weighted_mask_ratio = self.calculate_partial_weighted_mask_ratio_ins(
      stacked_masks_padded, MergeWeight_padded,
      tf.identity(self.depthwise_kernel), tf.identity(self.pointwise_kernel)
    )

    # ----------------------------
    # masked stacked images (element multiplication)
    masked_stacked_images_padded = tf.keras.layers.multiply(
      [stacked_images_padded, stacked_masks_padded], dtype=self._dtype_policy)

    # ----------------------------
    # FeatureWeighted_masked_stacked_images (element multiplication)
    ## [B, H+pad, W+pad, CA+CB] or [B, CA+CB, H+pad, W+pad]
    tXM_padded = tf.math.multiply(x=masked_stacked_images_padded, y=MergeWeight_padded)

    # tf.print("tXM: {}".format(tXM.shape))

    # ----------------------------
    # W * (t.(X.M))
    # [B, H, W, OutFeatures] or [B, OutFeatures, H, W]
    W_tXM = tf.keras.backend.separable_conv2d(
      tXM_padded, #tXM,
      self.depthwise_kernel,
      self.pointwise_kernel,
      strides=(1, 1),
      padding='valid', #'same',
      data_format=self.data_format,
      dilation_rate=(1, 1)
    )

    # ----------------------------
    # Normalize iamge output
    # [B, H, W, OutFeatures] or [B, OutFeatures, H, W]
    merged_img_output = tf.math.truediv(x=W_tXM, y=weighted_mask_ratio)

    # ----------------------------
    # Apply bias only to the image (if chosen to do so)
    if self.use_bias:
      merged_img_output = tf.keras.backend.bias_add(
        merged_img_output,
        self.bias,
        data_format=self.data_format
      )
      
    # ----------------------------
    # Perform batch normalization (if chosen to do so)
    if self.use_batch_normalization:
      merged_img_output = self.batch_normalization_layer(merged_img_output, training=training)

    # ----------------------------
    # Apply activations on the image
    if self.activation is not None:
      merged_img_output = self.activation(merged_img_output)

    return merged_img_output
