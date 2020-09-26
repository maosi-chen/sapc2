from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys

import tensorflow as tf
#import tensorflow.keras as K

from tensorflow.keras.layers import BatchNormalization

class MergeLayer(tf.keras.layers.Layer):

  #Constructor
  def __init__(self,
    #
    filters,
    kernel_size,
    #
    data_format='channels_last',
    #batch_normalization=None,
    activation=None,
    use_batch_normalization=False,
    use_bias=True,
    #
    kernel_initializer='he_normal',#tf.keras.initializers.he_normal(),
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

    self.LayerName = self.name
    #self.trainable

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
    self.bias_constraint = tf.keras.constraints.get(bias_constraint)#
    self.use_batch_normalization = use_batch_normalization

    #
    self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)


    # data_format: 'channels_first' [B, C, H, W] or 
    #              'channels_last'  [B, H, W, C]
    self.data_format = data_format
    #Check to see where the channel axis is, either ifrst or last
    if self.data_format == 'channels_first':
      self.channel_axis = 1
    else:
      self.channel_axis = -1

    #
    self.activation = tf.keras.activations.get(activation)

    #
    self.debug_print = debug_print

    # constant
    self.epsilon = 6e-8 # minimum of float16
    self.epsilon_ts = tf.convert_to_tensor(self.epsilon, dtype=self._compute_dtype)

    #
    #Batch normalization layer
    #Allows the batch_normalization layer to be passed in
    #if self.use_batch_normalization and self.batch_normalization is None:
    if self.use_batch_normalization:
      self.batch_normalization = BatchNormalization(
        trainable=True, epsilon = self.epsilon,
        axis=self.channel_axis,
        dtype=self._dtype_policy #dtype=self.dtype #self._dtype_policy
        )

  def build(self, input_shape):

    if (type(input_shape) == tf.TensorShape):
      input_shape = tuple(input_shape.as_list())

    if (type(input_shape) == list):
      ii_ts = 0
      for i_ts in input_shape:
        if (type(i_ts) == tf.TensorShape):
          input_shape[ii_ts] = tuple(i_ts.as_list())
          ii_ts = ii_ts + 1

    #presumably, input_shapes of images should match?
    self.InputImageA_shape = input_shape[0] 
    self.InputImageB_shape = input_shape[1]

    #Set the number of input_dims to the number of channels in A and B
    self.input_dim_ImageA = self.InputImageA_shape[self.channel_axis]
    self.input_dim_ImageB = self.InputImageB_shape[self.channel_axis]
    self.input_dim = self.input_dim_ImageA + self.input_dim_ImageB

    # Calculate padding size to achieve zero-padding
    self.pconv_padding = (
      (int((self.kernel_size[0]-1)/2), int((self.kernel_size[0]-1)/2)), 
      (int((self.kernel_size[0]-1)/2), int((self.kernel_size[0]-1)/2)), 
    )

    #Create the merge kernel
    # kernel (merge)
    dw_kn_shp = tuple(self.kernel_size) + tuple((self.input_dim, 1))
    print("dw_kn_shp (merge_layer)", dw_kn_shp)
    self.depthwise_kernel = self.add_weight(
      shape = dw_kn_shp,
      initializer=self.kernel_initializer,
      name='depthwise_kernel',
      regularizer=self.kernel_regularizer,
      constraint=self.kernel_constraint
    )

    pw_kn_shp = (1, 1, self.input_dim, self.filters)
    self.pointwise_kernel = self.add_weight(
      shape = pw_kn_shp,
      initializer=self.kernel_initializer,
      name='pointwise_kernel',
      regularizer=self.kernel_regularizer,
      constraint=self.kernel_constraint
    )

    # bias
    if self.use_bias:
      self.bias = self.add_weight(
        shape=(self.filters,),
        initializer=self.bias_initializer,
        name='merge_bias',
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint
      )
    else:
      self.bias = None

    # Window size - used for normalization
    self.window_size = self.kernel_size[0] * self.kernel_size[1] * self.input_dim

    ## get the dynamic shape of self.stacked_masks
    MaskImageA_shape = list(input_shape[0])
    MaskImageB_shape = list(input_shape[1])
    stacked_masks_shape = list(input_shape[0])
    if self.data_format == "channels_first":
      stacked_masks_shape[1] = int(MaskImageA_shape[1] + MaskImageB_shape[1])
    else:
      stacked_masks_shape[3] = int(MaskImageA_shape[3] + MaskImageB_shape[3])
      
    stacked_masks_padded_shape = stacked_masks_shape
    if self.data_format == "channels_first":
      stacked_masks_padded_shape[2] = int(stacked_masks_shape[2]) + int(self.kernel_size[0]-1)
      stacked_masks_padded_shape[3] = int(stacked_masks_shape[3]) + int(self.kernel_size[0]-1)
    else:
      stacked_masks_padded_shape[1] = int(stacked_masks_shape[1]) + int(self.kernel_size[0]-1)
      stacked_masks_padded_shape[2] = int(stacked_masks_shape[2]) + int(self.kernel_size[0]-1)

    stacked_masks_padded_dyn_shp = tf.TensorShape((None,)).concatenate(tf.TensorShape(stacked_masks_padded_shape[1:]))
    stacked_masks_padded_TSpec = tf.TensorSpec(shape=stacked_masks_padded_dyn_shp, dtype=self._compute_dtype)
    dw_kn_TSpec = tf.TensorSpec(shape=tf.TensorShape(dw_kn_shp), dtype=self._compute_dtype)
    pw_kn_TSpec = tf.TensorSpec(shape=tf.TensorShape(pw_kn_shp), dtype=self._compute_dtype)
    self.calculate_weighted_mask_ratio_ins = self.calculate_weighted_mask_ratio.get_concrete_function(
      stacked_masks_padded_TSpec,
      dw_kn_TSpec, pw_kn_TSpec
    )
    
    self.built = True

  ## implement this method as required by keras.
  def compute_output_shape(self, input_shape):
    #Takes the same input shape as build, and calculates the output shape based on that
    #B H and W are the same as either input image. 
    #However Outfeatures is specified in the constructor, so we utilize that 
    
    #Set the output and leave B, H and W alone
    output_shape = input_shape[0]
    
    #Change the number of channels to n_outputs
    output_shape[self.channel_axis] = self.OutFeatures

    return tf.TensorShape(output_shape)

  @tf.function
  def calculate_weighted_mask_ratio(
    self,
    stacked_masks_padded,
    in_depthwise_kernel, in_pointwise_kernel
    ):
    #----------------------------
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
      padded_ones_1,
      depthwise_kernel_abs,
      pointwise_kernel_abs, #self.kernel_energy_pointwise_kernel,
      strides=(1,1),
      padding='valid',
      data_format=self.data_format,
      dilation_rate=(1,1)
    )        

    #----------------------------
    # energy of the masked part
    ## [B, H, W, fout] or [B, fout, H, W]
    kernel_energy_weighted_Cnt2 = tf.keras.backend.separable_conv2d(
      stacked_masks_padded,
      depthwise_kernel_abs,
      pointwise_kernel_abs,
      strides=(1,1),
      padding='valid', #'same',
      data_format=self.data_format,
      dilation_rate=(1,1)
    )        

    #----------------------------
    # Calculate the mask ratio on each pixel in the output mask
    ## [B, H, W, fout] or [B, fout, H, W]
    # flip the numerator and denominator of weighted_mask_ratio to avoid overflow under fp16.
    weighted_mask_ratio = tf.math.truediv(kernel_energy_weighted_Cnt2, kernel_energy_window_sum)

    return weighted_mask_ratio

  def call(self, inputs, training = None): #True

    # inputs should be a list of three input images
    # [0] InputImageA
    # [1] InputImageB
    InputImageA = inputs[0]
    InputImageB = inputs[1]

    InputImageA = tf.cast(InputImageA, dtype=self._compute_dtype)
    InputImageB = tf.cast(InputImageB, dtype=self._compute_dtype)

    # stack the images into one
    stacked_images = tf.keras.backend.concatenate([InputImageA, InputImageB], axis=self.channel_axis)

    MaskImageA = tf.ones_like(InputImageA, dtype=self._compute_dtype, name=self.name + '_MaskImageA')
    MaskImageB = tf.ones_like(InputImageB, dtype=self._compute_dtype, name=self.name + '_MaskImageB')
    stacked_masks = tf.keras.backend.concatenate([MaskImageA, MaskImageB], axis=self.channel_axis)
    
    ## [B, H+pad, W+pad, fin] or [B, fin, H+pad, W+pad]
    stacked_masks_padded = tf.keras.backend.spatial_2d_padding(
      stacked_masks,
      self.pconv_padding,
      self.data_format
    )

    #----------------------------
    # the correction ratio:
    #      SUM(|W|.M)
    # r = ------------
    #      SUM(|W|.1)
    ## [B, H, W, OutFeatures] or [B, OutFeatures, H, W]
    self.weighted_mask_ratio = self.calculate_weighted_mask_ratio_ins(
      stacked_masks_padded,
      tf.identity(self.depthwise_kernel), tf.identity(self.pointwise_kernel)
    )

    ## [B, H, W, CA+CB] or [B, CA+CB, H, W]
    masked_stacked_image = tf.keras.layers.multiply([stacked_images, stacked_masks], dtype=self._dtype_policy)
    ## [B, H+pad, W+pad, CA+CB] or [B, CA+CB, H+pad, W+pad]
    masked_stacked_image_padded = tf.keras.backend.spatial_2d_padding(
      masked_stacked_image, self.pconv_padding, self.data_format
    )

    #Then, convolve the two layers to get the merged image. want to merge to outputfeatures
    ## [B, H, W, OutFeatures] or [B, OutFeatures, H, W]
    merged_images = tf.keras.backend.separable_conv2d(
      masked_stacked_image_padded, #masked_stacked_image,
      self.depthwise_kernel,
      self.pointwise_kernel,
      strides=(1,1),
      padding='valid', #'same',
      data_format=self.data_format,
      dilation_rate=(1,1)
    )

    ## note: flip the numerator and denominator of r (self.weighted_mask_ratio) to avoid the overflow under float16
    ##       therefore need to convert from multiply of r to division of r.
    merged_images = tf.raw_ops.DivNoNan(x=merged_images, y=self.weighted_mask_ratio)

    #If there we are using bias, do this
    # Apply bias only to the image (if chosen to do so)
    if self.use_bias:
      merged_images = tf.keras.backend.bias_add(
        merged_images,
        self.bias,
        data_format=self.data_format
      )

    #----------------------------
    if self.use_batch_normalization:
      merged_images = self.batch_normalization(merged_images,
                                               training=training)  # <--need to regularize typing

    if self.activation is not None:
      merged_images = self.activation(merged_images)

    return merged_images

  def get_config(self):
    config = super().get_config()
    config.update({
      'filters': self.filters,
      'kernel_size': self.kernel_size,
      'data_format': self.data_format,
      'activation': tf.keras.activations.serialize(self.activation),
      'use_batch_normalization': self.use_batch_normalization,
      'use_bias': self.use_bias,
      'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
      'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
      'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
      'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
      'activity_regularizer': tf.keras.regularizers.serialize(self.activity_regularizer),
      'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
      'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
      'dtype': self._dtype_policy, #self.dtype,
      'debug_print': self.debug_print
      })
    return config
    
