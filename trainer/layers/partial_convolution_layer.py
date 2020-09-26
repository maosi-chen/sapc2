from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys

import tensorflow as tf

import tensorflow.keras as K

from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.keras.utils import tf_utils

class PConv2D(K.layers.Layer):

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
    strides=1, #(1,1),
    #
    kernel_initializer='he_normal',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    #
    debug_print=False,
    #
    *args,
    **kwargs
    ):

    super().__init__(*args, **kwargs)
    self.input_spec = [
      tf.keras.layers.InputSpec(ndim=4), 
      tf.keras.layers.InputSpec(ndim=4)
    ]
    self.LayerName = self.name

    #
    self.filters = filters
    self.kernel_size = kernel_size

    #
    self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
    #
    self.use_bias = use_bias
    #
    self.strides = strides
    #
    self.bias_initializer = tf.keras.initializers.get(bias_initializer)
    self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self.bias_constraint = tf.keras.constraints.get(bias_constraint)

    #
    self.use_batch_normalization = use_batch_normalization
    # data_format: 'channels_first' [B, C, H, W] or
    #              'channels_last'  [B, H, W, C]
    self.data_format = data_format
    if self.data_format == 'channels_first':
      self.channel_axis = 1
      self.BN_reduce_axes = [0,2,3]
    else:
      self.channel_axis = -1
      self.BN_reduce_axes = [0,1,2]
      
    self.channel_axis_ts = tf.convert_to_tensor(self.channel_axis, dtype=tf.int32)
    #
    self.activation = tf.keras.activations.get(activation)

    #
    self.debug_print = tf.convert_to_tensor(debug_print, dtype=tf.bool)
    # constant
    self.epsilon = 6e-8 # minimum for float16
    self.epsilon_ts = tf.convert_to_tensor(self.epsilon, dtype=self._compute_dtype)
    
    # Batch normalization layer
    if self.use_batch_normalization:
      # Trainable is true, adjust gamma and beta during training
      # Fused does not need to be set, it defaults to none. If fused can be used, it will be for faster operations
      # Epsilon, sets small value to avoid divide by zero.
      self.batch_normalization_layer = BatchNormalization(
        trainable=True, epsilon=self.epsilon, axis=self.channel_axis,
        dtype=self._dtype_policy
        )
    else:
      self.batch_normalization_layer = None

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    self.input_dim = input_shape[0][self.channel_axis]

    # Image kernel (PConv2D_v2)
    dw_kn_shp = tuple(self.kernel_size) + tuple((self.input_dim, 1))
    dw_kn_shp_0 = tuple(self.kernel_size) + tuple((1, 1))
    print("dw_kn_shp (partial_convolution_layer)", dw_kn_shp)
    self.depthwise_kernel = self.add_weight(
      shape=dw_kn_shp,
      initializer=self.kernel_initializer,
      name=self.name + '_depthwise_kernel',
      regularizer=self.kernel_regularizer,
      constraint=self.kernel_constraint
    )
    pw_kn_shp = (1, 1, self.input_dim, self.filters)
    pw_kn_shp_0 = (1, 1, 1, self.filters)
    self.pointwise_kernel = self.add_weight(
      shape=pw_kn_shp,
      initializer=self.kernel_initializer,
      name=self.name + '_pointwise_kernel',
      regularizer=self.kernel_regularizer,
      constraint=self.kernel_constraint
    )
    # Mask kernel
    self.depthwise_kernel_mask = tf.ones(shape=dw_kn_shp_0, dtype=self._compute_dtype)
    self.pointwise_kernel_mask = tf.ones(shape=pw_kn_shp_0, dtype=self._compute_dtype)
    # Calculate padding size to achieve zero-padding
    self.pconv_padding = (
      (int((self.kernel_size[0] - 1) / 2), int((self.kernel_size[0] - 1) / 2)),
      (int((self.kernel_size[0] - 1) / 2), int((self.kernel_size[0] - 1) / 2)),
    )
    # Window size - used for normalization
    # self.window_size = self.kernel_size[0] * self.kernel_size[1]
    #self.window_size = self.kernel_size[0] * self.kernel_size[1] * self.input_dim
    #self.window_size = tf.cast(self.kernel_size[0] * self.kernel_size[1] * self.input_dim, dtype=tf.float32)

    # bias
    if self.use_bias:
      self.bias = self.add_weight(
        shape=(self.filters,),
        initializer=self.bias_initializer,
        name=self.name + '_bias',
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint
      )
    else:
      self.bias = None

    img_output_shape = list(input_shape[0])
    img_output_shape[self.channel_axis] = self.filters
    mask_padded_shaped = list(input_shape[1])
    if self.data_format == "channels_first":
      #H and W are 2, 3
      img_output_shape[2] = int(img_output_shape[2] / self.strides)
      img_output_shape[3] = int(img_output_shape[3] / self.strides)
      mask_padded_shaped[2] = int(mask_padded_shaped[2] + self.pconv_padding[0][0] + self.pconv_padding[0][1])
      mask_padded_shaped[3] = int(mask_padded_shaped[3] + self.pconv_padding[1][0] + self.pconv_padding[1][1])
    else:
      #H and W are 1, 2
      img_output_shape[1] = int(img_output_shape[1] / self.strides)
      img_output_shape[2] = int(img_output_shape[2] / self.strides)
      mask_padded_shaped[1] = int(mask_padded_shaped[1] + self.pconv_padding[0][0] + self.pconv_padding[0][1])
      mask_padded_shaped[2] = int(mask_padded_shaped[2] + self.pconv_padding[1][0] + self.pconv_padding[1][1])

    self.img_output_shape = img_output_shape

    ## get the dynamic shape of padded masks
    ### value: (None, H+pad, W+pad, fin) or (None, fin, H+pad, W+pad)
    masks_dyn_shp = tf.TensorShape((None,)).concatenate(tf.TensorShape(mask_padded_shaped[1:]))
    masks_TSpec = tf.TensorSpec(shape=masks_dyn_shp, dtype=self._compute_dtype)
    dw_kn_mask_TSpec = tf.TensorSpec(shape=tf.TensorShape(dw_kn_shp_0), dtype=self._compute_dtype)
    pw_kn_mask_TSpec = tf.TensorSpec(shape=tf.TensorShape(pw_kn_shp_0), dtype=self._compute_dtype)
    dw_kn_TSpec = tf.TensorSpec(shape=tf.TensorShape(dw_kn_shp), dtype=self._compute_dtype)
    pw_kn_TSpec = tf.TensorSpec(shape=tf.TensorShape(pw_kn_shp), dtype=self._compute_dtype)
    self.calculate_partial_weighted_mask_ratio_v_padded_ins = \
      self.calculate_partial_weighted_mask_ratio_v_padded.get_concrete_function(
        masks_TSpec,
        dw_kn_mask_TSpec, pw_kn_mask_TSpec,
        dw_kn_TSpec, pw_kn_TSpec
      )
    
    ## get the dynamic shape of img_output, mask_output (same)
    img_output_dyn_shp = tf.TensorShape((None,)).concatenate(tf.TensorShape(img_output_shape[1:]))
    img_output_TSpec = tf.TensorSpec(shape=img_output_dyn_shp, dtype=self.dtype)

    self.Is_full_mask_ins = self.Is_full_mask.get_concrete_function(img_output_TSpec)

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
      "strides": self.strides,
      'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
      'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
      'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
      'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
      'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
      'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
      "dtype": self._dtype_policy,
      "debug_print": bool(self.debug_print.numpy())
    })
    return config

  ## implement this method as required by keras.
  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    #Output of this function must match [img_output, mask_output, PConv_output]
    #input_shape[0] -> image
    #input_shape[1] -> mask
    #img_output = [B, H/S, W/S, Outfeatures]
    #pconv_output = [B, H/S, W/S, Outfeatures]
    #mask_output = [B, H/S, W/S, 1|C] ?? Looks like channels to me
    
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

    #return all outputs in array.
    return [img_output, img_output, img_output]

  @staticmethod
  @tf.function
  def Is_full_mask(tfMask):
    tfMask_min = tf.math.reduce_min(tfMask)
    
    res = tf.keras.backend.switch(
      tf.math.equal(tfMask_min, tf.constant(0, dtype=tfMask_min.dtype)),
      lambda: tf.constant(False, dtype=tf.bool),
      lambda: tf.constant(True, dtype=tf.bool)
    )

    return res

  @tf.function
  def generate_random_masked_values(self, tfInput, tfMask): #, channel_idx):
  
    ## tfInput and tfMask: [B, H, W, C] or [B, C, H, W]
    # set masked values to zero
    tfInput = tfInput * tfMask
    
    # count good pixels
    ## [1, 1, 1, C] or [1, C, 1, 1]
    cnts_good = tf.math.reduce_sum(tf.cast(tfMask, dtype=self._compute_dtype), keepdims=True, axis=self.BN_reduce_axes)
    
    # count bad pixels
    ## [1, 1, 1, C] or [1, C, 1, 1]
    reversedMask = tf.cast(1.-tfMask, dtype=self._compute_dtype)
    cnts_bad  = tf.math.reduce_sum(reversedMask, keepdims=True, axis=self.BN_reduce_axes)
    
    # count all pixels
    ## [1, 1, 1, C] or [1, C, 1, 1]
    cnts_glb = cnts_good + cnts_bad

    # calculate the glb mean, sum, and var of tfInput
    ## [1, 1, 1, C] or [1, C, 1, 1]
    sums_glb = tf.math.reduce_sum(tfInput, keepdims=True, axis=self.BN_reduce_axes)
    vars_glb = tf.math.reduce_variance(tfInput, keepdims=True, axis=self.BN_reduce_axes)
    means_glb = tf.math.reduce_mean(tfInput, keepdims=True, axis=self.BN_reduce_axes)
  
    # calculate the mean of tfInput on good pixels
    ## [1, 1, 1, C] or [1, C, 1, 1]
    means_good = sums_glb / cnts_good
    
    # std_1_pop = sqrt( (N*std_g_pop^2 - N0*mu_g^2 - N1*(mu_1-mu_g)^2) / N1 )
    std_good_pop = tf.math.sqrt(
      (cnts_glb * vars_glb - \
       cnts_bad * tf.math.square(means_glb) - \
       cnts_good * tf.math.square(means_good - means_glb)
      ) / \
      cnts_good
    )
    
    ## [B, H, W, C] or [B, C, H, W]
    random_distribution = tf.random.normal(
      shape = tf.shape(tfInput),
      mean = means_good,
      stddev = std_good_pop,
      dtype = self._compute_dtype #self.dtype
      )
    
    mask_filled = tfInput + random_distribution * reversedMask
    
    return mask_filled

  @tf.function
  def calculate_partial_weighted_mask_ratio_v_padded(
    self,
    masks,
    in_depthwise_kernel_mask, in_pointwise_kernel_mask,
    in_depthwise_kernel, in_pointwise_kernel
    ):
    # ----------------------------
    # count 2
    ## masks_0: [B, H+pad, W+pad, 1] or [B, 1, H+pad, W+pad]
    masks_0 = tf.gather(masks, indices=[0], axis=self.channel_axis)

    # [B, H/S, W/S, fout] or [B, fout, H/S, W/S]
    Cnt2 = K.backend.separable_conv2d(
      masks_0, #masks,
      in_depthwise_kernel_mask,
      in_pointwise_kernel_mask,
      strides=(self.strides, self.strides),
      padding='valid', #'same',
      data_format=self.data_format,
      dilation_rate=(1, 1)
    )

    # ----------------------------
    # the correction ratio:
    #      SUM(|W|.M)
    # r = ------------
    #      SUM(|W|.1)

    # ----------------------------
    # abs of the two kernels
    ## self.depthwise_kernel: [k_x, k_y, fin, 1]
    ## depthwise_kernel_abs: [k_x, k_y, fin, 1]
    depthwise_kernel_abs = tf.cast(tf.math.abs(in_depthwise_kernel), dtype=self._compute_dtype)
    ## self.pointwise_kernel: [1, 1, fin, fout]
    ## pointwise_kernel_abs: [1, 1, fin, fout]
    pointwise_kernel_abs = tf.cast(tf.math.abs(in_pointwise_kernel), dtype=self._compute_dtype)
    
    # ----------------------------
    # energy of the entire window

    ## padded_ones_1: an all-one matrix with shape of padded ones_like_masks
    # [B, H+pad, W+pad, fin] or [B, fin, H+pad, W+pad]
    padded_ones_1 = tf.ones_like(masks, dtype=self._compute_dtype)

    ## [B, H/S, W/S, fout] or [B, fout, H/S, W/S]
    kernel_energy_window_sum = K.backend.separable_conv2d(
      padded_ones_1,
      depthwise_kernel_abs,
      pointwise_kernel_abs,
      strides=(self.strides, self.strides), #(1, 1),
      padding='valid',
      data_format=self.data_format,
      dilation_rate=(1, 1)
    )

    # ----------------------------
    # energy of the masked part
    ## [B, H/S, W/S, fout] or [B, fout, H/S, W/S]
    kernel_energy_weighted_Cnt2 = K.backend.separable_conv2d(
      masks,
      depthwise_kernel_abs, #depthwise_kernel_abs,
      pointwise_kernel_abs, #pointwise_kernel_abs,
      strides=(self.strides, self.strides), #(1, 1),
      padding='valid', #'same',
      data_format=self.data_format,
      dilation_rate=(1, 1)
    )

    # ----------------------------
    # Calculate the mask ratio on each pixel in the output mask
    ## [B, H/S, W/S, fout] or [B, fout, H/S, W/S]
    # flip the numerator and denominator of weighted_mask_ratio to avoid overflow under fp16.
    weighted_mask_ratio = tf.math.truediv(kernel_energy_weighted_Cnt2, kernel_energy_window_sum)
    
    # ----------------------------
    # Clip output to be between 0 and 1
    ## [B, H/S, W/S, fout] or [B, fout, H/S, W/S]
    mask_output = tf.clip_by_value(Cnt2, tf.constant(0, dtype=self._compute_dtype),
                                 tf.constant(1, dtype=self._compute_dtype))
    
    # ----------------------------
    # Remove ratio values where there are holes
    ## [B, H/S, W/S, fout] or [B, fout, H/S, W/S]
    weighted_mask_ratio = tf.math.multiply(weighted_mask_ratio, mask_output, name="weighted_mask_ratio")

    return weighted_mask_ratio, mask_output

  def call(self, inputs, training=True):
    # 
    images = inputs[0]
    masks = inputs[1]    

    images = tf.cast(images, dtype=self._compute_dtype)
    masks  = tf.cast(masks, dtype=self._compute_dtype)

    ## have to get the concrete function of generate_random_masked_values in call
    ## because batch_size is None in Build
    ## If batch_size has been traced before, it won't trigger new trace in future call.
    batch_size = images.get_shape()[0]
    img_output_dyn_shp_fixed = tf.TensorShape((batch_size,)).concatenate(tf.TensorShape(self.img_output_shape[1:]))
    img_output_TSpec_fixed = tf.TensorSpec(shape=img_output_dyn_shp_fixed, dtype=self._compute_dtype)
    self.generate_random_masked_values_ins = self.generate_random_masked_values.get_concrete_function(
      img_output_TSpec_fixed, img_output_TSpec_fixed #, tf.TensorSpec([], dtype=tf.int32)
    )

    # Padding done explicitly so that padding becomes part of the masked partial convolution
    ## K.spatial_2d_padding -> tf.keras.backend.spatial_2d_padding
    # [B, H+pad, W+pad, fin] or [B, fin, H+pad, W+pad]
    images = tf.keras.backend.spatial_2d_padding(images, self.pconv_padding, self.data_format)

    # Padding the masks explicitly (with zeros)
    # [B, H+pad, W+pad, fin] or [B, fin, H+pad, W+pad]
    masks = tf.keras.backend.spatial_2d_padding(masks, self.pconv_padding, self.data_format)

    # ----------------------------
    # count 2
    # ----------------------------
    # the correction ratio:
    #      SUM(|W|.M)
    # r = ------------
    #      SUM(|W|.1)
    ## [B, H/S, W/S, OutFeatures] or [B, OutFeatures, H/S, W/S]
    self.weighted_mask_ratio, mask_output = self.calculate_partial_weighted_mask_ratio_v_padded_ins(
      masks,
      self.depthwise_kernel_mask, self.pointwise_kernel_mask,
      tf.identity(self.depthwise_kernel), tf.identity(self.pointwise_kernel)
    )

    # ----------------------------
    # masked images (element multiplication)
    # [B, H+pad, W+pad, fin] or [B, fin, H+pad, W+pad]
    masked_images = K.layers.multiply([images, masks], dtype=self._dtype_policy)

    # ----------------------------
    # W * (X.M)
    # [B, H/S, W/S, OutFeatures] or [B, OutFeatures, H/S, W/S]
    W_XM = K.backend.separable_conv2d(
      masked_images, 
      self.depthwise_kernel,
      self.pointwise_kernel,
      strides=(self.strides, self.strides), #(1, 1),
      padding='valid', #'same',
      data_format=self.data_format,
      dilation_rate=(1, 1)
    )

    # ----------------------------
    # Normalize iamge output
    # [B, H/S, W/S, OutFeatures] or [B, OutFeatures, H/S, W/S]
    ## note: flip the numerator and denominator of r (self.weighted_mask_ratio) to avoid the overflow under float16
    ##       therefore need to convert from multiply of r to division of r.
    img_output = tf.raw_ops.DivNoNan(x=W_XM, y=self.weighted_mask_ratio)

    # ----------------------------
    # Apply bias only to the image (if chosen to do so)
    # [B, H/S, W/S, OutFeatures] or [B, OutFeatures, H/S, W/S]
    if self.use_bias:
      img_output = K.backend.bias_add(
        img_output,
        self.bias,
        data_format=self.data_format
      )

    # PConv_output (before activation, for merge)
    PConv_output = tf.identity(img_output, name=self.LayerName + '_output')

    # ----------------------------
    # Perform batch normalization (if chosen to do so)
    # [B, H/S, W/S, OutFeatures] or [B, OutFeatures, H/S, W/S]
    if self.use_batch_normalization:
      img_output = tf.keras.backend.switch(
        self.Is_full_mask_ins(mask_output),
        lambda: img_output,
        lambda: self.generate_random_masked_values_ins(img_output, mask_output)
      )

      img_output = self.batch_normalization_layer(img_output, training=training)

    # ----------------------------
    # Apply activations on the image
    if self.activation is not None:
      img_output = self.activation(img_output)

    return [img_output, mask_output, PConv_output]
