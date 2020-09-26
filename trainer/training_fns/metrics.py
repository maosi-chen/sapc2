import tensorflow as tf
import tensorflow_probability as tfp
#import six
from ..utils.tf_utility import sobel_edges_tfpad

from ..parameters.config import DNN_PARAMS

@tf.function(input_signature=[
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
])
def calc_unmasked_MSE_local(y_true_stacked, y_pred_stacked):
  # y_true_stacked: [B, H, W, 2]
  ## 0: y_true:     [B, H, W]
  ## 1: mask:       [B, H, W]
  # y_pred_stacked: [B, H, W, 2]
  ## 0: y_pred:     [B, H, W]
  ## 1: ref:        [B, H, W]
  y_true = y_true_stacked[:, :, :, 0]
  mask = y_true_stacked[:, :, :, 1]
  y_pred = y_pred_stacked[:, :, :, 0]

  ###
  ## [B, H, W]
  mask = tf.cast(mask, dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
  y_true = y_true * mask
  y_pred = y_pred * mask

  ## []
  unmasked_count = tf.math.reduce_sum(mask)
  #batch_size_ts = tf.cast(tf.shape(y_true)[0], DNN_PARAMS['custom_dtype']._variable_dtype)  # Bi

  ## []
  unmasked_SSE_ts = tf.math.reduce_sum(
    tf.math.square(y_pred - y_true)
  )
  ###

  unmasked_MSE_ts = tf.math.divide(unmasked_SSE_ts, unmasked_count)
  
  return unmasked_MSE_ts

@tf.function(input_signature=[
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
])
def calc_masked_MSE_local(y_true_stacked, y_pred_stacked):
  # y_true_stacked: [B, H, W, 2]
  ## 0: y_true:     [B, H, W]
  ## 1: mask:       [B, H, W]
  # y_pred_stacked: [B, H, W, 2]
  ## 0: y_pred:     [B, H, W]
  ## 1: ref:       [B, H, W]
  y_true = y_true_stacked[:, :, :, 0]
  mask = y_true_stacked[:, :, :, 1]
  y_pred = y_pred_stacked[:, :, :, 0]
  
  ###
  ## [B, H, W]
  revMask = tf.constant(1, dtype=DNN_PARAMS['custom_dtype']._variable_dtype) - mask
  masked_y_true = y_true * revMask
  masked_y_pred = y_pred * revMask

  ## []
  masked_count = tf.math.reduce_sum(revMask)
  #batch_size_ts = tf.cast(tf.shape(y_true)[0], DNN_PARAMS['custom_dtype']._variable_dtype)  # Bi

  ## []
  masked_SSE_ts = tf.math.reduce_sum(
    tf.math.square(masked_y_pred - masked_y_true)
  )
  ###
  
  masked_MSE_ts = tf.math.divide(masked_SSE_ts, masked_count)

  return masked_MSE_ts

@tf.function(input_signature=[
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
])
def calc_avg_masked_unmasked_MSE_local(y_true_stacked, y_pred_stacked):
  # y_true_stacked: [B, H, W, 2]
  ## 0: y_true:     [B, H, W]
  ## 1: mask:       [B, H, W]
  # y_pred_stacked: [B, H, W, 2]
  ## 0: y_pred:     [B, H, W]
  ## 1: ref:       [B, H, W]
  masked_MSE_ts = calc_masked_MSE_local(y_true_stacked, y_pred_stacked)
  unmasked_MSE_ts = calc_unmasked_MSE_local(y_true_stacked, y_pred_stacked)
  avg_masked_unmasked_MSE_ts = (masked_MSE_ts + unmasked_MSE_ts) / tf.constant(2.0, dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
  return avg_masked_unmasked_MSE_ts

@tf.function(input_signature=[
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
])
def calc_unweighted_MSE_local(y_true_stacked, y_pred_stacked):
  # y_true_stacked: [B, H, W, 2]
  ## 0: y_true:     [B, H, W]
  ## 1: mask:       [B, H, W]
  # y_pred_stacked: [B, H, W, 2]
  ## 0: y_pred:     [B, H, W]
  ## 1: ref:        [B, H, W]
  y_true = y_true_stacked[:, :, :, 0]
  y_pred = y_pred_stacked[:, :, :, 0]
  
  unweighted_SSE_ts = tf.math.reduce_sum(
    tf.math.square(y_pred - y_true)
  )
  
  pixel_counts = tf.cast(tf.size(y_true), DNN_PARAMS['custom_dtype']._variable_dtype)  # Bi * 64 * 64

  unweighted_MSE_ts = tf.math.divide(unweighted_SSE_ts, pixel_counts)

  return unweighted_MSE_ts

@tf.function(input_signature=[
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
])
def calc_orig_out_img_edge_unmasked_MSE_local(y_true_stacked, y_pred_stacked):
  # y_true_stacked: [B, H, W, 2]
  ## 0: y_true:     [B, H, W]
  ## 1: mask:       [B, H, W]
  # y_pred_stacked: [B, H, W, 2]
  ## 0: y_pred:     [B, H, W]
  ## 1: ref:        [B, H, W]
  
  y_true = y_true_stacked[:, :, :, 0]
  mask = y_true_stacked[:, :, :, 1]
  y_pred = y_pred_stacked[:, :, :, 0]
  
  ###
  ## [B, H, W]
  mask = tf.cast(mask, dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
  ## [B, H, W, 1, 1]
  mask_5D = tf.expand_dims(tf.expand_dims(mask, axis=-1), axis=-1)
  # ## [B, H, W, 2, 2]
  # mask_5D = tf.tile(mask_5D, [1,1,1,2,2])

  # [B, H, W, 2]
  y_stacked = tf.stack([y_true, y_pred], axis=-1)

  # [B, H, W, 2, 2]
  #edges = tf.image.sobel_edges(y_stacked)
  edges = sobel_edges_tfpad(y_stacked)
  edges = edges * mask_5D

  # [] # Bi * 64 * 64 * fi * 2 (last 2: y&x, fi: fraction of good (1) pixels)
  ## [DONE] _TODO: remove the * 2 part and set hyperparameter edge_weight /= 2
  unmasked_count = tf.math.reduce_sum(mask) #* tf.constant(2.0, dtype=DNN_PARAMS['custom_dtype']._variable_dtype)

  # []
  edges_unmasked_SSE_ts = tf.math.reduce_sum(
    tf.math.square(edges[:, :, :, 0, :] - edges[:, :, :, 1, :])
  ) / tf.constant(2.0, dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
  ###
  
  edges_unmasked_MSE_ts = tf.math.divide(edges_unmasked_SSE_ts, unmasked_count)
  
  return edges_unmasked_MSE_ts

@tf.function(input_signature=[
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2],
                dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2],
                dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
])
def calc_orig_out_img_edge_masked_MSE_local(y_true_stacked, y_pred_stacked):
  # y_true_stacked: [B, H, W, 2]
  ## 0: y_true:     [B, H, W]
  ## 1: mask:       [B, H, W]
  # y_pred_stacked: [B, H, W, 2]
  ## 0: y_pred:     [B, H, W]
  ## 1: ref:        [B, H, W]
  
  # [B, H, W]
  y_true = y_true_stacked[:, :, :, 0]
  mask = y_true_stacked[:, :, :, 1]
  y_pred = y_pred_stacked[:, :, :, 0]
  ref = y_pred_stacked[:, :, :, 1]
  
  ###
  ## [B, H, W]
  reversedMask = tf.constant(1, dtype=DNN_PARAMS['custom_dtype']._variable_dtype) - mask

  ## [B, H, W, 1, 1]
  reversedMask_5D = tf.expand_dims(tf.expand_dims(reversedMask, axis=-1), axis=-1)

  ## [] # B * H * W * (1-fi)
  masked_count = tf.math.reduce_sum(reversedMask)
  ###

  batch_size_ts = tf.cast(tf.shape(y_true)[0], DNN_PARAMS['custom_dtype']._variable_dtype)  # Bi
  
  # reshape to 2D [B, H*W]
  shp_2D = (tf.cast(batch_size_ts, tf.int32), tf.constant(-1, dtype=tf.int32))
  ref_2D = tf.reshape(ref, shp_2D)
  y_true_2D = tf.reshape(y_true, shp_2D)

  # correlation coefficient between ref and y_true (complete target)
  ## [B]
  corr_1D = tfp.stats.correlation(ref_2D, y_true_2D, sample_axis=1, event_axis=None)
  
  abs_corr_1D = tf.cast(tf.math.abs(corr_1D), dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
  
  # mosaic the unmasked part of y_true and masked part of output.
  ## [B, H, W]
  # mosaic_unmsk_true_msk_pred = y_true * mask + y_pred * (
  #   tf.constant(1.0, dtype=DNN_PARAMS['custom_dtype']._variable_dtype) - mask)
  
  # calculate the edge loss between the mosaic image and y_true
  ## the unmasked part's edge loss can cancel out, therefore don't need to do the calculation on masked area.
  ## instead do the calculation on the entire image.
  # [B, H, W, 2]
  #y_stacked = tf.stack([y_true, mosaic_unmsk_true_msk_pred], axis=-1)
  y_stacked = tf.stack([y_true, y_pred], axis=-1)
  
  # [B, H, W, 2, 2]
  edges = sobel_edges_tfpad(y_stacked) * tf.math.sqrt(
    tf.reshape(abs_corr_1D, tf.constant((-1, 1, 1, 1, 1), dtype=tf.int32)))
  edges = edges * reversedMask_5D

  # [B, H, W,    2]
  edges_masked_SSE_ts = tf.math.reduce_sum(
    tf.math.square(edges[:, :, :, 0, :] - edges[:, :, :, 1, :])
  ) / tf.constant(2.0, dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
  
  #
  edges_masked_MSE_ts = tf.math.divide(edges_masked_SSE_ts, masked_count)
  
  return edges_masked_MSE_ts


# class Metric_Mean_wrapper(tf.keras.metrics.Mean):
#   def __init__(self, fn, weight=1.0, name=None, dtype="float32", **kwargs):
#     super().__init__(name=name, dtype=dtype)
#     self._fn = fn
#     self._fn_kwargs = kwargs
#     self.weight = tf.convert_to_tensor(weight, dtype=dtype)
#
#   def update_state(self, y_true, y_pred, sample_weight=None):
#     # (y_true_stacked, y_pred_stacked)
#     y_true = tf.cast(y_true, self.dtype)
#     y_pred = tf.cast(y_pred, self.dtype)
#
#     weighted_metric = self._fn(y_true, y_pred, **self._fn_kwargs) * self.weight
#
#     return super().update_state(
#       weighted_metric, sample_weight=sample_weight)
#
#
#   def get_config(self):
#     config = {}
#     for k, v in six.iteritems(self._fn_kwargs):
#       config[k] = K.eval(v) if is_tensor_or_variable(v) else v
#     base_config = super().get_config()
#     base_config.update(config)
#     base_config.update({'weight': tf.keras.backend.eval(self.weight)})
#     return base_config
#
# class Metric_unmasked_MSE(Metric_Mean_wrapper):
#   def __init__(self, weight=1.0, name='Metric_unmasked_MSE', dtype='float32'):
#     super().__init__(fn=calc_unmasked_MSE_local, weight=weight, name=name, dtype=dtype)
#
# class Metric_masked_MSE(Metric_Mean_wrapper):
#   def __init__(self, weight=1.0, name='Metric_masked_MSE', dtype='float32'):
#     super().__init__(fn=calc_masked_MSE_local, weight=weight, name=name, dtype=dtype)
#
# class Metric_unweighted_MSE(Metric_Mean_wrapper):
#   def __init__(self, weight=1.0, name='Metric_unweighted_MSE', dtype='float32'):
#     super().__init__(fn=calc_unweighted_MSE_local, weight=weight, name=name, dtype=dtype)
#
# class Metric_orig_out_img_edge_unmasked_MSE(Metric_Mean_wrapper):
#   def __init__(self, weight=1.0, name='orig_out_img_edge_unmasked_MSE', dtype='float32'):
#     super().__init__(fn=calc_orig_out_img_edge_unmasked_MSE_local, weight=weight, name=name, dtype=dtype)
#
#
# class Metric_mosaic_out_img_edge_masked_MSE(Metric_Mean_wrapper):
#   def __init__(self, weight=1.0, name='mosaic_out_img_edge_masked_MSE', dtype='float32'):
#     super().__init__(fn=calc_mosaic_out_img_edge_masked_MSE_local, weight=weight, name=name, dtype=dtype)
