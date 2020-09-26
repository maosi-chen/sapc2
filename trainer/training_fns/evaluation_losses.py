import tensorflow as tf
import tensorflow_probability as tfp

from GCP_fnAPI_tf2.trainer.utils.tf_utility import sobel_edges_tfpad

from GCP_fnAPI_tf2.trainer.parameters.config import DNN_PARAMS

## Loss/Metrics functions

@tf.function(input_signature=[
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
])
def calc_unmasked_MSE_per_image(y_true_stacked, y_pred_stacked):
  # y_true_stacked: [B, H, W, 2]
  ## 0: y_true:     [B, H, W]
  ## 1: mask:       [B, H, W]
  # y_pred_stacked: [B, H, W, 2]
  ## 0: y_pred:     [B, H, W]
  ## 1: ref:        [B, H, W]
  y_true = y_true_stacked[:,:,:,0]
  mask   = y_true_stacked[:,:,:,1]
  y_pred = y_pred_stacked[:,:,:,0]

  ## [B, H, W]
  true_unmasked_values = tf.math.multiply(y_true, mask)
  predicted_unmasked_values = tf.math.multiply(y_pred, mask)

  ## [B]
  unmasked_count = tf.math.reduce_sum(mask, axis = [1,2])

  ## [B]
  SSE = tf.math.reduce_sum(
    tf.math.square(true_unmasked_values - predicted_unmasked_values),
    axis = [1,2]
  )

  ## [B]
  MSE = tf.math.divide(SSE, tf.cast(unmasked_count, tf.float32))
  
  return MSE

@tf.function(input_signature=[
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
])
def calc_masked_MSE_per_image(y_true_stacked, y_pred_stacked):
  # y_true_stacked: [B, H, W, 2]
  ## 0: y_true:     [B, H, W]
  ## 1: mask:       [B, H, W]
  # y_pred_stacked: [B, H, W, 2]
  ## 0: y_pred:     [B, H, W]
  ## 1: ref:       [B, H, W]
  y_true = y_true_stacked[:,:,:,0]
  mask   = y_true_stacked[:,:,:,1]
  y_pred = y_pred_stacked[:,:,:,0]

  #Must invert mask to zero out un-masked area
  #accomplish this with abs(mask -1) 
  ## [B, H, W]
  reverse_mask = 1.0 - mask #tf.math.abs(tf.math.subtract(mask, tf.constant(1, dtype = mask.dtype, shape = mask.shape)))
  
  #[B, H, W]
  true_masked_values = tf.math.multiply(y_true, reverse_mask)
  predicted_masked_values = tf.math.multiply(y_pred, reverse_mask)

  #[B]
  masked_count = tf.math.reduce_sum(reverse_mask, axis = [1,2])

  ## [B]
  #masked_MSE_ts = tf.losses.MSE(true_masked_values, predicted_masked_values)
  masked_SSE_ts = tf.math.reduce_sum(
    tf.math.square(predicted_masked_values - true_masked_values),
    axis = [1, 2]
  )

  ## [B]
  masked_MSE_ts = tf.math.divide(masked_SSE_ts, tf.cast(masked_count, tf.float32))
  
  return masked_MSE_ts

@tf.function(input_signature=[
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
])
def calc_whole_MSE_per_image(y_true_stacked, y_pred_stacked):
  # y_true_stacked: [B, H, W, 2]
  ## 0: y_true:     [B, H, W]
  ## 1: mask:       [B, H, W]
  # y_pred_stacked: [B, H, W, 2]
  ## 0: y_pred:     [B, H, W]
  ## 1: ref:        [B, H, W]
  y_true = y_true_stacked[:,:,:,0]
  y_pred = y_pred_stacked[:,:,:,0]


  #[B]
  unweighted_SSE_ts = tf.math.reduce_sum(
    tf.math.square(y_pred - y_true),
    axis = [1,2]
  )

  #[]
  pixel_count_ts = tf.cast(tf.math.multiply(tf.shape(y_true)[1], tf.shape(y_true)[2]), dtype = tf.float32)

  unweighted_MSE_ts = tf.math.divide(unweighted_SSE_ts, pixel_count_ts)
  
  return unweighted_MSE_ts


@tf.function(input_signature=[
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2],
                dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2],
                dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
])
def calc_whole_mosaic_MSE_per_image(y_true_stacked, y_pred_stacked):
  # y_true_stacked: [B, H, W, 2]
  ## 0: y_true:     [B, H, W]
  ## 1: mask:       [B, H, W]
  # y_pred_stacked: [B, H, W, 2]
  ## 0: y_pred:     [B, H, W]
  ## 1: ref:        [B, H, W]

  ## [B, H, W]
  y_true = y_true_stacked[:, :, :, 0]
  mask = y_true_stacked[:, :, :, 1]
  y_pred = y_pred_stacked[:, :, :, 0]

  ## [B, H, W]
  revMask = tf.constant(1., dtype=DNN_PARAMS['custom_dtype']._variable_dtype) - mask

  ## [B, H, W]
  y_mosaic = y_true * mask + y_pred * revMask

  # [B]
  unweighted_SSE_ts = tf.math.reduce_sum(
    tf.math.square(y_mosaic - y_true),
    axis=[1, 2]
  )

  # []
  pixel_count_ts = tf.cast(tf.math.multiply(tf.shape(y_true)[1], tf.shape(y_true)[2]), dtype=tf.float32)

  unweighted_MSE_ts = tf.math.divide(unweighted_SSE_ts, pixel_count_ts)

  return unweighted_MSE_ts


@tf.function(input_signature=[
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
])
def calc_orig_out_img_edge_unmasked_MSE_per_image(y_true_stacked, y_pred_stacked):
  # y_true_stacked: [B, H, W, 2]
  ## 0: y_true:     [B, H, W]
  ## 1: mask:       [B, H, W]
  # y_pred_stacked: [B, H, W, 2]
  ## 0: y_pred:     [B, H, W]
  ## 1: ref:        [B, H, W]

  y_true = y_true_stacked[:,:,:,0]
  mask   = y_true_stacked[:,:,:,1]
  y_pred = y_pred_stacked[:,:,:,0]


  ##[B]
  unmasked_count = tf.math.reduce_sum(mask, axis = [1,2])

  # [B, H, W, 2]
  y_stacked = tf.stack([y_true, y_pred], axis=-1)
  mask_stacked = tf.stack([mask, mask], axis = -1)

  # [B, H, W, 2, 2]
  #edges = tf.image.sobel_edges(y_stacked)
  edges = sobel_edges_tfpad(y_stacked)

  
  #[B, H, W, 2]
  true_edge_unmasked_values = tf.math.multiply(edges[:,:,:,0,:], mask_stacked)
  pred_edge_unmasked_values = tf.math.multiply(edges[:,:,:,1,:], mask_stacked) 


  # [B]
  edges_unmasked_SSE_ts = tf.math.reduce_sum(
    tf.math.square( true_edge_unmasked_values - pred_edge_unmasked_values ),
    axis = [1,2,3]
  ) / 2.0

  edges_unmasked_MSE_ts = tf.math.divide(edges_unmasked_SSE_ts, unmasked_count)

  
  return edges_unmasked_MSE_ts

@tf.function(input_signature=[
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2],
                dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2],
                dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
])
def calc_orig_out_img_edge_masked_MSE_per_image(y_true_stacked, y_pred_stacked):
  # y_true_stacked: [B, H, W, 2]
  ## 0: y_true:     [B, H, W]
  ## 1: mask:       [B, H, W]
  # y_pred_stacked: [B, H, W, 2]
  ## 0: y_pred:     [B, H, W]
  ## 1: ref:        [B, H, W]

  # [B, H, W]
  y_true = y_true_stacked[:,:,:,0]
  mask   = y_true_stacked[:,:,:,1]
  y_pred = y_pred_stacked[:,:,:,0]
  ref    = y_pred_stacked[:,:,:,1]

  reverse_mask = 1.0 - mask
  ## [B, H, W, 1, 1]
  reversedMask_5D = tf.expand_dims(tf.expand_dims(reverse_mask, axis=-1), axis=-1)


  ## [B]
  masked_count = tf.math.reduce_sum(reverse_mask, axis=[1, 2])

  batch_size_ts = tf.cast(tf.shape(y_true)[0], tf.dtypes.float32)  # Bi

  # reshape to 2D [B, H*W]
  ref_2D = tf.reshape(ref, (batch_size_ts, -1))
  y_true_2D = tf.reshape(y_true, (batch_size_ts, -1))
  
  # correlation coefficient between ref and y_true (complete target)
  ## [B, B]
  # corr_matrix = tfp.stats.correlation(ref_2D, y_true_2D, sample_axis = 1, event_axis = 0)
  ## [B]
  # corr_1D = tf.linalg.tensor_diag_part(corr_matrix)

  ## [B]
  corr_1D = tfp.stats.correlation(ref_2D, y_true_2D, sample_axis=1, event_axis=None)

  abs_corr_1D = tf.math.abs(corr_1D)

  # mosaic the unmasked part of y_true and masked part of output.
  ## [B, H, W]
  #mosaic_unmsk_true_msk_pred = y_true * mask + y_pred * (1.0 - mask)

  # calculate the edge loss between the mosaic image and y_true
  ## the unmasked part's edge loss can cancel out, therefore don't need to do the calculation on masked area.
  ## instead do the calculation on the entire image.
  # [B, H, W, 2]
  #y_stacked = tf.stack([y_true, mosaic_unmsk_true_msk_pred], axis=-1)
  y_stacked = tf.stack([y_true, y_pred], axis=-1)

  # [B, H, W, 2, 2]
  # edges = tf.image.sobel_edges(y_stacked) * tf.math.sqrt( tf.reshape(abs_corr_1D, (-1, 1, 1, 1, 1)) )
  edges = sobel_edges_tfpad(y_stacked) * tf.math.sqrt(tf.reshape(abs_corr_1D, (-1, 1, 1, 1, 1)))

  edges = edges * reversedMask_5D

  # input to reduce_sum: [B, H, W,    2]
  # edges_masked_SSE_ts: [B]
  edges_masked_SSE_ts = tf.math.reduce_sum(
    tf.math.square(edges[:, :, :, 0, :] - edges[:, :, :, 1, :]),
    axis=[1, 2, 3]
  ) / 2.0

  edges_masked_MSE_ts = tf.math.divide(edges_masked_SSE_ts, masked_count)

  return edges_masked_MSE_ts


@tf.function(input_signature=[
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
])
def calc_tgt_variance(y_true_stacked, y_pred_stacked):

  y_true = y_true_stacked[:,:,:,0]
  #ref    = y_pred_stacked[:,:,:,1]

  batch_size_ts = tf.cast(tf.shape(y_true)[0], tf.dtypes.float32) # Bi

  # reshape to 2D [B, H*W]
  #ref_2D = tf.reshape(ref, (batch_size_ts, -1))
  y_true_2D = tf.reshape(y_true, (batch_size_ts, -1))

  variance = tf.math.reduce_variance(y_true, axis = [1,2])

  return variance

@tf.function(input_signature=[
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
])
def calc_correlation_coefficient_ref_tgt(y_true_stacked, y_pred_stacked):

  y_true = y_true_stacked[:,:,:,0]
  ref    = y_pred_stacked[:,:,:,1]

  batch_size_ts = tf.cast(tf.shape(y_true)[0], tf.dtypes.float32) # Bi

  # reshape to 2D [B, H*W]
  ref_2D = tf.reshape(ref, (batch_size_ts, -1))
  y_true_2D = tf.reshape(y_true, (batch_size_ts, -1))

  # correlation coefficient between ref and y_true (complete target)
  ## [B, B]
  #corr_matrix = tfp.stats.correlation(ref_2D, y_true_2D, sample_axis = 1, event_axis = 0)
  ## [B]
  #corr_1D = tf.linalg.tensor_diag_part(corr_matrix)
  corr_1D = tfp.stats.correlation(ref_2D, y_true_2D, sample_axis = 1, event_axis = None)
  abs_corr_1D = tf.math.abs(corr_1D)

  return abs_corr_1D
  
@tf.function(input_signature=[
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
])
def calc_cloud_fraction(y_true_stacked, y_pred_stacked):

  # [B, H, W]
  mask = tf.cast(y_true_stacked[:,:,:,1], tf.dtypes.float32)
  ones_mask = tf.ones_like(mask, dtype=tf.dtypes.float32)

  # [B]
  cloud_fraction = tf.math.reduce_sum(ones_mask - mask, axis=[1,2]) / tf.math.reduce_sum(ones_mask, axis=[1,2])

  return cloud_fraction


@tf.function(input_signature=[
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[], dtype=DNN_PARAMS['custom_dtype']._variable_dtype), # mask_weight
  tf.TensorSpec(shape=[], dtype=DNN_PARAMS['custom_dtype']._variable_dtype), # unmask_weight
  tf.TensorSpec(shape=[], dtype=DNN_PARAMS['custom_dtype']._variable_dtype) # edge_weight
])
def weighted_loss_per_image(y_true_stacked, y_pred_stacked,
                            mask_weight = tf.constant(1.0, dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
                            unmask_weight = tf.constant(1.0, dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
                            edge_weight = tf.constant(0.1, dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
                            ):
  # y_true_stacked: [B, H, W, 2]
  ## 0: y_true:     [B, H, W]
  ## 1: mask:       [B, H, W]
  # y_pred_stacked: [B, H, W, 2]
  ## 0: y_pred:     [B, H, W]
  ## 1: ref:        [B, H, W]

  # def loss(y_true_stacked, y_pred_stacked):
  unmasked_MSE = calc_unmasked_MSE_per_image(y_true_stacked, y_pred_stacked)
  masked_MSE = calc_masked_MSE_per_image(y_true_stacked, y_pred_stacked)

  orig_out_img_edge_unmasked_MSE = calc_orig_out_img_edge_unmasked_MSE_per_image(y_true_stacked, y_pred_stacked)
  #mosaic_out_img_edge_masked_MSE = calc_mosaic_out_img_edge_masked_MSE_per_image(y_true_stacked, y_pred_stacked)
  mosaic_out_img_edge_masked_MSE = calc_orig_out_img_edge_masked_MSE_per_image(y_true_stacked, y_pred_stacked)

  #weighted_total_MSE = mask_weight * masked_MSE + unmask_weight * unmasked_MSE + \
  #                      edge_weight * (orig_out_img_edge_unmasked_MSE + mosaic_out_img_edge_masked_MSE)
  weighted_total_MSE = \
    mask_weight * masked_MSE + \
    unmask_weight * unmasked_MSE + \
    mask_weight * edge_weight * mosaic_out_img_edge_masked_MSE + \
    unmask_weight * edge_weight * orig_out_img_edge_unmasked_MSE

  return weighted_total_MSE
  # return loss

## keras regularization loss (metric)
#def model_regularization_loss(model_losses):
#  def regularization_loss(y_true_stacked, y_pred_stacked):
#    return tf.math.add_n(model_losses)
#  return regularization_loss

@tf.function(input_signature=[
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
])
def calc_correlation_coefficient_tgt_true_pred_per_full_image(y_true_stacked, y_pred_stacked):
  # y_true_stacked: [B, H, W, 2]
  ## 0: y_true:     [B, H, W]
  ## 1: mask:       [B, H, W]
  # y_pred_stacked: [B, H, W, 2]
  ## 0: y_pred:     [B, H, W]
  ## 1: ref:        [B, H, W]
  
  # [B, H, W]
  y_true = y_true_stacked[:, :, :, 0]
  # mask   = y_true_stacked[:,:,:,1]
  y_pred = y_pred_stacked[:, :, :, 0]
  # ref    = y_pred_stacked[:,:,:,1]
  
  batch_size_ts = tf.cast(tf.shape(y_true)[0], tf.dtypes.float32)  # Bi
  
  # reshape to 2D [B, H*W]
  y_true_2D = tf.reshape(y_true, (batch_size_ts, -1))
  y_pred_2D = tf.reshape(y_pred, (batch_size_ts, -1))
  
  # correlation coefficient between ref and y_true (complete target)
  ## [B, B]
  # corr_matrix = tfp.stats.correlation(ref_2D, y_true_2D, sample_axis = 1, event_axis = 0)
  ## [B]
  # corr_1D = tf.linalg.tensor_diag_part(corr_matrix)
  corr_1D = tfp.stats.correlation(y_true_2D, y_pred_2D, sample_axis=1, event_axis=None)
  abs_corr_1D = tf.math.abs(corr_1D)
  
  return abs_corr_1D


@tf.function(input_signature=[
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2],
                dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2],
                dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
])
def calc_correlation_coefficient_tgt_true_mosaic_per_full_image(y_true_stacked, y_pred_stacked):
  # y_true_stacked: [B, H, W, 2]
  ## 0: y_true:     [B, H, W]
  ## 1: mask:       [B, H, W]
  # y_pred_stacked: [B, H, W, 2]
  ## 0: y_pred:     [B, H, W]
  ## 1: ref:        [B, H, W]

  # [B, H, W]
  y_true = y_true_stacked[:, :, :, 0]
  mask   = y_true_stacked[:,:,:,1]
  y_pred = y_pred_stacked[:, :, :, 0]
  # ref    = y_pred_stacked[:,:,:,1]

  ## [B, H, W]
  revMask = tf.constant(1, dtype=DNN_PARAMS['custom_dtype']._variable_dtype) - mask

  # [B, H, W]
  y_mosaic = y_true * mask + y_pred * revMask

  batch_size_ts = tf.cast(tf.shape(y_true)[0], tf.dtypes.float32)  # Bi

  # reshape to 2D [B, H*W]
  y_true_2D = tf.reshape(y_true, (batch_size_ts, -1))
  #y_pred_2D = tf.reshape(y_pred, (batch_size_ts, -1))
  y_pred_2D = tf.reshape(y_mosaic, (batch_size_ts, -1))

  # correlation coefficient between ref and y_true (complete target)
  ## [B, B]
  # corr_matrix = tfp.stats.correlation(ref_2D, y_true_2D, sample_axis = 1, event_axis = 0)
  ## [B]
  # corr_1D = tf.linalg.tensor_diag_part(corr_matrix)
  corr_1D = tfp.stats.correlation(y_true_2D, y_pred_2D, sample_axis=1, event_axis=None)
  abs_corr_1D = tf.math.abs(corr_1D)

  return abs_corr_1D


@tf.function(input_signature=[
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[], dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
])
def calc_psnr_per_full_image(y_true_stacked, y_pred_stacked,
                             max_val=tf.constant(1.0, dtype=DNN_PARAMS['custom_dtype']._variable_dtype)):
  # def psnr_batch_fn(y_true_stacked, y_pred_stacked):
  # y_true_stacked: [B, H, W, 2]
  ## 0: y_true:     [B, H, W]
  ## 1: mask:       [B, H, W]
  # y_pred_stacked: [B, H, W, 2]
  ## 0: y_pred:     [B, H, W]
  ## 1: ref:        [B, H, W]
  
  # [B, H, W, 1]
  y_true = y_true_stacked[:, :, :, 0:1]
  # mask   = y_true_stacked[:,:,:,1:2]
  y_pred = y_pred_stacked[:, :, :, 0:1]
  # ref    = y_pred_stacked[:,:,:,1:2]
  
  psnr_batch = tf.image.psnr(y_true, y_pred, max_val)
  
  return psnr_batch
  #return psnr_batch_fn


@tf.function(input_signature=[
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2],
                dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2],
                dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[], dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
])
def calc_psnr_mosaic_per_full_image(y_true_stacked, y_pred_stacked,
                             max_val=tf.constant(1.0, dtype=DNN_PARAMS['custom_dtype']._variable_dtype)):
  # def psnr_batch_fn(y_true_stacked, y_pred_stacked):
  # y_true_stacked: [B, H, W, 2]
  ## 0: y_true:     [B, H, W]
  ## 1: mask:       [B, H, W]
  # y_pred_stacked: [B, H, W, 2]
  ## 0: y_pred:     [B, H, W]
  ## 1: ref:        [B, H, W]

  # [B, H, W, 1]
  y_true = y_true_stacked[:, :, :, 0:1]
  mask   = y_true_stacked[:,:,:,1:2]
  y_pred = y_pred_stacked[:, :, :, 0:1]
  # ref    = y_pred_stacked[:,:,:,1:2]

  ## [B, H, W, 1]
  revMask = tf.constant(1, dtype=DNN_PARAMS['custom_dtype']._variable_dtype) - mask

  # [B, H, W, 1]
  y_mosaic = y_true * mask + y_pred * revMask

  #psnr_batch = tf.image.psnr(y_true, y_pred, max_val)
  psnr_batch = tf.image.psnr(y_true, y_mosaic, max_val)

  return psnr_batch
  # return psnr_batch_fn


# @tf.function(input_signature=[
#   tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
#   tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
#   tf.TensorSpec(shape=[], dtype=DNN_PARAMS['custom_dtype']._variable_dtype), #max_val
#   tf.TensorSpec(shape=[], dtype=DNN_PARAMS['custom_dtype']._variable_dtype), #filter_size
#   tf.TensorSpec(shape=[], dtype=DNN_PARAMS['custom_dtype']._variable_dtype), #filter_sigma
#   tf.TensorSpec(shape=[], dtype=DNN_PARAMS['custom_dtype']._variable_dtype), #k1
#   tf.TensorSpec(shape=[], dtype=DNN_PARAMS['custom_dtype']._variable_dtype) #k2
# ])
def calc_ssim_per_full_image(y_true_stacked, y_pred_stacked,
                             max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03
                             ):
  #def ssim(y_true_stacked, y_pred_stacked):
  # y_true_stacked: [B, H, W, 2]
  ## 0: y_true:     [B, H, W]
  ## 1: mask:       [B, H, W]
  # y_pred_stacked: [B, H, W, 2]
  ## 0: y_pred:     [B, H, W]
  ## 1: ref:        [B, H, W]
  
  # [B, H, W, 1]
  y_true = y_true_stacked[:,:,:,0:1]
  #mask   = y_true_stacked[:,:,:,1:2]
  y_pred = y_pred_stacked[:,:,:,0:1]
  #ref    = y_pred_stacked[:,:,:,1:2]
  
  ssim_batch = tf.image.ssim(y_true, y_pred,
    max_val=max_val, filter_size=filter_size, filter_sigma=filter_sigma, k1=k1, k2=k2)
  
  return ssim_batch
  #return ssim


def calc_ssim_mosaic_per_full_image(
  y_true_stacked, y_pred_stacked,
  max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03
  ):
  # def ssim(y_true_stacked, y_pred_stacked):
  # y_true_stacked: [B, H, W, 2]
  ## 0: y_true:     [B, H, W]
  ## 1: mask:       [B, H, W]
  # y_pred_stacked: [B, H, W, 2]
  ## 0: y_pred:     [B, H, W]
  ## 1: ref:        [B, H, W]

  # [B, H, W, 1]
  y_true = y_true_stacked[:, :, :, 0:1]
  mask   = y_true_stacked[:,:,:,1:2]
  y_pred = y_pred_stacked[:, :, :, 0:1]
  # ref    = y_pred_stacked[:,:,:,1:2]

  ## [B, H, W, 1]
  revMask = tf.constant(1, dtype=DNN_PARAMS['custom_dtype']._variable_dtype) - mask

  # [B, H, W, 1]
  y_mosaic = y_true * mask + y_pred * revMask

  #ssim_batch = tf.image.ssim(y_true, y_pred,
  ssim_batch = tf.image.ssim(y_true, y_mosaic,
                             max_val=max_val, filter_size=filter_size, filter_sigma=filter_sigma, k1=k1, k2=k2)

  return ssim_batch
  # return ssim
