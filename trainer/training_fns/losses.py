import tensorflow as tf
import tensorflow_probability as tfp

from GCP_fnAPI_tf2.trainer.utils.tf_utility import sobel_edges_tfpad

from GCP_fnAPI_tf2.trainer.parameters.config import DNN_PARAMS

## Loss/Metrics functions

# @tf.function(input_signature=[
#   tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=tf.float32),
#   tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=tf.float32)
# ])
# @tf.function(input_signature=[
#   tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._compute_dtype),
#   tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._compute_dtype)
# ])
@tf.function(input_signature=[
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
])
def calc_unmasked_MSE(y_true_stacked, y_pred_stacked):
  # y_true_stacked: [B, H, W, 2]
  ## 0: y_true:     [B, H, W]
  ## 1: mask:       [B, H, W]
  # y_pred_stacked: [B, H, W, 2]
  ## 0: y_pred:     [B, H, W]
  ## 1: ref:        [B, H, W]
  y_true = y_true_stacked[:,:,:,0]
  mask   = y_true_stacked[:,:,:,1]
  y_pred = y_pred_stacked[:,:,:,0]

  # unmasked_indices = tf.where(tf.cast(mask, tf.bool))
  # #true_unmasked_values = tf.reshape(tf.gather_nd(y_true, unmasked_indices), [-1])
  # #predicted_unmasked_values = tf.reshape(tf.gather_nd(y_pred, unmasked_indices), [-1])
  # true_unmasked_values = tf.reshape(tf.gather_nd(y_true, unmasked_indices), tf.constant([-1], dtype=tf.int32))
  # predicted_unmasked_values = tf.reshape(tf.gather_nd(y_pred, unmasked_indices), tf.constant([-1], dtype=tf.int32))
  #
  # #unmasked_count = tf.cast(tf.size(true_unmasked_values), tf.dtypes.float32) # Bi * 64 * 64 * fi
  # #unmasked_count = tf.cast(tf.size(true_unmasked_values), DNN_PARAMS['custom_dtype']._compute_dtype)  # Bi * 64 * 64 * fi
  # unmasked_count = tf.cast(tf.size(true_unmasked_values), DNN_PARAMS['custom_dtype']._variable_dtype)  # Bi * 64 * 64 * fi
  #
  # #batch_size_ts = tf.cast(tf.shape(y_true)[0], tf.dtypes.float32) # Bi
  # #batch_size_ts = tf.cast(tf.shape(y_true)[0], DNN_PARAMS['custom_dtype']._compute_dtype)  # Bi
  # batch_size_ts = tf.cast(tf.shape(y_true)[0], DNN_PARAMS['custom_dtype']._variable_dtype)  # Bi
  #
  # unmasked_SSE_ts = tf.math.reduce_sum(
  #   tf.math.square(predicted_unmasked_values - true_unmasked_values)
  # )

  ###
  ## [B, H, W]
  mask = tf.cast(mask, dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
  y_true = y_true * mask
  y_pred = y_pred * mask

  ## []
  unmasked_count = tf.math.reduce_sum(mask)
  batch_size_ts = tf.cast(tf.shape(y_true)[0], DNN_PARAMS['custom_dtype']._variable_dtype)  # Bi
  
  ## []
  unmasked_SSE_ts = tf.math.reduce_sum(
    tf.math.square(y_pred - y_true)
  )
  ###
  
  #unmasked_MSE_ts = (unmasked_SSE_ts / unmasked_count) * ( batch_size_ts / (1.0*FLAGS.global_batch_size) )
  unmasked_MSE_ts = tf.math.divide(
    tf.math.multiply(
      tf.math.divide(unmasked_SSE_ts, unmasked_count), 
      batch_size_ts
    ),
    #tf.constant(DNN_PARAMS['global_batch_size'], dtype=tf.float32)
    #tf.constant(DNN_PARAMS['global_batch_size'], dtype=DNN_PARAMS['custom_dtype']._compute_dtype)
    tf.constant(DNN_PARAMS['global_batch_size'], dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
  )
  return unmasked_MSE_ts

# @tf.function(input_signature=[
#   tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=tf.float32),
#   tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=tf.float32)
# ])
# @tf.function(input_signature=[
#   tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._compute_dtype),
#   tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._compute_dtype)
# ])
@tf.function(input_signature=[
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
])
def calc_masked_MSE(y_true_stacked, y_pred_stacked):
  # y_true_stacked: [B, H, W, 2]
  ## 0: y_true:     [B, H, W]
  ## 1: mask:       [B, H, W]
  # y_pred_stacked: [B, H, W, 2]
  ## 0: y_pred:     [B, H, W]
  ## 1: ref:       [B, H, W]
  y_true = y_true_stacked[:,:,:,0]
  mask   = y_true_stacked[:,:,:,1]
  y_pred = y_pred_stacked[:,:,:,0]
  
  # #masked_indices = tf.where(tf.cast(1 - mask, tf.bool))
  # #true_masked_values = tf.reshape(tf.gather_nd(y_true, masked_indices), [-1])
  # #predicted_masked_values = tf.reshape(tf.gather_nd(y_pred, masked_indices), [-1])
  # #masked_indices = tf.where(tf.cast(tf.constant(1, dtype=tf.float32) - mask, tf.bool))
  # #masked_indices = tf.where(tf.cast(tf.constant(1, dtype=DNN_PARAMS['custom_dtype']._compute_dtype) - mask, tf.bool))
  # masked_indices = tf.where(tf.cast(tf.constant(1, dtype=DNN_PARAMS['custom_dtype']._variable_dtype) - mask, tf.bool))
  # true_masked_values = tf.reshape(tf.gather_nd(y_true, masked_indices), tf.constant([-1], dtype=tf.int32))
  # predicted_masked_values = tf.reshape(tf.gather_nd(y_pred, masked_indices), tf.constant([-1], dtype=tf.int32))
  #
  # #masked_count = tf.cast(tf.size(true_masked_values), tf.dtypes.float32) # Bi * 64 * 64 * (1-fi)
  # #batch_size_ts = tf.cast(tf.shape(y_true)[0], tf.dtypes.float32) # Bi
  # #masked_count = tf.cast(tf.size(true_masked_values), DNN_PARAMS['custom_dtype']._compute_dtype) # Bi * 64 * 64 * (1-fi)
  # #batch_size_ts = tf.cast(tf.shape(y_true)[0], DNN_PARAMS['custom_dtype']._compute_dtype) # Bi
  # masked_count = tf.cast(tf.size(true_masked_values), DNN_PARAMS['custom_dtype']._variable_dtype) # Bi * 64 * 64 * (1-fi)
  # batch_size_ts = tf.cast(tf.shape(y_true)[0], DNN_PARAMS['custom_dtype']._variable_dtype) # Bi
  #
  # #masked_MSE_ts = tf.losses.MSE(true_masked_values, predicted_masked_values)
  # masked_SSE_ts = tf.math.reduce_sum(
  #   tf.math.square(predicted_masked_values - true_masked_values)
  # )
  
  ###
  ## [B, H, W]
  revMask = tf.constant(1, dtype=DNN_PARAMS['custom_dtype']._variable_dtype) - mask
  masked_y_true = y_true * revMask
  masked_y_pred = y_pred * revMask

  ## []
  masked_count = tf.math.reduce_sum(revMask)
  batch_size_ts = tf.cast(tf.shape(y_true)[0], DNN_PARAMS['custom_dtype']._variable_dtype)  # Bi
  
  ## []
  masked_SSE_ts = tf.math.reduce_sum(
    tf.math.square(masked_y_pred - masked_y_true)
  )
  ###
  
  #masked_MSE_ts = (masked_SSE_ts / masked_count) * ( batch_size_ts / (1.0*FLAGS.global_batch_size) )
  masked_MSE_ts = tf.divide(
    tf.math.multiply(
      tf.math.divide(masked_SSE_ts, masked_count),
      batch_size_ts
    ),
    #tf.constant(DNN_PARAMS['global_batch_size'], dtype=tf.float32)
    #tf.constant(DNN_PARAMS['global_batch_size'], dtype=DNN_PARAMS['custom_dtype']._compute_dtype)
    tf.constant(DNN_PARAMS['global_batch_size'], dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
  )
  return masked_MSE_ts

# @tf.function(input_signature=[
#   tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=tf.float32),
#   tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=tf.float32)
# ])
# @tf.function(input_signature=[
#   tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._compute_dtype),
#   tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._compute_dtype)
# ])
@tf.function(input_signature=[
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
])
def calc_unweighted_MSE(y_true_stacked, y_pred_stacked):
  # y_true_stacked: [B, H, W, 2]
  ## 0: y_true:     [B, H, W]
  ## 1: mask:       [B, H, W]
  # y_pred_stacked: [B, H, W, 2]
  ## 0: y_pred:     [B, H, W]
  ## 1: ref:        [B, H, W]
  y_true = y_true_stacked[:,:,:,0]
  y_pred = y_pred_stacked[:,:,:,0]

  unweighted_SSE_ts = tf.math.reduce_sum(
    tf.math.square(y_pred - y_true)
  )
  
  #pixel_counts = tf.cast(tf.size(y_true), tf.dtypes.float32) # Bi * 64 * 64
  #batch_size_ts = tf.cast(tf.shape(y_true)[0], tf.dtypes.float32) # Bi
  #pixel_counts = tf.cast(tf.size(y_true), DNN_PARAMS['custom_dtype']._compute_dtype) # Bi * 64 * 64
  #batch_size_ts = tf.cast(tf.shape(y_true)[0], DNN_PARAMS['custom_dtype']._compute_dtype) # Bi
  pixel_counts = tf.cast(tf.size(y_true), DNN_PARAMS['custom_dtype']._variable_dtype) # Bi * 64 * 64
  batch_size_ts = tf.cast(tf.shape(y_true)[0], DNN_PARAMS['custom_dtype']._variable_dtype) # Bi

  #unweighted_MSE_ts = (unweighted_SSE_ts / pixel_counts) * ( batch_size_ts / (1.0*FLAGS.global_batch_size) )
  unweighted_MSE_ts = tf.divide(
    tf.math.multiply(
      tf.math.divide(unweighted_SSE_ts, pixel_counts),
      batch_size_ts
    ),
    #tf.constant(DNN_PARAMS['global_batch_size'], dtype=tf.float32)
    #tf.constant(DNN_PARAMS['global_batch_size'], dtype=DNN_PARAMS['custom_dtype']._compute_dtype)
    tf.constant(DNN_PARAMS['global_batch_size'], dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
  )
  return unweighted_MSE_ts

# @tf.function(input_signature=[
#   tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=tf.float32),
#   tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=tf.float32)
# ])
# @tf.function(input_signature=[
#   tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._compute_dtype),
#   tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._compute_dtype)
# ])
@tf.function(input_signature=[
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
])
def calc_orig_out_img_edge_unmasked_MSE(y_true_stacked, y_pred_stacked):
  # y_true_stacked: [B, H, W, 2]
  ## 0: y_true:     [B, H, W]
  ## 1: mask:       [B, H, W]
  # y_pred_stacked: [B, H, W, 2]
  ## 0: y_pred:     [B, H, W]
  ## 1: ref:        [B, H, W]

  y_true = y_true_stacked[:,:,:,0]
  mask   = y_true_stacked[:,:,:,1]
  y_pred = y_pred_stacked[:,:,:,0]

  # # [B, H, W, 2]
  # y_stacked = tf.stack([y_true, y_pred], axis=-1)
  #
  # # [B, H, W, 2, 2]
  # #edges = tf.image.sobel_edges(y_stacked)
  # edges = sobel_edges_tfpad(y_stacked)
  #
  #
  # # [?, 3]
  # unmasked_indices = tf.where(tf.cast(mask, tf.bool))
  #
  # # [?, 2]
  # #true_edge_unmasked_values = tf.reshape(tf.gather_nd(edges[:,:,:,0,:], unmasked_indices), [-1])
  # #pred_edge_unmasked_values = tf.reshape(tf.gather_nd(edges[:,:,:,1,:], unmasked_indices), [-1])
  # true_edge_unmasked_values = tf.reshape(tf.gather_nd(edges[:, :, :, 0, :], unmasked_indices), tf.constant([-1], dtype=tf.int32))
  # pred_edge_unmasked_values = tf.reshape(tf.gather_nd(edges[:, :, :, 1, :], unmasked_indices), tf.constant([-1], dtype=tf.int32))
  #
  # # scalars
  # #unmasked_count = tf.cast(tf.size(true_edge_unmasked_values), tf.dtypes.float32) # Bi * 64 * 64 * fi * 2
  # #batch_size_ts = tf.cast(tf.shape(y_true)[0], tf.dtypes.float32) # Bi
  # #unmasked_count = tf.cast(tf.size(true_edge_unmasked_values), DNN_PARAMS['custom_dtype']._compute_dtype) # Bi * 64 * 64 * fi * 2
  # #batch_size_ts = tf.cast(tf.shape(y_true)[0], DNN_PARAMS['custom_dtype']._compute_dtype) # Bi
  # unmasked_count = tf.cast(tf.size(true_edge_unmasked_values), DNN_PARAMS['custom_dtype']._variable_dtype) # Bi * 64 * 64 * fi * 2
  # batch_size_ts = tf.cast(tf.shape(y_true)[0], DNN_PARAMS['custom_dtype']._variable_dtype) # Bi
  #
  # # scalar
  # edges_unmasked_SSE_ts = tf.math.reduce_sum(
  #   tf.math.square(true_edge_unmasked_values - pred_edge_unmasked_values)
  #   # ) / 2.0
  #   # ) / tf.constant(2.0, dtype=tf.float32)
  #   # ) / tf.constant(2.0, dtype=DNN_PARAMS['custom_dtype']._compute_dtype)
  # ) / tf.constant(2.0, dtype=DNN_PARAMS['custom_dtype']._variable_dtype)

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
  batch_size_ts = tf.cast(tf.shape(y_true)[0], DNN_PARAMS['custom_dtype']._variable_dtype)  # Bi
  
  # []
  edges_unmasked_SSE_ts = tf.math.reduce_sum(
    tf.math.square(edges[:, :, :, 0, :] - edges[:, :, :, 1, :])
  ) / tf.constant(2.0, dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
  ###
  
  edges_unmasked_MSE_ts = tf.math.divide(
    tf.math.multiply(
      tf.math.divide(edges_unmasked_SSE_ts, unmasked_count), 
      batch_size_ts
    ),
    #tf.constant(DNN_PARAMS['global_batch_size'], dtype=tf.float32)
    #tf.constant(DNN_PARAMS['global_batch_size'], dtype=DNN_PARAMS['custom_dtype']._compute_dtype)
    tf.constant(DNN_PARAMS['global_batch_size'], dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
  )
  
  return edges_unmasked_MSE_ts

# @tf.function(input_signature=[
#   tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=tf.float32),
#   tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=tf.float32)
# ])
# @tf.function(input_signature=[
#   tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._compute_dtype),
#   tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._compute_dtype)
# ])
@tf.function(input_signature=[
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
])
def calc_mosaic_out_img_edge_masked_MSE(y_true_stacked, y_pred_stacked):
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

  # #
  # #masked_indices = tf.where(tf.cast(1 - mask, tf.bool))
  # #masked_indices = tf.where(tf.cast(tf.constant(1, dtype=tf.float32) - mask, tf.bool))
  # #masked_indices = tf.where(tf.cast(tf.constant(1, dtype=DNN_PARAMS['custom_dtype']._compute_dtype) - mask, tf.bool))
  # masked_indices = tf.where(tf.cast(tf.constant(1, dtype=DNN_PARAMS['custom_dtype']._variable_dtype) - mask, tf.bool))
  #
  # #true_masked_values = tf.reshape(tf.gather_nd(y_true, masked_indices), [-1])
  # true_masked_values = tf.reshape(tf.gather_nd(y_true, masked_indices), tf.constant([-1], dtype=tf.int32))
  # #predicted_masked_values = tf.reshape(tf.gather_nd(y_pred, masked_indices), [-1])
  # predicted_masked_values = tf.reshape(tf.gather_nd(y_pred, masked_indices), tf.constant([-1], dtype=tf.int32))
  #
  # #masked_count = tf.cast(tf.size(true_masked_values), tf.dtypes.float32) # Bi * 64 * 64 * (1-fi)
  # #batch_size_ts = tf.cast(tf.shape(y_true)[0], tf.dtypes.float32) # Bi
  # #masked_count = tf.cast(tf.size(true_masked_values), DNN_PARAMS['custom_dtype']._compute_dtype) # Bi * 64 * 64 * (1-fi)
  # #batch_size_ts = tf.cast(tf.shape(y_true)[0], DNN_PARAMS['custom_dtype']._compute_dtype) # Bi
  # masked_count = tf.cast(tf.size(true_masked_values), DNN_PARAMS['custom_dtype']._variable_dtype) # Bi * 64 * 64 * (1-fi)
  
  ###
  ## [B, H, W]
  reversedMask = tf.constant(1, dtype=DNN_PARAMS['custom_dtype']._variable_dtype) - mask
  
  ## [] # B * H * W * (1-fi)
  masked_count = tf.math.reduce_sum(reversedMask)
  ###
  
  batch_size_ts = tf.cast(tf.shape(y_true)[0], DNN_PARAMS['custom_dtype']._variable_dtype) # Bi

  # reshape to 2D [B, H*W]
  # ref_2D = tf.reshape(ref, (batch_size_ts, -1))
  # y_true_2D = tf.reshape(y_true, (batch_size_ts, -1))
  shp_2D = (tf.cast(batch_size_ts, tf.int32), tf.constant(-1, dtype=tf.int32))
  ref_2D = tf.reshape(ref, shp_2D)
  y_true_2D = tf.reshape(y_true, shp_2D)
  
  # correlation coefficient between ref and y_true (complete target)
  ## [B, B]
  #corr_matrix = tfp.stats.correlation(ref_2D, y_true_2D, sample_axis = 1, event_axis = 0)
  ## [B]
  #corr_1D = tf.linalg.tensor_diag_part(corr_matrix)

  ## [B]
  corr_1D = tfp.stats.correlation(ref_2D, y_true_2D, sample_axis = 1, event_axis = None)

  #abs_corr_1D = tf.math.abs(corr_1D)
  #abs_corr_1D = tf.cast(tf.math.abs(corr_1D), dtype=DNN_PARAMS['custom_dtype']._compute_dtype)
  abs_corr_1D = tf.cast(tf.math.abs(corr_1D), dtype=DNN_PARAMS['custom_dtype']._variable_dtype)

  # mosaic the unmasked part of y_true and masked part of output.
  ## [B, H, W]
  #mosaic_unmsk_true_msk_pred = y_true * mask + y_pred * (1.0 - mask)
  #mosaic_unmsk_true_msk_pred = y_true * mask + y_pred * (tf.constant(1.0, dtype=tf.float32) - mask)
  #mosaic_unmsk_true_msk_pred = y_true * mask + y_pred * (tf.constant(1.0, dtype=DNN_PARAMS['custom_dtype']._compute_dtype) - mask)
  mosaic_unmsk_true_msk_pred = y_true * mask + y_pred * (tf.constant(1.0, dtype=DNN_PARAMS['custom_dtype']._variable_dtype) - mask)

  # calculate the edge loss between the mosaic image and y_true
  ## the unmasked part's edge loss can cancel out, therefore don't need to do the calculation on masked area.
  ## instead do the calculation on the entire image.
  # [B, H, W, 2]
  y_stacked = tf.stack([y_true, mosaic_unmsk_true_msk_pred], axis=-1)

  # [B, H, W, 2, 2]
  #edges = tf.image.sobel_edges(y_stacked) * tf.math.sqrt( tf.reshape(abs_corr_1D, (-1, 1, 1, 1, 1)) )
  # edges = tf.image.sobel_edges(y_stacked) * tf.math.sqrt(
  #   tf.reshape(abs_corr_1D, tf.constant((-1, 1, 1, 1, 1), dtype=tf.int32)))
  edges = sobel_edges_tfpad(y_stacked) * tf.math.sqrt(
    tf.reshape(abs_corr_1D, tf.constant((-1, 1, 1, 1, 1), dtype=tf.int32)))
  
  # [B, H, W,    2]
  edges_masked_SSE_ts = tf.math.reduce_sum(
    tf.math.square( edges[:,:,:,0,:] - edges[:,:,:,1,:] )
  # ) / 2.0
  #) / tf.constant(2.0, dtype=tf.float32)
  #) / tf.constant(2.0, dtype=DNN_PARAMS['custom_dtype']._compute_dtype)
  ) / tf.constant(2.0, dtype=DNN_PARAMS['custom_dtype']._variable_dtype)

  # 
  edges_masked_MSE_ts = tf.math.divide(
    tf.math.multiply(
      tf.math.divide(edges_masked_SSE_ts, masked_count), 
      batch_size_ts
    ),
    #tf.constant(DNN_PARAMS['global_batch_size'], dtype=tf.float32)
    #tf.constant(DNN_PARAMS['global_batch_size'], dtype=DNN_PARAMS['custom_dtype']._compute_dtype)
    tf.constant(DNN_PARAMS['global_batch_size'], dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
  )

  return edges_masked_MSE_ts


def calc_mosaic_out_img_edge_masked_MSE_v2(y_true_stacked, y_pred_stacked):
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
  
  # #
  # #masked_indices = tf.where(tf.cast(1 - mask, tf.bool))
  # #masked_indices = tf.where(tf.cast(tf.constant(1, dtype=tf.float32) - mask, tf.bool))
  # #masked_indices = tf.where(tf.cast(tf.constant(1, dtype=DNN_PARAMS['custom_dtype']._compute_dtype) - mask, tf.bool))
  # masked_indices = tf.where(tf.cast(tf.constant(1, dtype=DNN_PARAMS['custom_dtype']._variable_dtype) - mask, tf.bool))
  #
  # #true_masked_values = tf.reshape(tf.gather_nd(y_true, masked_indices), [-1])
  # true_masked_values = tf.reshape(tf.gather_nd(y_true, masked_indices), tf.constant([-1], dtype=tf.int32))
  # #predicted_masked_values = tf.reshape(tf.gather_nd(y_pred, masked_indices), [-1])
  # predicted_masked_values = tf.reshape(tf.gather_nd(y_pred, masked_indices), tf.constant([-1], dtype=tf.int32))
  #
  # #masked_count = tf.cast(tf.size(true_masked_values), tf.dtypes.float32) # Bi * 64 * 64 * (1-fi)
  # #batch_size_ts = tf.cast(tf.shape(y_true)[0], tf.dtypes.float32) # Bi
  # #masked_count = tf.cast(tf.size(true_masked_values), DNN_PARAMS['custom_dtype']._compute_dtype) # Bi * 64 * 64 * (1-fi)
  # #batch_size_ts = tf.cast(tf.shape(y_true)[0], DNN_PARAMS['custom_dtype']._compute_dtype) # Bi
  # masked_count = tf.cast(tf.size(true_masked_values), DNN_PARAMS['custom_dtype']._variable_dtype) # Bi * 64 * 64 * (1-fi)
  
  ###
  ## [B, H, W]
  reversedMask = tf.constant(1, dtype=DNN_PARAMS['custom_dtype']._variable_dtype) - mask
  
  ## [] # B * H * W * (1-fi)
  masked_count = tf.math.reduce_sum(reversedMask)
  ###
  
  ## [B, H, W, 1, 1]
  reversed_mask_5D = tf.expand_dims(tf.expand_dims(reversedMask, axis=-1), axis=-1)
  
  #
  batch_size_ts = tf.cast(tf.shape(y_true)[0], DNN_PARAMS['custom_dtype']._variable_dtype)  # Bi
  
  # reshape to 2D [B, H*W]
  # ref_2D = tf.reshape(ref, (batch_size_ts, -1))
  # y_true_2D = tf.reshape(y_true, (batch_size_ts, -1))
  shp_2D = (tf.cast(batch_size_ts, tf.int32), tf.constant(-1, dtype=tf.int32))
  ref_2D = tf.reshape(ref, shp_2D)
  y_true_2D = tf.reshape(y_true, shp_2D)
  
  # correlation coefficient between ref and y_true (complete target)
  ## [B, B]
  # corr_matrix = tfp.stats.correlation(ref_2D, y_true_2D, sample_axis = 1, event_axis = 0)
  ## [B]
  # corr_1D = tf.linalg.tensor_diag_part(corr_matrix)
  
  ## [B]
  corr_1D = tfp.stats.correlation(ref_2D, y_true_2D, sample_axis=1, event_axis=None)
  
  # abs_corr_1D = tf.math.abs(corr_1D)
  # abs_corr_1D = tf.cast(tf.math.abs(corr_1D), dtype=DNN_PARAMS['custom_dtype']._compute_dtype)
  abs_corr_1D = tf.cast(tf.math.abs(corr_1D), dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
  
  # mosaic the unmasked part of y_true and masked part of output.
  ## [B, H, W]
  # mosaic_unmsk_true_msk_pred = y_true * mask + y_pred * (1.0 - mask)
  # mosaic_unmsk_true_msk_pred = y_true * mask + y_pred * (tf.constant(1.0, dtype=tf.float32) - mask)
  # mosaic_unmsk_true_msk_pred = y_true * mask + y_pred * (tf.constant(1.0, dtype=DNN_PARAMS['custom_dtype']._compute_dtype) - mask)
  # mosaic_unmsk_true_msk_pred = y_true * mask + y_pred * (
  #          tf.constant(1.0, dtype=DNN_PARAMS['custom_dtype']._variable_dtype) - mask)
  
  # calculate the edge loss between the mosaic image and y_true
  ## the unmasked part's edge loss can cancel out, therefore don't need to do the calculation on masked area.
  ## instead do the calculation on the entire image.
  # [B, H, W, 2]
  # y_stacked = tf.stack([y_true, mosaic_unmsk_true_msk_pred], axis=-1)
  y_stacked = tf.stack([y_true, y_pred], axis=-1)
  
  # [B, H, W, 2, 2]
  # edges = tf.image.sobel_edges(y_stacked) * tf.math.sqrt( tf.reshape(abs_corr_1D, (-1, 1, 1, 1, 1)) )
  #edges = tf.image.sobel_edges(y_stacked)
  edges = sobel_edges_tfpad(y_stacked)
  edges = edges * reversed_mask_5D * tf.math.sqrt(
    tf.reshape(abs_corr_1D, tf.constant((-1, 1, 1, 1, 1), dtype=tf.int32)))
  
  # [B, H, W,    2]
  edges_masked_SSE_ts = tf.math.reduce_sum(
    tf.math.square(edges[:, :, :, 0, :] - edges[:, :, :, 1, :])
  ) / tf.constant(2.0, dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
  
  #
  edges_masked_MSE_ts = tf.math.divide(
    tf.math.multiply(
      tf.math.divide(edges_masked_SSE_ts, masked_count),
      batch_size_ts
    ),
    # tf.constant(DNN_PARAMS['global_batch_size'], dtype=tf.float32)
    # tf.constant(DNN_PARAMS['global_batch_size'], dtype=DNN_PARAMS['custom_dtype']._compute_dtype)
    tf.constant(DNN_PARAMS['global_batch_size'], dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
  )
  
  return edges_masked_MSE_ts


# def weighted_loss(mask_weight = 1.0, unmask_weight = 1.0, edge_weight = 0.1):
#   # y_true_stacked: [B, H, W, 2]
#   ## 0: y_true:     [B, H, W]
#   ## 1: mask:       [B, H, W]
#   # y_pred_stacked: [B, H, W, 2]
#   ## 0: y_pred:     [B, H, W]
#   ## 1: ref:        [B, H, W]
#   def loss(y_true_stacked, y_pred_stacked):
#     unmasked_MSE = calc_unmasked_MSE(y_true_stacked, y_pred_stacked)
#     masked_MSE = calc_masked_MSE(y_true_stacked, y_pred_stacked)
#
#     orig_out_img_edge_unmasked_MSE = calc_orig_out_img_edge_unmasked_MSE(y_true_stacked, y_pred_stacked)
#     mosaic_out_img_edge_masked_MSE = calc_mosaic_out_img_edge_masked_MSE(y_true_stacked, y_pred_stacked)
#
#     weighted_total_MSE = mask_weight * masked_MSE + unmask_weight * unmasked_MSE + \
#                           edge_weight * (orig_out_img_edge_unmasked_MSE + mosaic_out_img_edge_masked_MSE)
#
#     return weighted_total_MSE
#   return loss

class weighted_loss_class(tf.keras.losses.Loss):
  def __init__(self, mask_weight=1.0, unmask_weight=1.0, edge_weight=0.1, **kwargs):
    self.mask_weight = mask_weight
    self.unmask_weight = unmask_weight
    self.edge_weight = edge_weight
    super().__init__(**kwargs)

  def call(self, y_true_stacked, y_pred_stacked):
    # y_true_stacked: [B, H, W, 2]
    ## 0: y_true:     [B, H, W]
    ## 1: mask:       [B, H, W]
    # y_pred_stacked: [B, H, W, 2]
    ## 0: y_pred:     [B, H, W]
    ## 1: ref:        [B, H, W]
    unmasked_MSE = calc_unmasked_MSE(y_true_stacked, y_pred_stacked)
    masked_MSE = calc_masked_MSE(y_true_stacked, y_pred_stacked)

    orig_out_img_edge_unmasked_MSE = calc_orig_out_img_edge_unmasked_MSE(y_true_stacked, y_pred_stacked)
    mosaic_out_img_edge_masked_MSE = calc_mosaic_out_img_edge_masked_MSE(y_true_stacked, y_pred_stacked)

    weighted_total_MSE = self.mask_weight * (masked_MSE + self.edge_weight * mosaic_out_img_edge_masked_MSE) + \
                         self.unmask_weight * (unmasked_MSE + self.edge_weight * orig_out_img_edge_unmasked_MSE)
                         
    # weighted_total_MSE = self.mask_weight * masked_MSE + self.unmask_weight * unmasked_MSE + \
    #                      self.edge_weight * (orig_out_img_edge_unmasked_MSE + mosaic_out_img_edge_masked_MSE)

    return weighted_total_MSE

  def get_config(self):
    config = super().get_config()
    config.update({
      "mask_weight": self.mask_weight,
      "unmask_weight": self.unmask_weight,
      "edge_weight": self.edge_weight
    })
    return config


## keras regularization loss (metric)
#def model_regularization_loss(model_losses):
#  def regularization_loss(y_true_stacked, y_pred_stacked):
#    return tf.math.add_n(model_losses)
#  return regularization_loss