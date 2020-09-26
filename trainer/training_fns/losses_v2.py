import tensorflow as tf
import tensorflow_probability as tfp

from ..utils.tf_utility import sobel_edges_tfpad

from ..parameters.config import DNN_PARAMS

## Loss/Metrics functions

@tf.function(input_signature=[
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[], dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[], dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
])
def calc_MSE(
  y_true_stacked,
  y_pred_stacked,
  mask_weight=tf.constant(1.0, dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  unmask_weight=tf.constant(1.0, dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
  ):
  # y_true_stacked: [B, H, W, 2]
  ## 0: y_true:     [B, H, W]
  ## 1: mask:       [B, H, W]
  # y_pred_stacked: [B, H, W, 2]
  ## 0: y_pred:     [B, H, W]
  ## 1: ref:        [B, H, W]
  y_true = y_true_stacked[:,:,:,0]
  mask   = y_true_stacked[:,:,:,1]
  y_pred = y_pred_stacked[:,:,:,0]

  ###
  ## [B, H, W]
  mask = tf.cast(mask, dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
  
  ## N1: unmasked count, []
  ## N0: masked count, []
  N1 = unmasked_count = tf.math.reduce_sum(mask)
  N0 = masked_count = tf.cast(tf.size(mask), dtype=DNN_PARAMS['custom_dtype']._variable_dtype) - N1
  
  ## rWN: ratio of w/N depending on mask
  rWN = tf.where(
    tf.cast(mask, tf.bool),
    x=unmask_weight/N1,
    y=mask_weight/N0
  )
  
  ### Squared Error
  ## [B, H, W]
  SE = tf.math.square(y_pred - y_true)
  
  ### weighted MSE
  ## []
  wMSE = tf.math.reduce_sum(rWN * SE)
  
  ## []
  batch_size_ts = tf.cast(tf.shape(y_true)[0], DNN_PARAMS['custom_dtype']._variable_dtype)  # Bi

  ## weighted MSE corrected by global_bach_size
  wMSE_glb_ts = tf.math.divide(
    tf.math.multiply(
      wMSE,
      batch_size_ts
    ),
    tf.constant(DNN_PARAMS['global_batch_size'], dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
  )
  return wMSE_glb_ts


@tf.function(input_signature=[
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[], dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  tf.TensorSpec(shape=[], dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
])
def calc_edge_MSE(
  y_true_stacked,
  y_pred_stacked,
  mask_edge_weight=tf.constant(0.48, dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
  unmask_edge_weight=tf.constant(0.48, dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
):
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

  ###
  ## [B, H, W]
  mask = tf.cast(mask, dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
  
  ## N1: unmasked count, []
  ## N0: masked count, []
  N1 = unmasked_count = tf.math.reduce_sum(mask)
  N0 = masked_count = tf.cast(tf.size(mask), dtype=DNN_PARAMS['custom_dtype']._variable_dtype) - N1
  
  #
  batch_size_ts = tf.cast(tf.shape(y_true)[0], DNN_PARAMS['custom_dtype']._variable_dtype) # Bi

  # reshape to 2D [B, H*W]
  # ref_2D = tf.reshape(ref, (batch_size_ts, -1))
  # y_true_2D = tf.reshape(y_true, (batch_size_ts, -1))
  shp_2D = (tf.cast(batch_size_ts, tf.int32), tf.constant(-1, dtype=tf.int32))
  ref_2D = tf.reshape(ref, shp_2D)
  y_true_2D = tf.reshape(y_true, shp_2D)

  # correlation coefficient between ref and y_true (complete target)
  ## [B]
  corr_1D = tfp.stats.correlation(ref_2D, y_true_2D, sample_axis = 1, event_axis = None)
  abs_corr_1D = tf.cast(tf.math.abs(corr_1D), dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
  ## [B, 1, 1,   1]
  abs_corr_5D = tf.reshape(abs_corr_1D, tf.constant((-1, 1, 1,   1), dtype=tf.int32))
  
  ## rWNC: ratio of w/N depending on mask and abs. corr. coef. b/w y_true and ref
  ## [B, H, W,   1]
  rWNC = tf.where(
    tf.cast(tf.expand_dims(mask, axis=-1), tf.bool),
    x=tf.ones_like(abs_corr_5D) * unmask_edge_weight / N1,  # true:  1, unmasked
    y=             abs_corr_5D  * mask_edge_weight   / N0   # false: 0, masked
  )
  
  # [B, H, W, 2]
  y_stacked = tf.stack([y_true, y_pred], axis=-1)

  # [B, H, W, 2, 2]
  #edges = tf.image.sobel_edges(y_stacked)
  edges = sobel_edges_tfpad(y_stacked)
  
  # Squared Error
  # [B, H, W,    2]
  edge_SE = tf.math.square(edges[:, :, :, 0, :] - edges[:, :, :, 1, :]) / \
            tf.constant(2.0, dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
  
  # weighted MSE
  ## []
  w_edge_MSE = tf.math.reduce_sum(rWNC * edge_SE)

  #
  w_edge_MSE_glb_ts = tf.math.divide(
    tf.math.multiply(
      w_edge_MSE,
      batch_size_ts
    ),
    tf.constant(DNN_PARAMS['global_batch_size'], dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
  )

  return w_edge_MSE_glb_ts


class weighted_loss_class(tf.keras.losses.Loss):
  def __init__(self, mask_weight=1.0, unmask_weight=1.0, edge_weight=0.1, **kwargs):
    self.mask_weight = tf.convert_to_tensor(mask_weight, dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
    self.unmask_weight = tf.convert_to_tensor(unmask_weight, dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
    self.edge_weight = tf.convert_to_tensor(edge_weight, dtype=DNN_PARAMS['custom_dtype']._variable_dtype)
    super().__init__(**kwargs)

  def call(self, y_true_stacked, y_pred_stacked):
    # y_true_stacked: [B, H, W, 2]
    ## 0: y_true:     [B, H, W]
    ## 1: mask:       [B, H, W]
    # y_pred_stacked: [B, H, W, 2]
    ## 0: y_pred:     [B, H, W]
    ## 1: ref:        [B, H, W]

    cmb_pixel_MSE = calc_MSE(
      y_true_stacked, y_pred_stacked,
      mask_weight=self.mask_weight,
      unmask_weight=self.unmask_weight
      )
    
    cmb_edge_MSE = calc_edge_MSE(
      y_true_stacked, y_pred_stacked,
      mask_edge_weight=self.mask_weight * self.edge_weight,
      unmask_edge_weight=self.unmask_weight * self.edge_weight
    )

    weighted_total_MSE = cmb_pixel_MSE + cmb_edge_MSE

    return weighted_total_MSE

  def get_config(self):
    config = super().get_config()
    config.update({
      "mask_weight": self.mask_weight,
      "unmask_weight": self.unmask_weight,
      "edge_weight": self.edge_weight
    })
    return config
