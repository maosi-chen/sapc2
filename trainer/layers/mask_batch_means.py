import tensorflow as tf

from ..parameters.config import DNN_PARAMS

def get_batch_mean_variance(tfInput, tfMask):
    
  # for mean
  sum_mi = tf.math.reduce_sum(tfMask, axis = [1,2,3])

  weighted_x = tfMask * tfInput

  sum_weighted_x = tf.math.reduce_sum(weighted_x, axis = [1,2,3])

  batch_means = sum_weighted_x / sum_mi

  # for variance
  masked_means = tf.reshape(batch_means,(-1,1,1,1)) * tfMask

  batch_variances = tf.math.reduce_sum(tf.math.square(weighted_x - masked_means), axis = [1,2,3]) / sum_mi

  return batch_means, batch_variances

def rescale_based_on_masked_img(tbr_image, masked_image, mask):

  batch_means, batch_variances = get_batch_mean_variance(masked_image, mask)

  tbr_original_batch_means = tf.math.reduce_mean(tbr_image, axis = [1,2,3])
  tbr_original_batch_variances = tf.math.reduce_variance(tbr_image, axis = [1,2,3])

  mean_corrections = batch_means - tbr_original_batch_means
  variance_corrections = batch_variances / tbr_original_batch_variances
  sd_corrections = tf.math.sqrt(variance_corrections)

  corrected_image = ((tbr_image - tf.reshape(tbr_original_batch_means, (-1,1,1,1))) * tf.reshape(sd_corrections, (-1,1,1,1))) + tf.reshape(batch_means, (-1,1,1,1))

  return corrected_image

def rescale_based_on_masked_img_v2(tbr_image, masked_image, mask):

  batch_means, batch_variances = get_batch_mean_variance(masked_image, mask)

  tbr_original_unmasked_batch_means, tbr_original_unmasked_batch_variances = \
    get_batch_mean_variance(tbr_image, mask)

  mean_corrections = batch_means - tbr_original_unmasked_batch_means
  variance_corrections = batch_variances / tbr_original_unmasked_batch_variances
  sd_corrections = tf.math.sqrt(variance_corrections)

  tbr_original_batch_means = tf.math.reduce_mean(tbr_image, axis = [1,2,3])

  corrected_image = (
    (tbr_image - tf.reshape(tbr_original_batch_means, (-1,1,1,1))) \
     * tf.reshape(sd_corrections, (-1,1,1,1))) \
     + tf.reshape(tbr_original_batch_means, (-1,1,1,1)) \
     + tf.reshape(mean_corrections, (-1,1,1,1))
  
  #print("shape of corrected_image", tf.shape(corrected_image))
  #print("corrected_image", corrected_image)
  #tf.print("[exec] shape of corrected_image", tf.shape(corrected_image))
  #tf.print("[exec] corrected_image", corrected_image)
  
  return corrected_image

@tf.function(input_signature=(
  (tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 1], dtype=DNN_PARAMS['custom_dtype']._compute_dtype),
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width']], dtype=DNN_PARAMS['custom_dtype']._compute_dtype),
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width']], dtype=DNN_PARAMS['custom_dtype']._compute_dtype)
  ),)
)
def rescale_based_on_masked_img_wrap(x):

  (tbr_image, masked_image, mask) = x

  #tbr_input is 3d, so it must be reshaped
  masked_image_shape = tf.shape(masked_image)
  masked_image = tf.reshape(masked_image, (-1, masked_image_shape[1], masked_image_shape[2], 1))

  #tbr_input is 3d, so it must be reshaped
  mask_shape = tf.shape(mask)
  mask = tf.reshape(mask, (-1, mask_shape[1], mask_shape[2], 1))

  return rescale_based_on_masked_img_v2(tbr_image, masked_image, mask)



class rescale_based_on_masked_img_layer(tf.keras.layers.Layer):
  def __init__(self, spatial_dim_H, dtype='float32', *args, **kwargs):
    super().__init__(dtype=dtype, *args, **kwargs)
    self.input_spec = [tf.keras.layers.InputSpec(ndim=4), tf.keras.layers.InputSpec(ndim=3), tf.keras.layers.InputSpec(ndim=3)]
    self.spatial_dim_H = spatial_dim_H
    
    self.img_dyn_shp_raw = (None, self.spatial_dim_H, self.spatial_dim_H, 1)
    self.TSpec = tf.TensorSpec(shape=self.img_dyn_shp_raw, dtype=self._compute_dtype)
    
  def build(self, input_shape):
    self.get_batch_mean_variance_ins = self.get_batch_mean_variance.get_concrete_function(self.TSpec, self.TSpec)
    
    self.built = True
  
  @tf.function
  def get_batch_mean_variance(self, tfInput, tfMask):
    # for mean
    sum_mi = tf.math.reduce_sum(tfMask, axis=[1, 2, 3])
  
    weighted_x = tfMask * tfInput
  
    sum_weighted_x = tf.math.reduce_sum(weighted_x, axis=[1, 2, 3])
  
    batch_means = sum_weighted_x / sum_mi
  
    # for variance
    masked_means = tf.reshape(batch_means, (-1, 1, 1, 1)) * tfMask
  
    batch_variances = tf.math.reduce_sum(tf.math.square(weighted_x - masked_means), axis=[1, 2, 3]) / sum_mi
  
    return batch_means, batch_variances
  
  def call(self, inputs, training=True):
    (tbr_image, masked_image, mask) = inputs

    tbr_image = tf.cast(tbr_image, dtype=self._compute_dtype)
    masked_image = tf.cast(masked_image, dtype=self._compute_dtype)
    mask = tf.cast(mask, dtype = self._compute_dtype)

    # tbr_input is 3d, so it must be reshaped
    masked_image = tf.reshape(masked_image, (-1, self.spatial_dim_H, self.spatial_dim_H, 1))
    
    # tbr_input is 3d, so it must be reshaped
    mask = tf.reshape(mask, (-1, self.spatial_dim_H, self.spatial_dim_H, 1))

    batch_means, batch_variances = self.get_batch_mean_variance_ins(masked_image, mask)

    tbr_original_unmasked_batch_means, tbr_original_unmasked_batch_variances = \
      self.get_batch_mean_variance_ins(tbr_image, mask)

    mean_corrections = batch_means - tbr_original_unmasked_batch_means
    variance_corrections = batch_variances / tbr_original_unmasked_batch_variances
    sd_corrections = tf.math.sqrt(variance_corrections)

    tbr_original_batch_means = tf.math.reduce_mean(tbr_image, axis=[1, 2, 3])

    corrected_image = tf.math.add(
      (tbr_image - tf.reshape(tbr_original_batch_means, (-1, 1, 1, 1))) \
      * tf.reshape(sd_corrections, (-1, 1, 1, 1)),
      tf.reshape(tf.math.add(tbr_original_batch_means, mean_corrections), (-1, 1, 1, 1))
    )
    
    return corrected_image
  
  def get_config(self):
    config = super().get_config()
    config.update({
      "spatial_dim_H": self.spatial_dim_H,
      "dtype": self._dtype_policy, #self.dtype
    })
    return config
