import tensorflow as tf

## ----------------------------------------------- ##
## Lambda functions

class out_stack_layer(tf.keras.layers.Layer):
  def __init__(self, axis, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.input_spec = [tf.keras.layers.InputSpec(ndim=3), tf.keras.layers.InputSpec(ndim=3)]
    self.axis = axis

  def call(self, inputs, training=True):
    input_0 = tf.cast(inputs[0], dtype=self._compute_dtype)
    input_1 = tf.cast(inputs[1], dtype=self._compute_dtype)
    x = tf.stack([input_0, input_1], axis=self.axis)
    return x
  
  def get_config(self):
    config = super().get_config()
    config.update({
      "axis": self.axis
    })
    return config

class tensor_copy_layer(tf.keras.layers.Layer):
  def __init__(self, ndim=4, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.input_spec = [tf.keras.layers.InputSpec(ndim=ndim)]
    self.ndim = ndim

  def call(self, inputs, training=True):
    inputs = tf.cast(inputs, dtype=self._compute_dtype)
    x = tf.identity(inputs)
    return x
  
  def get_config(self):
    config = super().get_config()
    config.update({
      "ndim": self.ndim
    })
    return config

## ----------------------------------------------- ##

class slice_copy_layer(tensor_copy_layer):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.input_spec = [tf.keras.layers.InputSpec(ndim=self.ndim + 1)]

  def call(self, inputs, training=True):
    inputs = tf.cast(inputs, dtype=self._compute_dtype)
    x = tf.identity(inputs[:, :, :, 0])
    return x
  
  def get_config(self):
    config = super().get_config()
    return config
  