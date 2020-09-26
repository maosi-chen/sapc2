import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils


class normal_c0_top1_exp_sigma(tf.keras.layers.Layer):
  """
  Normal distribution activation function,
    center location (mu) is fixed to 0,
    f(x) is scale to 1 at the center location;
    standard deviation (sigma) is derived from the exponential function with trainable parameters;

  ```
  f(x) = exp( -1.0 * ((x-mu)^2) / (2*sigma^2) )
  sigma = exp(w * alpha)
  ```
  where
  `mu` is constant 0.0
  `alpha` is fixed parameter (e.g., 10.0).
  `w` is the trainable 1D array with its lengths equal to number of features of x

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as the input.

  Arguments:
    w_initializer: Initializer for the weights.
    w_regularizer: Regularizer for the weights.
    w_constraint: Constraint for the weights.

    mu: the center of the normal distribution, default: 0.0.
    alpha: the scaling factor in calculation of sigma
      (the standard deviation of the the normal distribution),
      default: 1.0.

    shared_axes: The axes along which to share learnable
      parameters for the activation function.
      For example,
        if the incoming feature maps are from a 2D convolution
        with output shape `(batch, height, width, channels)`,
        and you wish to share parameters across space
        so that each filter only has one set of parameters,
        set `shared_axes=[1, 2]`.
        if the incoming feature maps are from a 1D convolution
        with output shape `(batch, step, channels)`,
        and you wish to share parameters across space
        so that each filter only has one set of parameters,
        set `shared_axes=[1]`.

  """

  def __init__(self,
               w_initializer='zeros',
               w_regularizer=None,
               w_constraint=None,
               mu=0.0,
               alpha=1.0,
               shared_axes=None,
               **kwargs):
    super().__init__(**kwargs)

    self.w_initializer = tf.keras.initializers.get(w_initializer)
    self.w_regularizer = tf.keras.regularizers.get(w_regularizer)
    self.w_constraint = tf.keras.constraints.get(w_constraint)

    self.mu = mu
    self.tf_mu = tf.constant(mu, dtype=self._compute_dtype)

    self.alpha = alpha
    self.tf_alpha = tf.constant(alpha, dtype=self._compute_dtype)

    if shared_axes is None:
      self.shared_axes = None
    elif not isinstance(shared_axes, (list, tuple)):
      self.shared_axes = [shared_axes]
    else:
      self.shared_axes = list(shared_axes)

    # constants
    self.tf_n1 = tf.constant(-1.0, dtype=self._compute_dtype)
    self.tf_p2 = tf.constant( 2.0, dtype=self._compute_dtype)

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    param_shape = list(input_shape[1:])
    if self.shared_axes is not None:
      for i in self.shared_axes:
        param_shape[i - 1] = 1
    # add the batch dim with size of 1
    param_shape = (1,) + tuple(param_shape)
    self.w = self.add_weight(
      shape=param_shape,
      name='normal_c0_top1_exp_sigma__w',
      initializer=self.w_initializer,
      regularizer=self.w_regularizer,
      constraint=self.w_constraint)
    # Set input spec
    axes = {}
    if self.shared_axes:
      for i in range(1, len(input_shape)):
        if i not in self.shared_axes:
          axes[i] = input_shape[i]
    self.input_spec = tf.keras.layers.InputSpec(ndim=len(input_shape), axes=axes)
    self.built = True

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
      'w_initializer': tf.keras.initializers.serialize(self.w_initializer),
      'w_regularizer': tf.keras.regularizers.serialize(self.w_regularizer),
      'w_constraint': tf.keras.constraints.serialize(self.w_constraint),
      'mu': self.mu,
      'alpha': self.alpha,
      'shared_axes': self.shared_axes
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs):

    x = tf.cast(inputs, dtype=self._compute_dtype)

    #sigma = exp(w * alpha)
    #exp( -1.0 * ((x-mu)^2) / (2*sigma^2) )

    # [1, 1, 1, C] or [1, 1, C] or [1, C]
    sigma = tf.math.exp(self.w * self.tf_alpha)
    # [B, H, W, C] or [B, STEP, C], or [B, C]
    y = tf.math.exp( self.tf_n1 * tf.pow((x-self.tf_mu), self.tf_p2) / (self.tf_p2 * tf.pow(sigma, self.tf_p2)) )

    return y







class Emxsq(tf.keras.layers.Layer):
  """
  Parametric exponential of negative x square
  
  ```
  f(x) = exp(- alpha * x^2)
  ```
  where `alpha` is a learned array with the same shape as x.
  
  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.
    
  Output shape:
    Same shape as the input.
    
  Arguments:
    alpha_initializer: Initializer function for the weights.
    alpha_regularizer: Regularizer for the weights.
    alpha_constraint: Constraint for the weights.
    shared_axes: The axes along which to share learnable
      parameters for the activation function.
      For example,
        if the incoming feature maps are from a 2D convolution
        with output shape `(batch, height, width, channels)`,
        and you wish to share parameters across space
        so that each filter only has one set of parameters,
        set `shared_axes=[1, 2]`.
        if the incoming feature maps are from a 1D convolution
        with output shape `(batch, step, channels)`,
        and you wish to share parameters across space
        so that each filter only has one set of parameters,
        set `shared_axes=[1]`.
  
  """
  def __init__(self,
               alpha_initializer='zeros',
               alpha_regularizer=None,
               alpha_constraint=None,
               shared_axes=None,
               **kwargs):
    super().__init__(**kwargs)
    self.supports_masking = True
    self.alpha_initializer = tf.keras.initializers.get(alpha_initializer)
    self.alpha_regularizer = tf.keras.regularizers.get(alpha_regularizer)
    self.alpha_constraint = tf.keras.constraints.get(alpha_constraint)
    if shared_axes is None:
      self.shared_axes = None
    elif not isinstance(shared_axes, (list, tuple)):
      self.shared_axes = [shared_axes]
    else:
      self.shared_axes = list(shared_axes)

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    param_shape = list(input_shape[1:])
    if self.shared_axes is not None:
      for i in self.shared_axes:
        param_shape[i - 1] = 1
    self.alpha = self.add_weight(
      shape=param_shape,
      name='alpha',
      initializer=self.alpha_initializer,
      regularizer=self.alpha_regularizer,
      constraint=self.alpha_constraint)
    # Set input spec
    axes = {}
    if self.shared_axes:
      for i in range(1, len(input_shape)):
        if i not in self.shared_axes:
          axes[i] = input_shape[i]
    self.input_spec = tf.keras.layers.InputSpec(ndim=len(input_shape), axes=axes)
    self.built = True

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape
  
  def get_config(self):
    config = {
        'alpha_initializer': tf.keras.initializers.serialize(self.alpha_initializer),
        'alpha_regularizer': tf.keras.regularizers.serialize(self.alpha_regularizer),
        'alpha_constraint': tf.keras.constraints.serialize(self.alpha_constraint),
        'shared_axes': self.shared_axes
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs):
    # f(x) = exp(- alpha * x^2)
    x = tf.cast(inputs, dtype=self._compute_dtype)
    y = tf.math.exp(- self.alpha * tf.math.square(x))
    return y
  
  