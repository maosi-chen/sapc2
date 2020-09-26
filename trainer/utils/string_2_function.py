import tensorflow as tf

from ..parameters.config import DNN_PARAMS

def string_to_activation(params):

  if params['activation_function'] == 'relu':
      PConv_activation = tf.keras.activations.relu
  elif params['activation_function'] == 'leaky_relu':
      act_fn_param_dict = params['activation_parameter_dict']
      #PConv_activation = tf.keras.layers.LeakyReLU(alpha = .3)
      #PConv_activation = tf.keras.layers.LeakyReLU(**act_fn_param_dict)
      PConv_activation = tf.keras.layers.LeakyReLU(dtype=DNN_PARAMS['custom_dtype'], **act_fn_param_dict)
  elif params['activation_function'] == 'prelu':
      initializer1 = tf.keras.initializers.Constant(params['activation_parameter_dict']['init_alpha'])
      regularizer1 = tf.keras.regularizers.l2(l=params['regularization_factor'])
      PConv_activation = tf.keras.layers.PReLU(alpha_initializer=initializer1, alpha_regularizer=regularizer1)
  else:
      PConv_activation = None

  return PConv_activation

def string_to_regularizer(params, regularization_function_key='regularization_function', name_prefix=''):

  if params[regularization_function_key] == 'l1':
    regularization_function = tf.keras.regularizers.l1(l=params[name_prefix + 'l1_regularization_factor'])
  elif params[regularization_function_key] == 'l2':
    regularization_function = tf.keras.regularizers.l2(l=params[name_prefix + 'l2_regularization_factor'])
  elif params[regularization_function_key] == 'l1l2':
    regularization_function = tf.keras.regularizers.l1_l2(
      l1=params[name_prefix + 'l1_regularization_factor'],
      l2=params[name_prefix + 'l2_regularization_factor'])
  else:
    regularization_function = None

  return regularization_function

def string_to_regularizer_v2(in_reg_func_name, x, in_l1_factor=1e-3, in_l2_factor=1e-3):

  if in_reg_func_name == 'l1':
    return tf.keras.regularizers.l1(l=in_l1_factor)(x)
  elif in_reg_func_name == 'l2':
    return tf.keras.regularizers.l2(l=in_l2_factor)(x)
  elif in_reg_func_name == 'l1l2':
    return tf.keras.regularizers.l1_l2(l1=in_l1_factor, l2=in_l2_factor)(x)
  else:
    return tf.constant(0.0, dtype=DNN_PARAMS['custom_dtype']._compute_dtype)




