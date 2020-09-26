import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn

from ..parameters.config import DNN_PARAMS

@tf.function(input_signature=[
  tf.TensorSpec(shape=[None, DNN_PARAMS['in_img_width'], DNN_PARAMS['in_img_width'], 2], dtype=DNN_PARAMS['custom_dtype']._variable_dtype),
])
def sobel_edges_tfpad(image):
  """Returns a tensor holding Sobel edge maps.
     Note: image is CONSTANT(0) padded instead of REFLECT padded.
  Arguments:
    image: Image tensor with shape [batch_size, h, w, d] and type float32 or
      float64.  The image(s) must be 2x2 or larger.
  Returns:
    Tensor holding edge maps for each channel. Returns a tensor with shape
    [batch_size, h, w, d, 2] where the last two dimensions hold [[dy[0], dx[0]],
    [dy[1], dx[1]], ..., [dy[d-1], dx[d-1]]] calculated using the Sobel filter.
  """
  # Define vertical and horizontal Sobel filters.
  static_image_shape = image.get_shape()
  image_shape = array_ops.shape(image)

  num_kernels = 2
  sobel_kn = tf.constant(
      [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
       [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=tf.float32)
  kernels_tf = tf.expand_dims(tf.transpose(sobel_kn, perm=(1,2,0)), axis=-2)

  kernels_tf = array_ops.tile(
      kernels_tf, [1, 1, image_shape[-1], 1], name='sobel_filters')

  # Use depth-wise convolution to calculate edge maps per channel.
  #pad_sizes = [[0, 0], [1, 1], [1, 1], [0, 0]]
  pad_sizes = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=tf.int32)
  #padded = array_ops.pad(image, pad_sizes, mode='REFLECT')
  #padded = tf.pad(tensor=image, paddings=pad_sizes, mode='REFLECT')
  padded = tf.pad(tensor=image, paddings=pad_sizes, mode='CONSTANT')

  # Output tensor has shape [batch_size, h, w, d * num_kernels].
  strides = [1, 1, 1, 1]
  output = nn.depthwise_conv2d(padded, kernels_tf, strides, 'VALID')

  # Reshape to [batch_size, h, w, d, num_kernels].
  shape = array_ops.concat([image_shape, [num_kernels]], 0)
  output = array_ops.reshape(output, shape=shape)
  output.set_shape(static_image_shape.concatenate([num_kernels]))
  return output




