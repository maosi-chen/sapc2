import tensorflow as tf
import numpy as np
import random


DNN_PARAMS = {
  'in_img_width': 64,
  'custom_dtype': tf.keras.mixed_precision.experimental.Policy("mixed_float16", loss_scale="dynamic"),
}

def set_dnn_params(in_dict):
  DNN_PARAMS.update(in_dict)
  
def reset_random_seeds():
  #os.environ['PYTHONHASHSEED'] = str(1)
  random.seed(42)
  np.random.seed(42)
  tf.random.set_seed(42)

