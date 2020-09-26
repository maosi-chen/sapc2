# This file handles the training of the model created by model.py,
# and is the primary interface between GCP and our custom code

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import posixpath
import glob

import tensorflow as tf
from tensorflow.python.lib.io import file_io

from .model import create_MPR_Model_fnAPI
#from .data_preprocess_layer_v2 import create_combined_transformed_dataset_v3
from GCP_fnAPI_tf2.trainer.layers.data_preprocess_layer_v2 import (
  create_combined_transformed_dataset_v3d1_gs
)
from GCP_fnAPI_tf2.trainer.parameters.create_model_parameters_tf2 import create_params

from .merge_layer import MergeLayer
from .partial_merge_layer import PartialMergeLayer
from .partial_convolution_layer import PConv2D
from .reftgtencodelyr import RefTgtEncodeLyr
from .trainer.layers.reference_target_decoder_layer import DecodeLayer
from GCP_fnAPI_tf2.trainer.layers.data_preprocess_layer_v2 import missing_pixel_reconstruct_data_preprocess_layer_v2
##from .create_model_parameters_tfv1d13 import create_params


# Unmodified from the example, we can add our own args
def get_args():
  """Argument parser.
  Returns:
    Dictionary of arguments.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--job-dir', '-jd',
      type=str,
      required=True,
      help='local or GCS location for writing checkpoints and exporting models')
  parser.add_argument(
      '--data_dir', '-dd',
      type=str,
      required=True,
      help='local or GCS location for storing the train & test data')
  parser.add_argument(
      '--initial_epoch', '-initep',
      type=int,
      default=0,
      help='initial epoch of the training (default=0)')
  parser.add_argument(
      '--num_epochs', '-ne',
      type=int,
      default=5, #20,
      help='number of times to go through the data, default=20')
  parser.add_argument(
      '--batch_size', '-bs',
      default=128,
      type=int,
      help='number of records to read during each training step, default=128')
  parser.add_argument(
      '--learning_rate', '-lr',
      default=.01,
      type=float,
      help='learning rate for gradient descent, default=.01')
  parser.add_argument(
      '--sw_resume_training', '-swrt',
      type=lambda s: s.lower() in ['true', 't,', 'y', '1'],
      default=False,
      help='[default: %(default)s] whether to resume training from previous exported model')
  parser.add_argument(
      '--resume_model_source', '-rmsc',
      default='hdf', #'tf'
      type=str,
      help='If sw_resume_training, whether to restore the model in hdf5 format or in tf SavedModel format')
  parser.add_argument(
      '--verbosity', '-vb',
      choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
      default='INFO')
  args, _ = parser.parse_known_args()
  return args

# h5py workaround: copy local models over to GCS if the job_dir is GCS.
def copy_file_to_gcs(job_dir, file_path):
  with file_io.FileIO(file_path, mode='rb') as input_f:
    with file_io.FileIO(posixpath.join(job_dir, file_path), mode='wb') as output_f: #'wb+'
      output_f.write(input_f.read())

def copy_file_from_gcs(job_dir, file_path):
  with file_io.FileIO(posixpath.join(job_dir, file_path), mode='rb') as input_f:
    with file_io.FileIO(file_path, mode='wb') as output_f: #'wb+'
      output_f.write(input_f.read())
  

# a workaround callback to move locally saved checkpoints to the gs bucket
class ExportCheckpointGS(tf.keras.callbacks.Callback):
  """
  """
  def __init__(self,
               job_dir
               ):
    self.job_dir = job_dir
    #self.ckpt_local_file_name = ckpt_local_file_name

  def get_checkpoint_FSNs(self):
    model_path_glob = 'weights.*'
    if not self.job_dir.startswith('gs://'):
      model_path_glob = os.path.join(self.job_dir, model_path_glob)
    checkpoints = glob.glob(model_path_glob)
    return checkpoints

  def on_epoch_begin(self, epoch, logs={}):
    """Compile and save model."""
    if epoch > 0:
      # Unhappy hack to work around h5py not being able to write to GCS.
      # Force snapshots and saves to local filesystem, then copy them over to GCS.
      #model_path_glob = 'weights.*'
      #if not self.job_dir.startswith('gs://'):
      #  model_path_glob = os.path.join(self.job_dir, model_path_glob)
      #checkpoints = glob.glob(model_path_glob)
      checkpoints = self.get_checkpoint_FSNs()
      if len(checkpoints) > 0:
        checkpoints.sort()
        if self.job_dir.startswith('gs://'):
          copy_file_to_gcs(self.job_dir, checkpoints[-1])

  def on_train_end(self, logs={}):
    checkpoints = self.get_checkpoint_FSNs()
    if len(checkpoints) > 0:
      checkpoints.sort()
      if self.job_dir.startswith('gs://'):
        copy_file_to_gcs(self.job_dir, checkpoints[-1])

        ## copy the checkpoint file (list of available checkpoints in the current directory)
        #copy_file_to_gcs(self.job_dir, 'checkpoint')

# Callback for printing the LR at the end of each epoch.
class PrintLR(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print ('\nLearning rate for epoch {} is {}'.format(
      epoch + 1, tf.keras.backend.get_value(self.model.optimizer.lr)))
      #epoch + 1, tf.keras.backend.get_value(fn_MPR_Model.optimizer.lr)))

# callback to print model's regularization loss at the begin/end of an epoch
class PrintRegularizationLoss(tf.keras.callbacks.Callback):
  def __init__(self):
    super().__init__()
    self.reg_loss = []

  def get_current_reg_loss(self):
    reg_loss_ts = tf.math.add_n(self.model.losses)
    reg_loss_val = tf.keras.backend.get_value(reg_loss_ts)
    return reg_loss_val

  #def on_train_begin(self, logs={}):
  #  self.reg_loss.append(self.get_current_reg_loss())
  #  print("regularization loss (on_train_begin): {}".format(self.reg_loss[-1]))

  def on_epoch_end(self, epoch, logs={}):
    self.reg_loss.append(self.get_current_reg_loss())
    print("regularization loss (epoch {}): {}".format(epoch, self.reg_loss[-1]))




#Train the model using our custom training loop
# args are supplied by the function above, this could be used for #epochs
def train(args):

  args = vars(args)
  #Train parameters, defined like in colab.

  #proj_root_path, transformed_data_RootPath, logdir  <-create in cloud bucket?
  #proj_root_path = 'gs://missing_pixel_reconstruct_data/'
  transformed_data_RootPath = args['data_dir'] #posixpath.join(proj_root_path, 'transformed_data_full') 'gs://missing_pixel_reconstruct_data/transformed_data_full'
  logdir = args['job_dir'] #transformed_data_RootPath # cannot create subdir
  # training parameters
  DNN_PARAMS = {
    # dataset
    'global_batch_size': args['batch_size'], #128,  # 'batch_size': 20, #100,
    'buffer_size': 100,  # None,
    'shuffle_seed': 42,
    'transformed_file_dir_train': os.path.join(transformed_data_RootPath, 'train'),
    'transformed_file_dir_test': os.path.join(transformed_data_RootPath, 'test'),
    'num_parallel_reads': 8,
    'num_parallel_calls': 8,
    # training loop
    'initial_epoch': args['initial_epoch'],
    'num_epochs': args['initial_epoch'] + args['num_epochs'],  # 5,
    'train_screen_print_freq': 100,
    'train_checkpoint_freq': 1000,
    'sw_resume_training': args['sw_resume_training'],
    'resume_model_source': args['resume_model_source'],
    # tensorboard log
    'logdir': logdir,
    # model
    'parameter_dict': create_params((None, 64, 64, 3), 64, "channels_last", (7, 7), (3, 3), True, 'relu',
                                    'leaky_relu', False, min_shape=3, regularization_factor=args['learning_rate']),
    'flt_fs': 64,
    'data_format': 'channels_last',
    'custom_dtype': tf.float32,
    'debug_print': False,
    # optimizer
    'learning_rate': args['learning_rate'], #5e-4,
    # loss
    'mask_weight': 6.0,
    'unmask_weight': 1.0,
    # ckpt & model save/restore
    'max_to_keep': 10,
    'epoch_ckpt_incr': 10000,
    'CHECKPOINT_FILE_NAME': 'weights.{epoch:02d}-{val_loss:.6f}.hdf5',
    'SAVED_MODEL_FILE_NAME': 'mpr.hdf5',
    'tf_SavedModel_FILE_PATH': os.path.join(logdir, 'tf_SavedModel')
  }
  
  ### ------------------------------------------------------ ###
  ### dataset counts
  ### ------------------------------------------------------ ###
  #total_train_examples = count_tfrecords_examples_gs(posixpath.join(DNN_PARAMS['transformed_file_dir_train'], "*.gz"), compression_type="GZIP", num_parallel_reads=16),
  total_train_examples = 5386500
  if ("small" in transformed_data_RootPath):
    total_train_examples /= 10
  print("total_train_examples", total_train_examples)
  #total_test_examples  = count_tfrecords_examples_gs(posixpath.join(DNN_PARAMS['transformed_file_dir_test'], "*.gz"),  compression_type="GZIP", num_parallel_reads=16),
  total_test_examples = 283500
  if ("small" in transformed_data_RootPath):
    total_test_examples = int(total_test_examples / 5.5)
  print("total_test_examples", total_test_examples)

  ### ------------------------------------------------------ ###
  ### Datasets
  ### ------------------------------------------------------ ###
  combined_transformed_dataset_v3_train = create_combined_transformed_dataset_v3d1_gs(
    os.path.join(DNN_PARAMS['transformed_file_dir_train'], "*.gz"),
    DNN_PARAMS['global_batch_size'],
    DNN_PARAMS['buffer_size'],
    shuffle_seed=DNN_PARAMS['shuffle_seed'],
    num_epochs=None, #1,
    num_parallel_reads=DNN_PARAMS['num_parallel_reads'],
    num_parallel_calls=DNN_PARAMS['num_parallel_calls'],
    flt_fs=DNN_PARAMS['flt_fs']  # 64
  )
  combined_transformed_dataset_v3_test = create_combined_transformed_dataset_v3d1_gs(
    os.path.join(DNN_PARAMS['transformed_file_dir_test'], "*.gz"),
    DNN_PARAMS['global_batch_size'],
    None,  # DNN_PARAMS['buffer_size']
    shuffle_seed=DNN_PARAMS['shuffle_seed'],
    num_epochs=None, #1,
    num_parallel_reads=DNN_PARAMS['num_parallel_reads'],
    num_parallel_calls=DNN_PARAMS['num_parallel_calls'],
    flt_fs=DNN_PARAMS['flt_fs']  # 64
  )

  ### ------------------------------------------------------ ###
  ### Loss/Metrics functions
  ### ------------------------------------------------------ ###
  def calc_unmasked_MSE(y_true_stacked, y_pred_stacked):
    # y_true_stacked: [B, H, W, 2]
    ## 0: y_true:     [B, H, W]
    ## 1: mask:       [B, H, W]
    # y_pred_stacked: [B, H, W, 2]
    ## 0: y_pred:     [B, H, W]
    ## 1: mask:       [B, H, W]
    y_true = y_true_stacked[:,:,:,0]
    mask   = y_true_stacked[:,:,:,1]
    y_pred = y_pred_stacked[:,:,:,0]
  
    unmasked_indices = tf.where(tf.cast(mask, tf.bool))
    true_unmasked_values = tf.reshape(tf.gather_nd(y_true, unmasked_indices), [-1])
    predicted_unmasked_values = tf.reshape(tf.gather_nd(y_pred, unmasked_indices), [-1])
  
    unmasked_count = tf.cast(tf.size(true_unmasked_values), tf.dtypes.float32) # Bi * 64 * 64 * fi 
    batch_size_ts = tf.cast(tf.shape(y_true)[0], tf.dtypes.float32) # Bi
  
    unmasked_SSE_ts = tf.math.reduce_sum(
      tf.math.square(predicted_unmasked_values - true_unmasked_values)
    )
    #unmasked_MSE_ts = (unmasked_SSE_ts / unmasked_count) * ( batch_size_ts / (1.0*FLAGS.global_batch_size) )
    unmasked_MSE_ts = tf.math.divide(
      tf.math.multiply(
        tf.math.divide(unmasked_SSE_ts, unmasked_count), 
        batch_size_ts
      ),
      tf.constant(DNN_PARAMS['global_batch_size'], dtype=tf.float32)
    )
    return unmasked_MSE_ts
  
  def calc_masked_MSE(y_true_stacked, y_pred_stacked):
    # y_true_stacked: [B, H, W, 2]
    ## 0: y_true:     [B, H, W]
    ## 1: mask:       [B, H, W]
    # y_pred_stacked: [B, H, W, 2]
    ## 0: y_pred:     [B, H, W]
    ## 1: mask:       [B, H, W]
    y_true = y_true_stacked[:,:,:,0]
    mask   = y_true_stacked[:,:,:,1]
    y_pred = y_pred_stacked[:,:,:,0]
    
    masked_indices = tf.where(tf.cast(1 - mask, tf.bool))
    true_masked_values = tf.reshape(tf.gather_nd(y_true, masked_indices), [-1])
    predicted_masked_values = tf.reshape(tf.gather_nd(y_pred, masked_indices), [-1])
  
    masked_count = tf.cast(tf.size(true_masked_values), tf.dtypes.float32) # Bi * 64 * 64 * (1-fi)
    batch_size_ts = tf.cast(tf.shape(y_true)[0], tf.dtypes.float32) # Bi
  
    #masked_MSE_ts = tf.losses.MSE(true_masked_values, predicted_masked_values)
    masked_SSE_ts = tf.math.reduce_sum(
      tf.math.square(predicted_masked_values - true_masked_values)
    )
    #masked_MSE_ts = (masked_SSE_ts / masked_count) * ( batch_size_ts / (1.0*FLAGS.global_batch_size) )
    masked_MSE_ts = tf.divide(
      tf.math.multiply(
        tf.math.divide(masked_SSE_ts, masked_count),
        batch_size_ts
      ),
      tf.constant(DNN_PARAMS['global_batch_size'], dtype=tf.float32)
    )
    return masked_MSE_ts
  
  def calc_unweighted_MSE(y_true_stacked, y_pred_stacked):
    # y_true_stacked: [B, H, W, 2]
    ## 0: y_true:     [B, H, W]
    ## 1: mask:       [B, H, W]
    # y_pred_stacked: [B, H, W, 2]
    ## 0: y_pred:     [B, H, W]
    ## 1: mask:       [B, H, W]
    y_true = y_true_stacked[:,:,:,0]
    y_pred = y_pred_stacked[:,:,:,0]
  
    unweighted_SSE_ts = tf.math.reduce_sum(
      tf.math.square(y_pred - y_true)
    )
    
    pixel_counts = tf.cast(tf.size(y_true), tf.dtypes.float32) # Bi * 64 * 64
    batch_size_ts = tf.cast(tf.shape(y_true)[0], tf.dtypes.float32) # Bi
    
    #unweighted_MSE_ts = (unweighted_SSE_ts / pixel_counts) * ( batch_size_ts / (1.0*FLAGS.global_batch_size) )
    unweighted_MSE_ts = tf.divide(
      tf.math.multiply(
        tf.math.divide(unweighted_SSE_ts, pixel_counts),
        batch_size_ts
      ),
      tf.constant(DNN_PARAMS['global_batch_size'], dtype=tf.float32)
    )
    return unweighted_MSE_ts
  
  def weighted_loss(mask_weight = 1, unmask_weight = 1):
    # y_true_stacked: [B, H, W, 2]
    ## 0: y_true:     [B, H, W]
    ## 1: mask:       [B, H, W]
    # y_pred_stacked: [B, H, W, 2]
    ## 0: y_pred:     [B, H, W]
    ## 1: mask:       [B, H, W]
    def loss(y_true_stacked, y_pred_stacked):
      unmasked_MSE = calc_unmasked_MSE(y_true_stacked, y_pred_stacked)
      masked_MSE = calc_masked_MSE(y_true_stacked, y_pred_stacked)
      weighted_total_MSE = mask_weight * masked_MSE + unmask_weight * unmasked_MSE
      return weighted_total_MSE
    return loss
  
  ## keras regularization loss (metric)
  #def model_regularization_loss(model_losses):
  #  def regularization_loss(y_true_stacked, y_pred_stacked):
  #    return tf.math.add_n(model_losses)
  #  return regularization_loss



  ### ------------------------------------------------------ ###
  ### create model
  ### ------------------------------------------------------ ###

  strategy = tf.distribute.MirroredStrategy()

  has_compiled = False
  
  print("DNN_PARAMS['sw_resume_training']: ", DNN_PARAMS['sw_resume_training'], DNN_PARAMS['sw_resume_training'] == True, type(DNN_PARAMS['sw_resume_training']))
  if (DNN_PARAMS['resume_model_source'] == 'hdf'):
    print( 'restore previous exported model (hdf) from {}'.format(os.path.join(DNN_PARAMS['logdir'], DNN_PARAMS['SAVED_MODEL_FILE_NAME'])) )
    if DNN_PARAMS['logdir'].startswith('gs://'):
      TBR_HDF_FWN = DNN_PARAMS['SAVED_MODEL_FILE_NAME']
      copy_file_from_gcs(DNN_PARAMS['logdir'], DNN_PARAMS['SAVED_MODEL_FILE_NAME'])
      with strategy.scope():
        fn_MPR_Model = create_MPR_Model_fnAPI(
          DNN_PARAMS['flt_fs'], #used_img_size
          DNN_PARAMS['custom_dtype'],
          DNN_PARAMS['data_format'],
          DNN_PARAMS['debug_print'],
          DNN_PARAMS['parameter_dict']
        )
        #
        optimizer = tf.keras.optimizers.Adam()
        # Specify the training configuration (optimizer, loss, metrics)
        fn_MPR_Model.compile(
          # Loss function to minimize
          loss=weighted_loss(
            mask_weight = DNN_PARAMS['mask_weight'], 
            unmask_weight = DNN_PARAMS['unmask_weight']), #keras.losses.SparseCategoricalCrossentropy(),
          # optimizer
          optimizer=optimizer,
          # List of metrics to monitor
          metrics=[
            calc_unweighted_MSE,
            calc_unmasked_MSE,
            calc_masked_MSE
            ]
          )
        fn_MPR_Model.load_weights( TBR_HDF_FWN )
      has_compiled = True
    else:
      TBR_HDF_FWN = os.path.join(DNN_PARAMS['logdir'], DNN_PARAMS['SAVED_MODEL_FILE_NAME'])
      with strategy.scope():
        fn_MPR_Model = tf.keras.models.load_model(
          TBR_HDF_FWN, #os.path.join(DNN_PARAMS['logdir'], DNN_PARAMS['SAVED_MODEL_FILE_NAME']),
          custom_objects={
            #'create_MPR_Model_fnAPI': create_MPR_Model_fnAPI,
            #'string_to_activation_v2': string_to_activation_v2,
            #'string_to_regularizer': string_to_regularizer,
            #'Is_full_mask': Is_full_mask,
            #'generate_random_masked_values': generate_random_masked_values,
            #'LeakyReLU': tf.keras.layers.LeakyReLU,
            'MergeLayer': MergeLayer,
            'PartialMergeLayer': PartialMergeLayer,
            'PConv2D': PConv2D,
            'RefTgtEncodeLyr': RefTgtEncodeLyr,
            'DecodeLayer': DecodeLayer,
            'missing_pixel_reconstruct_data_preprocess_layer_v2': missing_pixel_reconstruct_data_preprocess_layer_v2,
            'weighted_loss': weighted_loss,
            'calc_unweighted_MSE': calc_unweighted_MSE,
            'calc_masked_MSE': calc_masked_MSE,
            'calc_unmasked_MSE': calc_unmasked_MSE,
            'PrintRegularizationLoss': PrintRegularizationLoss,
            'PrintLR': PrintLR
          },
          compile=False
        )
    print('model loaded successfully. {}'.format(DNN_PARAMS['SAVED_MODEL_FILE_NAME']))
  elif (DNN_PARAMS['resume_model_source'] == 'tf'):
    print('restore previous exported model (tf SavedModel) from {}'.format(DNN_PARAMS['tf_SavedModel_FILE_PATH']))
    #strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
      fn_MPR_Model = tf.keras.experimental.load_from_saved_model(DNN_PARAMS['tf_SavedModel_FILE_PATH'], custom_objects={
        'MergeLayer': MergeLayer,
        'PartialMergeLayer': PartialMergeLayer,
        'PConv2D': PConv2D,
        'RefTgtEncodeLyr': RefTgtEncodeLyr,
        'DecodeLayer': DecodeLayer,
        'missing_pixel_reconstruct_data_preprocess_layer_v2': missing_pixel_reconstruct_data_preprocess_layer_v2,
        'weighted_loss': weighted_loss,
        'calc_unweighted_MSE': calc_unweighted_MSE,
        'calc_masked_MSE': calc_masked_MSE,
        'calc_unmasked_MSE': calc_unmasked_MSE,
        'PrintRegularizationLoss': PrintRegularizationLoss,
        'PrintLR': PrintLR
      })
      ## compile? yes, needed.
    print('model loaded successfully. {}'.format(DNN_PARAMS['tf_SavedModel_FILE_PATH']))
      


  if (has_compiled == False):
    with strategy.scope():
      # create optimizer
      optimizer = tf.keras.optimizers.Adam()

      # Specify the training configuration (optimizer, loss, metrics)
      fn_MPR_Model.compile(
        # Loss function to minimize
        loss=weighted_loss(
          mask_weight = DNN_PARAMS['mask_weight'], 
          unmask_weight = DNN_PARAMS['unmask_weight']), #keras.losses.SparseCategoricalCrossentropy(),
        # optimizer
        optimizer=optimizer,
        # List of metrics to monitor
        metrics=[
          calc_unweighted_MSE,
          calc_unmasked_MSE,
          calc_masked_MSE
          ]
        )

  ### ------------------------------------------------------ ###
  ### evaluation
  ### ------------------------------------------------------ ###
  # calling evaluate on the model and passing in the dataset created at the beginning of the tutorial. 
  # This step is the same whether you are distributing or not.
  print('# Fit fn_MPR_Model on training data')
  
  eval_losses_and_metrics = fn_MPR_Model.evaluate(
    x=combined_transformed_dataset_v3_test, #train_inputs, #combined_transformed_dataset_v3_train, #.make_initializable_iterator()
    #y=train_label,
    #batch_size=DNN_PARAMS['global_batch_size'],
    steps = total_test_examples // DNN_PARAMS['global_batch_size'], #283500 // DNN_PARAMS['global_batch_size'],
    verbose = 1 #1: progress bar; 2: one line per epoch
  )
  #the loss values and metric values
  print("eval_losses_and_metrics", eval_losses_and_metrics)

  # The returned "history" object holds a record
  # of the loss values and metric values during training
  #print('\nhistory dict:', history1.history)  


  

if __name__ == "__main__":
  args = get_args()
  train(args)
