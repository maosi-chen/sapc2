# This file handles the training of the model created by model.py,
# and is the primary interface between GCP and our custom code
### ------------------------------------------------------ ###
# imports 1

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os, sys
import posixpath
from datetime import datetime
import glob
import logging
import hypertune
import json

### ------------------------------------------------------ ###
# get env vars
TF_CONFIG_var = json.loads(os.environ.get('TF_CONFIG', '{}'))
if bool(TF_CONFIG_var) == False:
  print("bool(TF_CONFIG_var): False")
  exist_master_task = True
else:
  print("TF_CONFIG_var", TF_CONFIG_var)
  cluster_tasks = TF_CONFIG_var['cluster'].keys()
  cluster_tasks = [iTsk.lower() for iTsk in cluster_tasks]
  exist_master_task = 'master' in cluster_tasks

#print("os.environ['TF_CONFIG']", os.environ['TF_CONFIG'])

print("\nos.environ\n")
for iosKey in os.environ.keys():
  print(iosKey, ", |, ", os.environ[iosKey])

### ------------------------------------------------------ ###
# imports 2
import tensorflow as tf

### ------------------------------------------------------ ###
# create tpu strategy if machine has tpu available
tpu_name_var = os.environ.get('tpu_name', '')
if (len(tpu_name_var) > 0 and tpu_name_var != 'local'):
  cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name_var)
  print("tpu_name_var", tpu_name_var)
  print("cluster_resolver", cluster_resolver)
  print('Running on TPU ', cluster_resolver.cluster_spec().as_dict()['worker'])

  tf.config.experimental_connect_to_cluster(cluster_resolver)
  tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
  tpu_strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)

  print("tpu_strategy", tpu_strategy)


### ------------------------------------------------------ ###
tf.config.optimizer.set_jit(True)

### ------------------------------------------------------ ###
# imports 3

from .utils.distribution_utils import get_distribution_strategy
from tensorflow.python.lib.io import file_io
from .parameters.config import (DNN_PARAMS, set_dnn_params, reset_random_seeds)
from .model import create_MPR_Model_fnAPI
from .layers.data_preprocess_layer_v2 import (
  create_combined_transformed_dataset_v4,  # create_combined_transformed_dataset_v3d1_gs,
  create_combined_transformed_dataset_v4_wCFilter,
  count_tfrecords_examples_gs,
  count_tfrecords_examples_gs_tf2
)
from .training_fns.losses_v2 import (
  weighted_loss_class
)
from .training_fns.metrics import (
  calc_masked_MSE_local,
  calc_unmasked_MSE_local,
  calc_unweighted_MSE_local,
  calc_orig_out_img_edge_unmasked_MSE_local,
  calc_orig_out_img_edge_masked_MSE_local, #calc_mosaic_out_img_edge_masked_MSE_local,
  calc_avg_masked_unmasked_MSE_local
)
from .training_fns.callbacks import *
from .utils.py_utility import (
  check_and_create_file_path
)
from .parameters.create_model_parameters_tf2 import create_params


### ------------------------------------------------------ ###
# parsing commandline arguments
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
      '--initial_learning_rate', '-initlr',
      default=0.00006,
      type=float,
      help='initial learning rate (in TriCyclicalLearningRateDecay) for gradient descent, default=.01')
  parser.add_argument(
      '--one_epoch_learning_rate_cycles', '-oelrcs',
      default=1,
      type=int,
      help='number of min-max-min learning rate cycles in one epoch, default=1')
  parser.add_argument(
      '--regularization_factor', '-regfac',
      default=1e-5,
      type=float,
      help='regularization factor for variables')
  parser.add_argument(
    '--leakyrelu_alpha', '-lralpha',
    default=0.3,
    type=float,
    help='Parameter alpha for LeakyReLU activation function, default=0.3')
  parser.add_argument(
    '--se_reduction_ratio', '-serr',
    default=16,
    type=int,
    help='reduction ratio of channel squeeze and excitation layer, integer, default=16')
  parser.add_argument(
    '--se_kernel_initializer_ec', '-sekiec',
    default='he_normal',
    type=str,
    help='initializer name of the encoder FC in the Channel SE layer')
  parser.add_argument(
    '--se_kernel_initializer_dc', '-sekidc',
    default='glorot_uniform',
    type=str,
    help='initializer name of the decoder FC in the Channel SE layer')
  parser.add_argument(
    '--se_kernel_initializer', '-seki',
    default='glorot_uniform',
    type=str,
    help='initializer name of the FC in the Spatial SE layer')
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
    '--sw_hypertuning', '-swht',
    type=lambda s: s.lower() in ['true', 't,', 'y', '1'],
    default=False,
    help="This switch disables saving when running the model for hyper-parameter tuning." )
  parser.add_argument(
    '--ec_fst_kn_size', '-ecfks',
    type=int,
    default=7,
    help='encoder first layer kernel size (1D int)')
  parser.add_argument(
    '--in_img_width', '-iiw',
    type=int,
    default=64,
    help='input image width or height (1D int)')
  parser.add_argument(
    '--init_pmerge_filters', '-ipmf',
    type=int,
    default=8,
    help='initial pmerge output # of filters.')
  parser.add_argument(
    '--mask_weight', '-mskw',
    type=float,
    default=6.0,
    help='weight for the masked MSE')
  parser.add_argument(
    '--edge_weight', '-edgewt',
    type=float,
    default=0.1,
    help='weight for the sobel edges')
  parser.add_argument(
    '--compress_ratio', '-cr',
    type=float,
    default=1.0,
    help='In three way merge, the ratio at which the time difference features \
          will be compressed (to be concatenated with spatial features)')
  parser.add_argument(
    '--umloss_weight', '-umlsw',
    type=float,
    default=0.2,
    help='In decoder, the weight for the unmasked MSE between bypass and PSConv images',
  )
  parser.add_argument(
    '--cc_threshold', '-ccthresh',
    type=float,
    default=0.6,
    help='The threshold of correlation coefficient between tgt and ref images, above which the samples are used'
  )
  parser.add_argument(
    '--custom_dtype', '-cdtype',
    type=str,
    default="float32",
    help='dtype: \'float32\' (default) or \'mixed_float16\'.')

  ## ----- distributed training ----- ##
  parser.add_argument(
    '--distribution_strategy', '-diststr',
    default='mirrored',
    type=str,
    help='string specifying which distribution strategy to use. \'off\', \'one_device\', \
          \'mirrored\', \'parameter_server\', \'multi_worker_mirrored\', \'tpu\'')
  parser.add_argument(
    '--num_gpus', '-numgpu',
    default=0,
    type=int,
    help='Number of GPUs to run this model. For mirrored, if default(0), only cpu is used.')
  parser.add_argument(
    '--all_reduce_alg', '-allredalg',
    default=None,
    type=str,
    help='which algorithm to use when performing\
          all-reduce. For `MirroredStrategy`, valid values are "nccl" and\
          "hierarchical_copy". For `MultiWorkerMirroredStrategy`, valid values are\
          "ring" and "nccl".  If None, DistributionStrategy will choose based on\
          device topology.')
  parser.add_argument(
    '--num_packs', '-numpacks',
    default=1,
    type=int,
    help='Sets the `num_packs` in `tf.distribute.NcclAllReduce` \
          or `tf.distribute.HierarchicalCopyAllReduce` for `MirroredStrategy`')
  parser.add_argument(
    '--tpu_address', '-tpuaddress',
    default=None,
    type=str,
    help='String that represents TPU to connect to. Must not \
          be None if `distribution_strategy` is set to `tpu`')
  parser.add_argument(
    '--worker_hosts', '-workerh',
    default=None,
    type=json.loads,
    help='comma-separated list of worker ip:port pairs.')
  parser.add_argument(
    '--task_index', '-taskid',
    default=-1,
    type=int,
    help='current worker\'s task index')
  parser.add_argument(
    '--enable_get_next_as_optional', '-engetnextasopt',
    type=lambda s: s.lower() in ['true', 't,', 'y', '1'],
    default=False,
    help='whether enabling \
          get_next_as_optional behavior in DistributedIterator. If true, last \
          partial batch can be supported.')

  ## ----- tpu hypertuning ----- ##
  parser.add_argument(
    '--tpu_hypertuning_parameter_dict', '-tpuhppmd',
    type=json.loads,
    default=None,
    help='dictionary of hyperparameters for tpu hypertuning (when distribution_strategy is "tpu")')
  
  ## ----- distributed training ----- ##
  parser.add_argument(
    '--sw_tensorboard_callback', '-swtbcb',
    type=lambda s: s.lower() in ['true', 't,', 'y', '1'],
    default=False,
    help='[default: %(default)s] whether to turn on tensorboard callback (for profiling)')
  
  parser.add_argument(
    '--verbosity', '-vb',
    choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
    default='INFO')
  args, _ = parser.parse_known_args()
  return args

### ------------------------------------------------------ ###
#Train the model using our custom training loop
# args are supplied by the function above, this could be used for #epochs
def train(args):

  args = vars(args)
  #Train parameters, defined like in colab.

  ### ------------------------------------------------------ ###
  reset_random_seeds()

  ### ------------------------------------------------------ ###
  ### distributed training environment
  ### ------------------------------------------------------ ###
  print("args['distribution_strategy']", args['distribution_strategy'])
  print("args['num_gpus']", args['num_gpus'])
  print("args['all_reduce_alg']", args['all_reduce_alg'])
  if (args['distribution_strategy'].lower() == 'tpu'):
    print("Strategy is tpu")
    strategy = tpu_strategy
  else:
    strategy = get_distribution_strategy(
      distribution_strategy=args['distribution_strategy'],
      num_gpus=args['num_gpus'],
      all_reduce_alg=args['all_reduce_alg'],
      num_packs=args['num_packs'],
      tpu_address=args['tpu_address'])
  print("strategy", strategy)

  ### ------------------------------------------------------ ###
  # paths
  transformed_data_RootPath = args['data_dir']
  logdir = args['job_dir']

  # prepare log file for hypertuning on tpu
  if (args['distribution_strategy'].lower() == 'tpu' and args['sw_hypertuning']):
    #
    print('args[\'tpu_hypertuning_parameter_dict\']', args['tpu_hypertuning_parameter_dict'])
    # check whether the values in args['tpu_hypertuning_parameter_dict'] match the actual parameters
    for i_hp_name, i_hp_val in args['tpu_hypertuning_parameter_dict'].items():
      print(i_hp_name, i_hp_val)
      i_hp_name_lc = i_hp_name.lower()
      print("args[{}]".format(i_hp_name_lc), args[i_hp_name_lc])
      if (args[i_hp_name_lc] != i_hp_val):
        print("\n@@@@@@@@\nWarning: args[{}] ({}) doesn't match args['tpu_hypertuning_parameter_dict'][{}] ({})".format(
          i_hp_name_lc, args[i_hp_name_lc],
          i_hp_name, i_hp_val
        ))
        print('Overwrite with args[\'tpu_hypertuning_parameter_dict\'] values')
        args[i_hp_name_lc] = i_hp_val
    # setup the tpu_hp_log file
    tpu_hp_names = args['tpu_hypertuning_parameter_dict'].keys()
    tpu_hp_cmb_names_str = '_'.join(tpu_hp_names)
    tpu_hp_log_FWN = os.path.join(logdir, 'tpu_hp_log_' + tpu_hp_cmb_names_str + '.txt')
    # write the header (for the first time)
    tpu_hp_header_str = 'tpu hyperparameter tuning of ' + ' '.join(tpu_hp_names)
    print("tpu_hp_header_str", tpu_hp_header_str)
    if (not tf.io.gfile.exists(tpu_hp_log_FWN)):
      with tf.io.gfile.GFile(tpu_hp_log_FWN, "w") as f:
        f.write(tpu_hp_header_str + "\n")
    print("after writing tpu_hp_header_str")
  
  # training parameters
  ec_fst_kernel_shape = tuple((args['ec_fst_kn_size'], args['ec_fst_kn_size']))
  set_dnn_params({
    # dataset
    'global_batch_size': args['batch_size'], #128,  # 'batch_size': 20, #100,
    'buffer_size': 100,  # None,
    'shuffle_seed': 42,
    'transformed_file_dir_train': os.path.join(transformed_data_RootPath, 'train'),
    'transformed_file_dir_test': os.path.join(transformed_data_RootPath, 'test'),
    'num_parallel_reads': tf.data.experimental.AUTOTUNE, #8,
    'num_parallel_calls': tf.data.experimental.AUTOTUNE, #8,
    'in_img_width': args['in_img_width'], #64
    # training loop
    'initial_epoch': args['initial_epoch'],
    'num_epochs': args['initial_epoch'] + args['num_epochs'],  # 5,
    'train_screen_print_freq': 100,
    'train_checkpoint_freq': 1000,
    'sw_resume_training': args['sw_resume_training'],
    'resume_model_source': args['resume_model_source'],
    # log directory
    'logdir': logdir,
    # model
    'parameter_dict': create_params(
      (None, 64, 64, 3), 64, args['init_pmerge_filters'],
      "channels_last", ec_fst_kernel_shape, (3, 3), True,
      'prelu', 'prelu', 'prelu', #'relu', 'prelu', 'leaky_relu',
      encoder_activation_kwarg_dict={'init_alpha': args['leakyrelu_alpha']}, #None,
      encoder_gpmerge_activation_kwarg_dict={'init_alpha': args['leakyrelu_alpha']},
      decoder_activation_kwarg_dict={'init_alpha': args['leakyrelu_alpha']}, #{'alpha': args['leakyrelu_alpha']},
      last_ec_layer_act=True,
      first_layer_bn=True, last_ec_layer_bn=True, min_shape=3, regularization_factor=args['regularization_factor'],
      compress_ratio=args['compress_ratio'],
      umloss_weight=args['umloss_weight'],
      SE_reduction_ratio=args['se_reduction_ratio'],
      SE_kernel_initializer_ec=args['se_kernel_initializer_ec'],
      SE_kernel_initializer_dc=args['se_kernel_initializer_dc'],
      SE_kernel_initializer=args['se_kernel_initializer']
    ),
    'flt_fs': 64,
    'cc_threshold': args['cc_threshold'],
    'data_format': 'channels_last',
    'custom_dtype': tf.keras.mixed_precision.experimental.Policy("float32") if args['custom_dtype']=="float32" \
                    else tf.keras.mixed_precision.experimental.Policy("mixed_float16", loss_scale="dynamic"), #'float32', #tf.float32,
    'debug_print': False,
    # optimizer
    'learning_rate': args['learning_rate'], #5e-4,
    'initial_learning_rate': args['initial_learning_rate'],
    # loss
    'mask_weight': args['mask_weight'], # 6.0,
    'unmask_weight': 1.0,
    'edge_weight': args['edge_weight'], # 0.1,
    # distributed training
    'distribution_strategy': args['distribution_strategy'],
    'num_gpus': args['num_gpus'],
    'all_reduce_alg': args['all_reduce_alg'],
    'num_packs': args['num_packs'],
    'tpu_address': args['tpu_address'],
    'worker_hosts': args['worker_hosts'],
    'task_index': args['task_index'],
    'enable_get_next_as_optional': args['enable_get_next_as_optional'],
    # tensorboard profiling
    'sw_tensorboard_callback': args['sw_tensorboard_callback'],
    # ckpt & model save/restore
    'max_to_keep': 10,
    'epoch_ckpt_incr': 10000,
    'CHECKPOINT_FILE_NAME': 'weights.{epoch:02d}-{val_loss:.6f}.hdf5',
    'SAVED_MODEL_FILE_NAME': 'mpr.hdf5',
    'SAVED_MODEL_CONFIG': 'model_config.json',
    'SAVED_MODEL_WEIGHTS': 'model_weights.h5',
    'tf_SavedModel_FILE_PATH': os.path.join(logdir, 'tf_SavedModel')
  })


  ### ------------------------------------------------------ ###
  ### dataset counts
  ### ------------------------------------------------------ ###
  # total_train_examples = count_tfrecords_examples_gs(
  #   posixpath.join(DNN_PARAMS['transformed_file_dir_train'], "*.gz"), compression_type="GZIP", num_parallel_reads=16)
  total_train_examples = 2351906  # 2351906 for cc_threshold=0.8 # 3539818 for cc_threshold=0.6 #5386500 for cc_threshold=0.0
  if ("small" in transformed_data_RootPath):
    total_train_examples = int(total_train_examples / 10)
  elif ("medium" in transformed_data_RootPath):
    total_train_examples = int(total_train_examples / 10)
  print("total_train_examples", total_train_examples)
  # total_test_examples  = count_tfrecords_examples_gs(
  #   posixpath.join(DNN_PARAMS['transformed_file_dir_test'], "*.gz"),  compression_type="GZIP", num_parallel_reads=16)
  total_test_examples = 123614  # 123614 for cc_threshold=0.8  # 186151 for cc_threshold=0.6 #283500 for cc_threshold=0.0
  if ("small" in transformed_data_RootPath):
    total_test_examples = int(total_test_examples / 5.5)
  elif ("medium" in transformed_data_RootPath):
    total_test_examples = int(total_test_examples / 5.1)
  print("total_test_examples", total_test_examples)
  
  ### ------------------------------------------------------ ###
  # parameters for learning rate scheduler
  # /2 means one cycle per epoch, stepsize is the steps of a half cycle
  lr_schedule_stepsize = float((total_train_examples // DNN_PARAMS['global_batch_size']) \
                               / 2 / args['one_epoch_learning_rate_cycles'])
  lr_schedule_steps_per_epoch = int(total_train_examples // DNN_PARAMS['global_batch_size'])
  print('lr_schedule_stepsize', lr_schedule_stepsize)
  print('lr_schedule_steps_per_epoch', lr_schedule_steps_per_epoch)
  
  ### ------------------------------------------------------ ###
  ### Datasets
  ### ------------------------------------------------------ ###
  #combined_transformed_dataset_v3_train = create_combined_transformed_dataset_v4(
  combined_transformed_dataset_v3_train = create_combined_transformed_dataset_v4_wCFilter(
    os.path.join(DNN_PARAMS['transformed_file_dir_train'], "*.gz"),
    DNN_PARAMS['global_batch_size'],
    DNN_PARAMS['buffer_size'],
    shuffle_seed=DNN_PARAMS['shuffle_seed'],
    num_epochs=None, #1,
    num_parallel_reads=DNN_PARAMS['num_parallel_reads'],
    num_parallel_calls=DNN_PARAMS['num_parallel_calls'],
    flt_fs=DNN_PARAMS['flt_fs'],  # 64
    cc_threshold=DNN_PARAMS['cc_threshold']
  )
  #combined_transformed_dataset_v3_test = create_combined_transformed_dataset_v4(
  combined_transformed_dataset_v3_test = create_combined_transformed_dataset_v4_wCFilter(
    os.path.join(DNN_PARAMS['transformed_file_dir_test'], "*.gz"),
    DNN_PARAMS['global_batch_size'],
    None,
    shuffle_seed=DNN_PARAMS['shuffle_seed'],
    num_epochs=None, #1,
    num_parallel_reads=DNN_PARAMS['num_parallel_reads'],
    num_parallel_calls=DNN_PARAMS['num_parallel_calls'],
    flt_fs=DNN_PARAMS['flt_fs'],
    cc_threshold=DNN_PARAMS['cc_threshold']
  )

  ### ------------------------------------------------------ ###
  ### Loss/Metrics functions
  ### ------------------------------------------------------ ###
  if not args['sw_hypertuning']:
    metrics_list = [
      calc_masked_MSE_local,
      calc_unmasked_MSE_local,
      calc_unweighted_MSE_local,
      calc_orig_out_img_edge_unmasked_MSE_local,
      calc_orig_out_img_edge_masked_MSE_local #calc_mosaic_out_img_edge_masked_MSE_local
      ]
  else:
    metrics_list = [
      calc_avg_masked_unmasked_MSE_local
    ]

  ### ------------------------------------------------------ ###
  ### create model
  ### ------------------------------------------------------ ###
  has_compiled = False
  if (DNN_PARAMS['sw_resume_training'] == True):
    if (DNN_PARAMS['resume_model_source'] == 'hdf'):
      print( 'restore previous exported model weights (h5) from {}'.format(
        os.path.join(DNN_PARAMS['logdir'], DNN_PARAMS['SAVED_MODEL_WEIGHTS'])) )
      if DNN_PARAMS['logdir'].startswith('gs://'):
        copy_file_from_gcs(DNN_PARAMS['logdir'], DNN_PARAMS['SAVED_MODEL_WEIGHTS'])
        print(DNN_PARAMS['custom_dtype'])
        #print("DNN_PARAMS['custom_dtype'].name=='mixed_float16'", DNN_PARAMS['custom_dtype'].name == 'mixed_float16')

        with strategy.scope():
          tf.keras.mixed_precision.experimental.set_policy(DNN_PARAMS['custom_dtype'])
          # create model
          fn_MPR_Model = create_MPR_Model_fnAPI(
            DNN_PARAMS['flt_fs'], #used_img_size
            DNN_PARAMS['custom_dtype'],
            DNN_PARAMS['data_format'],
            DNN_PARAMS['debug_print'],
            DNN_PARAMS['parameter_dict']
          )
          # create optimizer
          optimizer = tf.keras.optimizers.Adam()
          # Specify the training configuration (optimizer, loss, metrics)
          fn_MPR_Model.compile(
            # Loss function to minimize
            loss=weighted_loss_class(
              mask_weight = DNN_PARAMS['mask_weight'], 
              unmask_weight = DNN_PARAMS['unmask_weight'],
              edge_weight = DNN_PARAMS['edge_weight'],
              reduction=tf.keras.losses.Reduction.SUM
            ),
            # optimizer
            optimizer=optimizer,
            # List of metrics to monitor
            metrics=metrics_list
            )

          #load the weights after compiliation  
          fn_MPR_Model.load_weights( DNN_PARAMS['SAVED_MODEL_WEIGHTS'] )
          print('model weights loaded sucessfully (h5). {}'.format(DNN_PARAMS['SAVED_MODEL_WEIGHTS']))

        has_compiled = True

      else:
        ## local resuming training (hdf format)
        ## TODO: implement the tf format loading
        print('Unsupported local resume training. {}'.format(DNN_PARAMS['SAVED_MODEL_FILE_NAME']))
        sys.exit('Unsupported local resume training. {}'.format(DNN_PARAMS['SAVED_MODEL_FILE_NAME']))

    elif (DNN_PARAMS['resume_model_source'] == 'tf'):
      # resume from saved model with tf format
      ## TODO: test whether tf.keras.models.load_model needs the same h5py hack (copy from gs:// storage before loading)
      ## TODO: implement the tf format loading
      #print('Unsupported \'resume_model_source\'. loading failed. {}'.format(DNN_PARAMS['tf_SavedModel_FILE_PATH']))
      sys.exit('Unsupported \'resume_model_source\'. loading failed. {}'.format(DNN_PARAMS['tf_SavedModel_FILE_PATH']))

  else:
    # new training (functional API model)
    print('create a new MPR model (fnAPI)')
    #strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
    #with strategy_scope:
      tf.keras.mixed_precision.experimental.set_policy(DNN_PARAMS['custom_dtype'])
      fn_MPR_Model = create_MPR_Model_fnAPI(
        DNN_PARAMS['flt_fs'], #used_img_size
        DNN_PARAMS['custom_dtype'],
        DNN_PARAMS['data_format'],
        DNN_PARAMS['debug_print'],
        DNN_PARAMS['parameter_dict']
      )

  if (has_compiled == False):
    print(DNN_PARAMS['custom_dtype'])
    print("DNN_PARAMS['custom_dtype'].name=='mixed_float16'", DNN_PARAMS['custom_dtype'].name=='mixed_float16')
    with strategy.scope():
    #with strategy_scope:
      # create optimizer
      optimizer = tf.keras.optimizers.Adam()
      # Specify the training configuration (optimizer, loss, metrics)
      fn_MPR_Model.compile(
        # Loss function to minimize
        loss=weighted_loss_class( #weighted_loss(
          mask_weight = DNN_PARAMS['mask_weight'], 
          unmask_weight = DNN_PARAMS['unmask_weight'],
          edge_weight = DNN_PARAMS['edge_weight'],
          reduction=tf.keras.losses.Reduction.SUM
        ),
        # optimizer
        optimizer=optimizer,
        # List of metrics to monitor
        metrics=metrics_list
        )
    print('model compiled.')

  ### ------------------------------------------------------ ###
  # Define callbacks (model independent)
  ### ------------------------------------------------------ ###

  ## Terminate on NaN
  tonan_callback = tf.keras.callbacks.TerminateOnNaN()
  
  ## Model Checkpoint: This callback saves the model after every epoch.
  # Unhappy hack to workaround h5py not being able to write to GCS.
  # Force snapshots and saves to local filesystem, then copy them over to GCS.
  checkpoint_path = DNN_PARAMS['CHECKPOINT_FILE_NAME']
  if not DNN_PARAMS['logdir'].startswith('gs://'):
    #os.path.join(DNN_PARAMS['logdir'], 'weights.{epoch:02d}-{val_loss:.6f}.hdf5'),
    check_and_create_file_path(DNN_PARAMS['logdir'])
    checkpoint_path = os.path.join(DNN_PARAMS['logdir'], DNN_PARAMS['CHECKPOINT_FILE_NAME'])
  ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, 
    save_best_only=False, #True,
    save_weights_only=True #False 
    )
  ckpt_expGS_callback = ExportCheckpointGS(DNN_PARAMS['logdir'])

  ## Learning Rate Scheduler
  lr_decay_callback = TriCyclicalLearningRateDecayCallback(
    initial_learning_rate=DNN_PARAMS['initial_learning_rate'],
    maximal_learning_rate=DNN_PARAMS['learning_rate'],
    step_size=lr_schedule_stepsize,
    steps_per_epoch=lr_schedule_steps_per_epoch,
    scale_mode='cycle',
    name='TriangularCyclicalLearningRate'
  )

  ### ------------------------------------------------------ ###
  # Define callbacks (model dependent)
  ### ------------------------------------------------------ ###
  #with strategy.scope():
  PrintLR_callback = PrintLR()
  # Callback for printing the sum of internal Model losses
  PrintModelLoss_callback = PrintModelLoss()

  #
  if args['sw_hypertuning']:
    callbacks = [
      tonan_callback,
      lr_decay_callback
    ]
  else:
    callbacks = [
      tonan_callback,
      ckpt_callback,
      ckpt_expGS_callback,
      lr_decay_callback,
      PrintLR_callback,
      PrintModelLoss_callback, #PrintRegLoss_callback
    ]

  ### ------------------------------------------------------ ###
  ### training
  ### ------------------------------------------------------ ###
  # train the model in the usual way, 
  # calling fit on the model and passing in the dataset created at the beginning of the tutorial. 
  # This step is the same whether you are distributing the training or not.
  print('# Fit fn_MPR_Model on training data')
  
  history = fn_MPR_Model.fit(
    x=combined_transformed_dataset_v3_train,
    initial_epoch=DNN_PARAMS['initial_epoch'],
    epochs=DNN_PARAMS['num_epochs'], #20,
    steps_per_epoch=total_train_examples // DNN_PARAMS['global_batch_size'],
    # We pass some validation for monitoring validation loss and metrics at the end of each epoch
    validation_data=combined_transformed_dataset_v3_test,
    validation_steps=total_test_examples // DNN_PARAMS['global_batch_size'],
    callbacks=callbacks,
    verbose = 2 # one line per epoch
    )

  # The returned "history" object holds a record
  # of the loss values and metric values during training
  print('\nhistory dict:', history.history)  
  if not args['sw_hypertuning']:
    print('\nreg_loss:', PrintModelLoss_callback.reg_loss)


  # Save JSON config to disk
  if not args['sw_hypertuning']:
    json_config = fn_MPR_Model.to_json()
    if DNN_PARAMS['logdir'].startswith('gs://'):
      with open(DNN_PARAMS['SAVED_MODEL_CONFIG'], 'w') as json_file:
        json_file.write(json_config)
      # Save weights to disk
      fn_MPR_Model.save_weights(DNN_PARAMS['SAVED_MODEL_WEIGHTS'])

      copy_file_to_gcs(DNN_PARAMS['logdir'], DNN_PARAMS['SAVED_MODEL_CONFIG'])
      copy_file_to_gcs(DNN_PARAMS['logdir'], DNN_PARAMS['SAVED_MODEL_WEIGHTS'])
    else:
      with open(os.path.join(DNN_PARAMS['logdir'], DNN_PARAMS['SAVED_MODEL_CONFIG']), 'w') as json_file:
        json_file.write(json_config)
      # Save weights to disk
      fn_MPR_Model.save_weights(os.path.join(DNN_PARAMS['logdir'], DNN_PARAMS['SAVED_MODEL_WEIGHTS']))
    
  if args['sw_hypertuning']:
    # get the val_loss metric
    hist_dict = history.history
    if (args['distribution_strategy'].lower() == 'tpu'):
      # Append write the result to the log file
      tpu_hp_cmb_name_value_pairs_str = ''
      for i_hp_name, i_hp_val in args['tpu_hypertuning_parameter_dict'].items():
        tpu_hp_cmb_name_value_pairs_str += '{}, {:.10e}, |, '.format(i_hp_name, i_hp_val)
      avg_m_um_MSE_epochs_cmb_str = ''
      for iMVal in hist_dict['val_calc_avg_masked_unmasked_MSE_local']:
        avg_m_um_MSE_epochs_cmb_str += '{:.10e}, '.format(iMVal)
      with tf.io.gfile.GFile(tpu_hp_log_FWN, "a") as f:
        f.write(tpu_hp_cmb_name_value_pairs_str + avg_m_um_MSE_epochs_cmb_str + "\n")
      
    #val_loss_hpt = hist_dict['val_loss'][-1]
    #return val_loss_hpt
    avg_m_um_MSE_hpt = hist_dict['val_calc_avg_masked_unmasked_MSE_local'][-1]
    return avg_m_um_MSE_hpt


if __name__ == "__main__":
  args = get_args()
  
  sw_hypertuning = args.sw_hypertuning

  if not sw_hypertuning:
    train(args)
  else:
    try:
      val_loss_hpt = train(args)
    except:
      val_loss_hpt = 9998.0
    args = vars(args)
    if (args['distribution_strategy'].lower() == 'tpu'):
      print("Trial result of the following hyperparameter(s)")
      for i_hp_name, i_hp_val in args['tpu_hypertuning_parameter_dict'].items():
        print('"{}": {}'.format(i_hp_name, i_hp_val))
      print('  Resulting val_calc_avg_masked_unmasked_MSE_local: {}'.format(val_loss_hpt))
    else:
      print('{} Resulting val_calc_avg_masked_unmasked_MSE_local: {}'.format(args.__dict__, val_loss_hpt))
      # write out the metric so that the executable can be invoked again with next set of metrics
      hpt = hypertune.HyperTune()
      hpt.report_hyperparameter_tuning_metric(
         hyperparameter_metric_tag='val_calc_avg_masked_unmasked_MSE_local',
         metric_value=val_loss_hpt
         )
