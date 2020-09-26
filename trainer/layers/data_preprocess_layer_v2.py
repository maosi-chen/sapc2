from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

import tensorflow.keras as K

from functools import partial 


def _parse_SeqExampleProto_zip_crop_flip_v3(flt_fs, in_example_proto):
    
  # Define how to parse the example
  features_dict = {
    'ARD_LST_ref': tf.io.FixedLenFeature(shape=[flt_fs*flt_fs], dtype=tf.float32), #_float_feature(ARD_LST_ref[iiBatch,:,:].reshape((-1))),
    'ARD_LST_tgt_masked': tf.io.FixedLenFeature(shape=[flt_fs*flt_fs], dtype=tf.float32), #_float_feature(ARD_LST_tgt_masked[iiBatch,:,:].reshape((-1))),
    'Mask_Img': tf.io.FixedLenFeature(shape=[flt_fs*flt_fs], dtype=tf.float32),
    'ref_tgt_DOY_difference': tf.io.FixedLenFeature(shape = [], dtype=tf.float32), #_float_feature(ref_tgt_DOY_difference[iiBatch]),
    'ref_DOY': tf.io.FixedLenFeature(shape = [], dtype=tf.float32), #_float_feature(ref_DOY[iiBatch]),
    'tgt_DOY': tf.io.FixedLenFeature(shape = [], dtype=tf.float32), #_float_feature(tgt_DOY[iiBatch]),
    'ref_tgt_DOY_difference_reverse': tf.io.FixedLenFeature(shape = [], dtype=tf.float32), #_float_feature(ref_tgt_DOY_difference_reverse[iiBatch]),
    #'mean_feature': tf.io.FixedLenFeature(shape = [], dtype=tf.float32), #_float_feature(mean_featureNP[iiBatch]),

    'ARD_LST_tgt': tf.io.FixedLenFeature(shape=[flt_fs*flt_fs], dtype=tf.float32) #_float_feature(ARD_LST_tgt[iiBatch,:,:].reshape((-1)))    
  }

  # parse
  features = tf.io.parse_single_example(
    in_example_proto, 
    features_dict)

  # distribute the fetched (context) features into tensors
  
  # inputs
  ARD_LST_ref1D = features["ARD_LST_ref"]
  ARD_LST_ref = tf.reshape(ARD_LST_ref1D, [flt_fs, flt_fs])
  
  ARD_LST_tgt_masked1D = features["ARD_LST_tgt_masked"]
  ARD_LST_tgt_masked = tf.reshape(ARD_LST_tgt_masked1D, [flt_fs, flt_fs])
  
  Mask_Img1D = features["Mask_Img"]
  Mask_Img = tf.reshape(Mask_Img1D, [flt_fs, flt_fs])
  Mask_Img_cpy = tf.identity(Mask_Img)
  
  DOYDiff_TmR = features['ref_tgt_DOY_difference']
  ref_DOY = features["ref_DOY"]
  tgt_DOY = features["tgt_DOY"]
  DOYDiff_RmT = features["ref_tgt_DOY_difference_reverse"]  

  #mean_feature = features['mean_feature']

  # label
  ARD_LST_tgt1D = features["ARD_LST_tgt"]
  ARD_LST_tgt = tf.reshape(ARD_LST_tgt1D, [flt_fs, flt_fs])
  ## Mask_Img will also be included in the label as the 2nd arg. for the loss function
  
  return (
      ARD_LST_ref, 
      ref_DOY, 
      DOYDiff_TmR, 
      ARD_LST_tgt_masked, 
      tgt_DOY, 
      DOYDiff_RmT,
      Mask_Img
    ), \
    tf.stack(
      [ARD_LST_tgt, Mask_Img_cpy], axis=-1
    )

def read_batch_zip_crop_flip_records_train_v3(
  filenames, 
  compression_type = "GZIP", #""
  batch_size = 20, 
  num_epochs = None, 
  buffer_size = 50000, 
  shuffle_seed = None,
  padding_values = (
    (
      tf.constant(-9998.0, dtype=tf.float32),
      tf.constant(-9998.0, dtype=tf.float32),
      tf.constant(-9998.0, dtype=tf.float32),
      tf.constant(-9998.0, dtype=tf.float32),
      tf.constant(-9998.0, dtype=tf.float32),
      tf.constant(-9998.0, dtype=tf.float32),
      tf.constant(-9998.0, dtype=tf.float32)
    ),
    #(
      tf.constant(-9998.0, dtype=tf.float32)#,
      #tf.constant(-9998.0, dtype=tf.float32)
    #)
  ),
  ## new Dataset v1.8 features
  num_parallel_reads = 1,
  num_parallel_calls = 4, # suggestion: use the number of available CPU cores
  prefetch_buffer_size = 1,
  ## new Dataset v1.13 features
  num_workers = 1,
  worker_index = 0,
  drop_remainder = False,
  #
  # application specific parameters
  parser_method=_parse_SeqExampleProto_zip_crop_flip_v3,
  flt_fs = 64
  ):

  # Make partial method for map call below. Must have only one input
  parser_method_single_input = partial(parser_method, flt_fs)

  # inputs
  #    'ARD_LST_ref':        tf.TensorShape([flt_fs, flt_fs]),
  #    'ref_DOY':            tf.TensorShape([]),
  #    'DOYDiff_TmR':        tf.TensorShape([]),
  #    'ARD_LST_tgt_masked': tf.TensorShape([flt_fs, flt_fs]),
  #    'tgt_DOY':            tf.TensorShape([]),
  #    'DOYDiff_RmT':        tf.TensorShape([]),
  #    'Mask_Img':           tf.TensorShape([flt_fs, flt_fs])
  # label
  #    'ARD_LST_tgt, Mask_Img': tf.TensorShape([flt_fs, flt_fs, 2])

  padded_shapes = (
    (
      tf.TensorShape([flt_fs, flt_fs]),
      tf.TensorShape([]),
      tf.TensorShape([]),
      tf.TensorShape([flt_fs, flt_fs]),
      tf.TensorShape([]),
      tf.TensorShape([]),
      tf.TensorShape([flt_fs, flt_fs])
    ),
    tf.TensorShape([flt_fs, flt_fs, 2]) #,
  )

  dataset = tf.data.TFRecordDataset(
    filenames, 
    compression_type = compression_type, 
    num_parallel_reads = num_parallel_reads)
  if (buffer_size is not None):
    #dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(
    # buffer_size=buffer_size, count=num_epochs, seed=shuffle_seed))
    #dataset.apply(tf.data.experimental.filter_for_shard(num_workers, worker_index))
    dataset = dataset.shard(num_workers, worker_index)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.shuffle(buffer_size, seed=shuffle_seed, reshuffle_each_iteration=True)
  else:
    dataset = dataset.repeat(num_epochs)

  dataset = dataset.map(parser_method_single_input, num_parallel_calls=num_parallel_calls)
  #dataset = dataset.cache() # cache to memory
  #dataset = dataset.padded_batch(
  #  batch_size, 
  #  padded_shapes, #padded_shapes
  #  padding_values=padding_values,
  #  drop_remainder=drop_remainder
  #)
  dataset = dataset.batch(
    batch_size, 
    drop_remainder=drop_remainder
  )
  dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

  return dataset



##---------------------------------------------------------##
## dataset read & parse functions 02/11/2020
## (1) wrapper instead of the partial
## (2) parse the entire batch instead of single example
def _parse_SeqExampleProto_zip_crop_flip_v4(flt_fs, in_example_protos):
  # Define how to parse the example
  features_dict = {
    'ARD_LST_ref': tf.io.FixedLenFeature(shape=[flt_fs * flt_fs], dtype=tf.float32),
    # _float_feature(ARD_LST_ref[iiBatch,:,:].reshape((-1))),
    'ARD_LST_tgt_masked': tf.io.FixedLenFeature(shape=[flt_fs * flt_fs], dtype=tf.float32),
    # _float_feature(ARD_LST_tgt_masked[iiBatch,:,:].reshape((-1))),
    'Mask_Img': tf.io.FixedLenFeature(shape=[flt_fs * flt_fs], dtype=tf.float32),
    'ref_tgt_DOY_difference': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
    # _float_feature(ref_tgt_DOY_difference[iiBatch]),
    'ref_DOY': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),  # _float_feature(ref_DOY[iiBatch]),
    'tgt_DOY': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),  # _float_feature(tgt_DOY[iiBatch]),
    'ref_tgt_DOY_difference_reverse': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
    # _float_feature(ref_tgt_DOY_difference_reverse[iiBatch]),
    # 'mean_feature': tf.io.FixedLenFeature(shape = [], dtype=tf.float32), #_float_feature(mean_featureNP[iiBatch]),
    
    'ARD_LST_tgt': tf.io.FixedLenFeature(shape=[flt_fs * flt_fs], dtype=tf.float32)
    # _float_feature(ARD_LST_tgt[iiBatch,:,:].reshape((-1)))
  }
  
  # parse
  # features = tf.io.parse_single_example(
  #   in_example_proto,
  #   features_dict)
  ## features: A dict mapping feature keys to Tensor, SparseTensor, and RaggedTensor values.
  features = tf.io.parse_example(
    in_example_protos,
    features_dict)
  
  # distribute the fetched (context) features into tensors
  
  # inputs
  ## shape of ARD_LST_ref1D: [B, flt_fs * flt_fs]
  ARD_LST_ref1D = features["ARD_LST_ref"]
  ARD_LST_ref = tf.reshape(ARD_LST_ref1D, [-1, flt_fs, flt_fs])
  # ARD_LST_ref = tf.reshape(ARD_LST_ref1D, [flt_fs, flt_fs])
  
  ARD_LST_tgt_masked1D = features["ARD_LST_tgt_masked"]
  ARD_LST_tgt_masked = tf.reshape(ARD_LST_tgt_masked1D, [-1, flt_fs, flt_fs])
  # ARD_LST_tgt_masked = tf.reshape(ARD_LST_tgt_masked1D, [flt_fs, flt_fs])
  
  Mask_Img1D = features["Mask_Img"]
  Mask_Img = tf.reshape(Mask_Img1D, [-1, flt_fs, flt_fs])
  # Mask_Img = tf.reshape(Mask_Img1D, [flt_fs, flt_fs])
  Mask_Img_cpy = tf.identity(Mask_Img)
  
  # shape of the scalars: [B]
  DOYDiff_TmR = features['ref_tgt_DOY_difference']
  ref_DOY = features["ref_DOY"]
  tgt_DOY = features["tgt_DOY"]
  DOYDiff_RmT = features["ref_tgt_DOY_difference_reverse"]
  
  # mean_feature = features['mean_feature']
  
  # label: [B, flt_fs * flt_fs]
  ARD_LST_tgt1D = features["ARD_LST_tgt"]
  ARD_LST_tgt = tf.reshape(ARD_LST_tgt1D, [-1, flt_fs, flt_fs])
  # ARD_LST_tgt = tf.reshape(ARD_LST_tgt1D, [flt_fs, flt_fs])
  ## Mask_Img will also be included in the label as the 2nd arg. for the loss function
  
  ## tf.stack([...], axis=-1) will generate a tensor with one-higher dimension of any tensor in [...] as the last dim.
  ## so the shape of stacked label is [B, flt_fs, flt_fs, 2]
  
  return (
           ARD_LST_ref,
           ref_DOY,
           DOYDiff_TmR,
           ARD_LST_tgt_masked,
           tgt_DOY,
           DOYDiff_RmT,
           Mask_Img
         ), \
         tf.stack(
           [ARD_LST_tgt, Mask_Img_cpy], axis=-1
         )

def parser_v4(flt_fs):
  # for v4, func == _parse_SeqExampleProto_zip_crop_flip_v4
  def wrapper(in_example_protos):
    return _parse_SeqExampleProto_zip_crop_flip_v4(flt_fs, in_example_protos)
  
  return wrapper

def read_batch_zip_crop_flip_records_train_v4(
  files_pattern, #filenames,
  compression_type="GZIP",  # ""
  batch_size=20,
  num_epochs=None,
  buffer_size=50000,
  shuffle_seed=None,
  ## new Dataset v1.8 features
  num_parallel_reads=1,
  num_parallel_calls=4,  # suggestion: use the number of available CPU cores
  prefetch_buffer_size=tf.data.experimental.AUTOTUNE, #1,
  ## new Dataset v1.13 features
  num_workers=1,
  worker_index=0,
  drop_remainder=False,
  #
  # application specific parameters
  parser_method=parser_v4(64), #_parse_SeqExampleProto_zip_crop_flip_v4,
  flt_fs=64
  ):
  
  # inputs
  #    'ARD_LST_ref':        tf.TensorShape([B, flt_fs, flt_fs]),
  #    'ref_DOY':            tf.TensorShape([B]),
  #    'DOYDiff_TmR':        tf.TensorShape([B]),
  #    'ARD_LST_tgt_masked': tf.TensorShape([B, flt_fs, flt_fs]),
  #    'tgt_DOY':            tf.TensorShape([B]),
  #    'DOYDiff_RmT':        tf.TensorShape([B]),
  #    'Mask_Img':           tf.TensorShape([B, flt_fs, flt_fs])
  # label
  #    'ARD_LST_tgt, Mask_Img': tf.TensorShape([B, flt_fs, flt_fs, 2])

  dataset = tf.data.Dataset.list_files(files_pattern, seed=shuffle_seed+100)
  if (buffer_size is not None):
    dataset = dataset.shard(num_workers, worker_index)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.shuffle(buffer_size, seed=shuffle_seed, reshuffle_each_iteration=True)
  else:
    dataset = dataset.repeat(num_epochs)

  dataset = dataset.interleave(
    lambda x: tf.data.TFRecordDataset(
      x,
      compression_type=compression_type,
      num_parallel_reads=1 #num_parallel_reads #1
    ),
    cycle_length=num_parallel_reads,
    block_length=1,
    num_parallel_calls=tf.data.experimental.AUTOTUNE, #num_parallel_calls #tf.data.experimental.AUTOTUNE
    deterministic=True
  )

  # dataset = tf.data.TFRecordDataset(
  #   filenames,
  #   compression_type=compression_type,
  #   num_parallel_reads=num_parallel_reads)
  # if (buffer_size is not None):
  #   # dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=buffer_size, count=num_epochs, seed=shuffle_seed))
  #   # dataset.apply(tf.data.experimental.filter_for_shard(num_workers, worker_index))
  #   dataset = dataset.shard(num_workers, worker_index)
  #   dataset = dataset.repeat(num_epochs)
  #   dataset = dataset.shuffle(buffer_size, seed=shuffle_seed, reshuffle_each_iteration=True)
  # else:
  #   dataset = dataset.repeat(num_epochs)

  dataset = dataset.batch(
    batch_size,
    drop_remainder=drop_remainder
  )
  #dataset = dataset.map(parser_method_single_input, num_parallel_calls=num_parallel_calls)
  #dataset = dataset.map(_parse_SeqExampleProto_zip_crop_flip_v4, num_parallel_calls=num_parallel_calls)

  #dataset = dataset.map(parser_v4(flt_fs), num_parallel_calls=num_parallel_calls)
  dataset = dataset.map(parser_method, num_parallel_calls=tf.data.experimental.AUTOTUNE) #num_parallel_calls)
  #dataset = dataset.cache()

  # dataset = dataset.cache() # cache to memory

  dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

  return dataset


def _filter_cc_fn(flt_fs, cc_threshold_ts, in_example_proto):
  ## parse a sample
  features_dict = {
    'ARD_LST_ref': tf.io.FixedLenFeature(shape=[flt_fs * flt_fs], dtype=tf.float32),
    'ARD_LST_tgt': tf.io.FixedLenFeature(shape=[flt_fs * flt_fs], dtype=tf.float32)
  }

  # parse
  features = tf.io.parse_single_example(
    in_example_proto,
    features_dict)
  # [flt_fs * flt_fs]
  ARD_LST_ref1D = features["ARD_LST_ref"]
  ARD_LST_tgt1D = features["ARD_LST_tgt"]

  ## calculate correlation coefficient
  # []
  cc = tfp.stats.correlation(ARD_LST_ref1D, ARD_LST_tgt1D, sample_axis=0, event_axis=None)

  ## determine whether the cc is above cc_threshold
  return tf.cond(
    tf.math.greater_equal(cc, cc_threshold_ts),
    lambda: True,
    lambda: False
  )

def filter_cc(flt_fs, cc_threshold_ts):
  def wrapper(in_sample):
    return _filter_cc_fn(flt_fs, cc_threshold_ts, in_sample)
  return wrapper


def read_batch_zip_crop_flip_records_train_v4_wCFilter(
  files_pattern,  # filenames,
  compression_type="GZIP",  # ""
  batch_size=20,
  num_epochs=None,
  buffer_size=50000,
  shuffle_seed=None,
  ## new Dataset v1.8 features
  num_parallel_reads=1,
  num_parallel_calls=4,  # suggestion: use the number of available CPU cores
  prefetch_buffer_size=tf.data.experimental.AUTOTUNE,  # 1,
  ## new Dataset v1.13 features
  num_workers=1,
  worker_index=0,
  drop_remainder=False,
  #
  cc_threshold = 0.0, # correlation coefficient threshold (examples with cc above which is allowed in.)
  #
  # application specific parameters
  parser_method=parser_v4(64),  # _parse_SeqExampleProto_zip_crop_flip_v4,
  flt_fs=64
  ):
  # inputs
  #    'ARD_LST_ref':        tf.TensorShape([B, flt_fs, flt_fs]),
  #    'ref_DOY':            tf.TensorShape([B]),
  #    'DOYDiff_TmR':        tf.TensorShape([B]),
  #    'ARD_LST_tgt_masked': tf.TensorShape([B, flt_fs, flt_fs]),
  #    'tgt_DOY':            tf.TensorShape([B]),
  #    'DOYDiff_RmT':        tf.TensorShape([B]),
  #    'Mask_Img':           tf.TensorShape([B, flt_fs, flt_fs])
  # label
  #    'ARD_LST_tgt, Mask_Img': tf.TensorShape([B, flt_fs, flt_fs, 2])

  cc_threshold_ts = tf.convert_to_tensor(cc_threshold, dtype=tf.float32)
  filter_cc_fn = filter_cc(flt_fs, cc_threshold_ts)

  dataset = tf.data.Dataset.list_files(files_pattern, seed=shuffle_seed + 100)
  if (buffer_size is not None):
    dataset = dataset.shard(num_workers, worker_index)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.shuffle(buffer_size, seed=shuffle_seed, reshuffle_each_iteration=True)
  else:
    dataset = dataset.repeat(num_epochs)
    
  dataset = dataset.interleave(
    lambda x: tf.data.TFRecordDataset(
      x,
      compression_type=compression_type,
      num_parallel_reads=1 #num_parallel_reads #1
    ),
    cycle_length=num_parallel_reads,
    block_length=1,
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
    deterministic=True
  )

  dataset = dataset.filter(filter_cc_fn)

  dataset = dataset.batch(
    batch_size,
    drop_remainder=drop_remainder
  )
  dataset = dataset.map(parser_method, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  # dataset = dataset.cache() # cache to memory

  dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
  
  return dataset


"""##  combined (transformed) data creation function"""

### ***** currently USED as of 2/7/2020 ***** ###
# this version supports finding files on gcp storage
def create_combined_transformed_dataset_v3d1_gs(
  gs_file_pattern, 
  batch_size, 
  buffer_size, 
  shuffle_seed=42, 
  num_epochs=1, 
  num_parallel_reads=1, 
  num_parallel_calls=2, 
  flt_fs=64
  ):
  
  transformed_file_list = tf.io.matching_files(gs_file_pattern)
  
  #read all combined zip_crop_flip tfrecords in transformed_file_list into a datset
  combined_transformed_dataset_v3 = read_batch_zip_crop_flip_records_train_v3(
    transformed_file_list, 
    compression_type = "GZIP",
    batch_size = batch_size,  #20
    num_epochs = num_epochs, #1, 
    buffer_size = buffer_size, 
    shuffle_seed = shuffle_seed,
    padding_values = (
      (
        tf.constant(-9998.0, dtype=tf.float32),
        tf.constant(-9998.0, dtype=tf.float32),
        tf.constant(-9998.0, dtype=tf.float32),
        tf.constant(-9998.0, dtype=tf.float32),
        tf.constant(-9998.0, dtype=tf.float32),
        tf.constant(-9998.0, dtype=tf.float32),
        tf.constant(-9998.0, dtype=tf.float32)
      ),
      (
        tf.constant(-9998.0, dtype=tf.float32),
        tf.constant(-9998.0, dtype=tf.float32)
      )
    ),
    ## new Dataset v1.8 features
    num_parallel_reads = num_parallel_reads, #1,
    num_parallel_calls = num_parallel_calls, #2, # suggestion: use the number of available CPU cores
    prefetch_buffer_size = 1,
    ## new Dataset v1.13 features
    num_workers = 1,
    worker_index = 0,
    drop_remainder = True, #False,
    #
    # application specific parameters
    parser_method= _parse_SeqExampleProto_zip_crop_flip_v3,
    flt_fs = flt_fs
  )
  
  return combined_transformed_dataset_v3


def create_combined_transformed_dataset_v3d1_gs_wCFilter(
  gs_file_pattern,
  batch_size,
  buffer_size,
  shuffle_seed=42,
  num_epochs=1,
  num_parallel_reads=1,
  num_parallel_calls=2,
  flt_fs=64,
  cc_threshold=0.0
):
  # files = tf.matching_files("gs://my-bucket/train-data-*.csv")
  # transformed_file_list = generate_file_list_transformed(transformed_file_dir)
  transformed_file_list = tf.io.matching_files(gs_file_pattern)

  # read all combined zip_crop_flip tfrecords in transformed_file_list into a datset
  combined_transformed_dataset_v3 = read_batch_zip_crop_flip_records_train_v3_wCFilter(
    transformed_file_list,
    compression_type="GZIP",
    batch_size=batch_size,  # 20
    num_epochs=num_epochs,  # 1,
    buffer_size=buffer_size,
    shuffle_seed=shuffle_seed,
    padding_values=(
      (
        tf.constant(-9998.0, dtype=tf.float32),
        tf.constant(-9998.0, dtype=tf.float32),
        tf.constant(-9998.0, dtype=tf.float32),
        tf.constant(-9998.0, dtype=tf.float32),
        tf.constant(-9998.0, dtype=tf.float32),
        tf.constant(-9998.0, dtype=tf.float32),
        tf.constant(-9998.0, dtype=tf.float32)
      ),
      (
        tf.constant(-9998.0, dtype=tf.float32),
        tf.constant(-9998.0, dtype=tf.float32)
      )
    ),
    ## new Dataset v1.8 features
    num_parallel_reads=num_parallel_reads,  # 1,
    num_parallel_calls=num_parallel_calls,  # 2, # suggestion: use the number of available CPU cores
    prefetch_buffer_size=1,
    ## new Dataset v1.13 features
    num_workers=1,
    worker_index=0,
    drop_remainder=True,  # False,
    #
    # application specific parameters
    parser_method=_parse_SeqExampleProto_zip_crop_flip_v3,
    flt_fs=flt_fs,
    cc_threshold=cc_threshold
  )

  return combined_transformed_dataset_v3


### ***** previously USED as of 02/11/2020 ***** ###
# this version supports finding
def create_combined_transformed_dataset_v4(
  gs_file_pattern,
  batch_size,
  buffer_size,
  shuffle_seed=42,
  num_epochs=1,
  num_parallel_reads=1,
  num_parallel_calls=2,
  flt_fs=64
  ):

  # read all combined zip_crop_flip tfrecords in transformed_file_list into a datset
  combined_transformed_dataset_v4 = read_batch_zip_crop_flip_records_train_v4(
    gs_file_pattern, #transformed_file_list,
    compression_type="GZIP",
    batch_size=batch_size,  # 20
    num_epochs=num_epochs,  # 1,
    buffer_size=buffer_size,
    shuffle_seed=shuffle_seed,
    ## new Dataset v1.8 features
    num_parallel_reads=num_parallel_reads,  # 1,
    num_parallel_calls=num_parallel_calls,  # 2, # suggestion: use the number of available CPU cores
    prefetch_buffer_size=tf.data.experimental.AUTOTUNE, #1,
    ## new Dataset v1.13 features
    num_workers=1,
    worker_index=0,
    drop_remainder=True,  # False,
    #
    # application specific parameters
    parser_method=parser_v4(flt_fs), #_parse_SeqExampleProto_zip_crop_flip_v4,
    flt_fs=flt_fs
  )
  
  return combined_transformed_dataset_v4

#####
### ***** currently USED as of 08/19/2020 ***** ###
# with filtering on correlation coefficient between the target and reference pair.
def create_combined_transformed_dataset_v4_wCFilter(
  gs_file_pattern,
  batch_size,
  buffer_size,
  shuffle_seed=42,
  num_epochs=1,
  num_parallel_reads=1,
  num_parallel_calls=2,
  flt_fs=64,
  cc_threshold = 0.0
  ):
  # files = tf.matching_files("gs://my-bucket/train-data-*.csv")
  # transformed_file_list = generate_file_list_transformed(transformed_file_dir)
  # transformed_file_list = tf.io.matching_files(gs_file_pattern)

  # read all combined zip_crop_flip tfrecords in transformed_file_list into a datset
  combined_transformed_dataset_v4_wCFilter = read_batch_zip_crop_flip_records_train_v4_wCFilter(
    gs_file_pattern,  # transformed_file_list,
    compression_type="GZIP",
    batch_size=batch_size,  # 20
    num_epochs=num_epochs,  # 1,
    buffer_size=buffer_size,
    shuffle_seed=shuffle_seed,
    ## new Dataset v1.8 features
    num_parallel_reads=num_parallel_reads,  # 1,
    num_parallel_calls=num_parallel_calls,  # 2, # suggestion: use the number of available CPU cores
    prefetch_buffer_size=tf.data.experimental.AUTOTUNE,  # 1,
    ## new Dataset v1.13 features
    num_workers=1,
    worker_index=0,
    drop_remainder=True,  # False,
    #
    cc_threshold = cc_threshold, # correlation coefficient threshold (examples with cc above which is allowed in.)
    #
    # application specific parameters
    parser_method=parser_v4(flt_fs),  # _parse_SeqExampleProto_zip_crop_flip_v4,
    flt_fs=flt_fs
  )

  return combined_transformed_dataset_v4_wCFilter


##----------------------------------------------------------------------------------------------##
def count_tfrecords_examples_gs(gs_file_pattern, compression_type="GZIP", num_parallel_reads=1):
  filenames = tf.io.matching_files(gs_file_pattern)
  dataset = tf.data.TFRecordDataset(
    filenames, 
    compression_type=compression_type, 
    num_parallel_reads=num_parallel_reads)

  # tf v2.0
  cnt = dataset.reduce(np.int64(0), lambda x, _: x + 1)
  cnt_val = cnt.numpy()
  if (isinstance(cnt_val, tuple) or isinstance(cnt_val, list)):
    cnt_val = cnt_val[0]  # b/c cnt returns a tuple e.g. (5386500,)
  return cnt_val

  # tf v1.x
  # dataset = dataset.batch(1)
  # dataset = dataset.repeat(1)
  # cnt = dataset.reduce(np.int64(0), lambda x, _: x + 1)
  #
  # with tf.Session() as sess:
  #   cnt_val = sess.run(cnt)
  #   if (isinstance(cnt_val, tuple) or isinstance(cnt_val, list)):
  #     cnt_val = cnt_val[0] # b/c cnt returns a tuple e.g. (5386500,)
  # return cnt_val

def count_tfrecords_examples_gs_tf2(gs_file_pattern, compression_type="GZIP", num_parallel_reads=1):
  filenames = tf.io.matching_files(gs_file_pattern)
  dataset = tf.data.TFRecordDataset(
    filenames,
    compression_type=compression_type,
    num_parallel_reads=num_parallel_reads)
  dataset = dataset.batch(1)
  dataset = dataset.repeat(1)

  cnt_val = dataset.reduce(np.int64(0), lambda x, _: x + 1).numpy()
  if (isinstance(cnt_val, tuple) or isinstance(cnt_val, list)):
    cnt_val = cnt_val[0]  # b/c cnt returns a tuple e.g. (5386500,)

  return cnt_val

##----------------------------------------------------------------------------------------------##

#     padding_values = (
#       {
#         'ARD_LST_ref':        tf.constant(-9998.0, dtype=tf.float32),
#         'ref_DOY':            tf.constant(-9998.0, dtype=tf.float32),
#         'DOYDiff_TmR':        tf.constant(-9998.0, dtype=tf.float32),
#         'ARD_LST_tgt_masked': tf.constant(-9998.0, dtype=tf.float32),
#         'tgt_DOY':            tf.constant(-9998.0, dtype=tf.float32),
#         'DOYDiff_RmT':        tf.constant(-9998.0, dtype=tf.float32),
#         'Mask_Img':           tf.constant(-9998.0, dtype=tf.float32)
#       },


##----------------------------------------------------------------------------------------------##
"""## data preprocess layer (v2, tuple instead of dict inputs, for normalized input data)
"""

#     'ARD_LST_ref': tf.io.FixedLenFeature(shape=[flt_fs*flt_fs], dtype=tf.float32), #_float_feature(ARD_LST_ref[iiBatch,:,:].reshape((-1))),
#     'ARD_LST_tgt_masked': tf.io.FixedLenFeature(shape=[flt_fs*flt_fs], dtype=tf.float32), #_float_feature(ARD_LST_tgt_masked[iiBatch,:,:].reshape((-1))),
#     'Mask_Img': tf.io.FixedLenFeature(shape=[flt_fs*flt_fs], dtype=tf.float32),
#     'ref_tgt_DOY_difference': tf.io.FixedLenFeature(shape = [], dtype=tf.float32), #_float_feature(ref_tgt_DOY_difference[iiBatch]),
#     'ref_DOY': tf.io.FixedLenFeature(shape = [], dtype=tf.float32), #_float_feature(ref_DOY[iiBatch]),
#     'tgt_DOY': tf.io.FixedLenFeature(shape = [], dtype=tf.float32), #_float_feature(tgt_DOY[iiBatch]),
#     'ref_tgt_DOY_difference_reverse': tf.io.FixedLenFeature(shape = [], dtype=tf.float32), #_float_feature(ref_tgt_DOY_difference_reverse[iiBatch]),
#     'mean_feature': tf.io.FixedLenFeature(shape = [], dtype=tf.float32), #_float_feature(mean_featureNP[iiBatch]),

#     'ARD_LST_tgt': tf.io.FixedLenFeature(shape=[flt_fs*flt_fs], dtype=tf.float32) #_float_feature(ARD_LST_tgt[iiBatch,:,:].reshape((-1)))

# This version is for normalized input data, i.e., no BN is performed in the class
class missing_pixel_reconstruct_data_preprocess_layer_v2(K.layers.Layer):
    
  #Constructor
  def __init__(
    self,
    # 
    Used_Img_size,
    data_format,
    #
    dtype="float32",
    debug_print=False,      
    #
    *args, 
    **kwargs
    ):
    
    super().__init__(dtype=dtype, *args, **kwargs)
    
    #
    self.Used_Img_size = Used_Img_size
    
    self.LayerName = self.name
    self.debug_print = debug_print
    
    self.data_format = data_format
    
    #
    self.epsilon = 6e-8 # minimum of float16
    
    #
    self.input_spec = (
      tf.keras.layers.InputSpec(ndim=3), # [B, H, W]
      tf.keras.layers.InputSpec(ndim=1), # [B]
      tf.keras.layers.InputSpec(ndim=1), # [B]
      tf.keras.layers.InputSpec(ndim=3), # [B, H, W]
      tf.keras.layers.InputSpec(ndim=1), # [B]
      tf.keras.layers.InputSpec(ndim=1), # [B]
      tf.keras.layers.InputSpec(ndim=3)  # [B, H, W]
    )

    #
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    self.channel_axis = channel_axis

  
  def build(self, input_shape):
    
    if (type(input_shape) == tf.TensorShape):
      input_shape = tuple(input_shape.as_list())

    if (type(input_shape) == list):
      ii_ts = 0
      for i_ts in input_shape:
        if (type(i_ts) == tf.TensorShape):
          input_shape[ii_ts] = tuple(i_ts.as_list())
          ii_ts = ii_ts + 1
    
    #print("data_preprocess_layer: input_shape[0]: ", input_shape[0])

    ## get the dynamic shape of img_output, mask_output (same)
    img3D_dyn_shp = tf.TensorShape((None,)).concatenate(tf.TensorShape(input_shape[0][1:]))
    img3D_TSpec = tf.TensorSpec(shape=img3D_dyn_shp, dtype=self._compute_dtype)

    self.populate_batch_scalar_to_batch_image_v2_ins = self.populate_batch_scalar_to_batch_image_v2 \
      .get_concrete_function(tf.TensorSpec(shape=(None,), dtype=self._compute_dtype), img3D_TSpec)

    self.built = True

  ## implement this method as required by keras.
  def compute_output_shape(self, input_shape):
    
    tf.print("Input shape of preprocessing layer:", input_shape)

    ## ref_BN_Img:        [B, Used_Img_size, Used_Img_size, 3]
    ## tgt_masked_BN_Img: [B, Used_Img_size, Used_Img_size, 3]
    ## Mask_Img:          [B, Used_Img_size, Used_Img_size, 1]
    
    batch_shape = (input_shape[0][0],)
    new_space = tuple(self.Used_Img_size, self.Used_Img_size)

    if self.data_format == 'channels_first':
      ref_BN_Img_shape = batch_shape + (3,) + new_space
      tgt_masked_BN_Img_shape = batch_shape + (3,) + new_space
      Mask_Img_shape = batch_shape + (1,) + new_space
    else:
      ref_BN_Img_shape = batch_shape + new_space + (3,)
      tgt_masked_BN_Img_shape = batch_shape + new_space + (3,)
      Mask_Img_shape = batch_shape + new_space + (1,)
    
    tf.print("Output shapes of preprocessing layer: ", tf.TensorShape(ref_BN_Img_shape), tf.TensorShape(tgt_masked_BN_Img_shape), tf.TensorShape(Mask_Img_shape))
    return [tf.TensorShape(ref_BN_Img_shape), tf.TensorShape(tgt_masked_BN_Img_shape), tf.TensorShape(Mask_Img_shape)]

  @tf.function
  def populate_batch_scalar_to_batch_image_v2(self, Scalar_ts, ts_w_img_shp):
    #ret_b_Img = tf.ones_like(ts_w_img_shp, dtype=self.dtype) * tf.reshape(Scalar_ts, [-1, 1, 1])
    ret_b_Img = tf.ones_like(ts_w_img_shp, dtype=self._compute_dtype) * tf.reshape(Scalar_ts, [-1, 1, 1])
    return ret_b_Img

  def call(self, inputs, training=True):
    
    # inputs:
    # (
    #   0: 'ARD_LST_ref':        [B, H, W]
    #   1: 'ref_DOY':            [B]
    #   2: 'DOYDiff_TmR':        [B]
    #   3: 'ARD_LST_tgt_masked': [B, H, W]
    #   4: 'tgt_DOY':            [B]
    #   5: 'DOYDiff_RmT':        [B]
    #   6: 'Mask_Img':           [B, H, W]
    # )
    # 

    ARD_LST_ref        = tf.cast(inputs[0], dtype=self._compute_dtype) # [B, H, W]
    ref_DOY            = tf.cast(inputs[1], dtype=self._compute_dtype) # [B]
    DOYDiff_TmR        = tf.cast(inputs[2], dtype=self._compute_dtype) # [B]
    ARD_LST_tgt_masked = tf.cast(inputs[3], dtype=self._compute_dtype) # [B, H, W]
    tgt_DOY            = tf.cast(inputs[4], dtype=self._compute_dtype) # [B]
    ## DOYDiff_RmT should be zeros, so no need to read in.
    #DOYDiff_RmT        = tf.cast(inputs[5], dtype=self.dtype) # [B]
    Mask_Img           = tf.cast(inputs[6], dtype=self._compute_dtype) # [B, H, W]

    # populat DOYs to images
    ## shape: [B, Used_Img_size, Used_Img_size]
    ref_DOY_Img = tf.ones_like(ARD_LST_ref, dtype=self._compute_dtype) * tf.reshape(ref_DOY, [-1, 1, 1])  # 'ref_DOY_Img'
    DOYDiff_TmR_Img = tf.ones_like(ARD_LST_ref, dtype=self._compute_dtype) * tf.reshape(DOYDiff_TmR, [-1, 1, 1])
    tgt_DOY_Img = tf.ones_like(ARD_LST_ref, dtype=self._compute_dtype) * tf.reshape(tgt_DOY, [-1, 1, 1])
    # DOYDiff_RmT_Img = tf.ones_like(ARD_LST_ref, dtype=self._compute_dtype) * tf.reshape(DOYDiff_RmT, [-1, 1, 1])

    DOYDiff_RmT_Img = tf.zeros_like(DOYDiff_TmR_Img, dtype=self._compute_dtype)

    # split into 3-band ref and tgt images
    # ref_BN_Img:        [B, Used_Img_size, Used_Img_size, 3]
    # tgt_masked_BN_Img: [B, Used_Img_size, Used_Img_size, 3]
    # ref_masked_BN_Img: [B, Used_Img_size, Used_Img_size, 3]
    ref_BN_Img        = tf.stack([ARD_LST_ref,        ref_DOY_Img, DOYDiff_TmR_Img], axis=-1)
    tgt_masked_BN_Img = tf.stack([ARD_LST_tgt_masked, tgt_DOY_Img, DOYDiff_RmT_Img], axis=-1)
    ref_masked_BN_Img = tf.stack([ARD_LST_ref * Mask_Img, ref_DOY_Img, DOYDiff_TmR_Img], axis=-1)
    
    # adjust mask shape to 4D tensor
    ## [B, Used_Img_size, Used_Img_size, 1]
    Mask_Img = tf.expand_dims(Mask_Img, axis=-1)

    ### ----------------------------------------------------------------------- ###
    ### duplicated/populated (in feature dim.) mask
    ### The reason perform this operation here is to avoid change input tensor shape for pconv layer,
    ###   which should be avoided when using tf.function
    ###
    num_input_channels = tgt_masked_BN_Img.shape[self.channel_axis]

    ### shape of Mask_Img_3bands: [B, Used_Img_size, Used_Img_size, 3]
    Mask_Img_3bands = tf.keras.backend.repeat_elements(
      Mask_Img, num_input_channels, self.channel_axis)

    # [B, Used_Img_size, Used_Img_size, 1]
    tgt_masked_BN_1ly = tf.expand_dims(ARD_LST_tgt_masked, -1)

    ### ----------------------------------------------------------------------- ###
    #
    ## ref_BN_Img:        [B, Used_Img_size, Used_Img_size, 3]
    ## tgt_masked_BN_Img: [B, Used_Img_size, Used_Img_size, 3]
    ## ref_masked_BN_Img: [B, Used_Img_size, Used_Img_size, 3]
    ## Mask_Img:          [B, Used_Img_size, Used_Img_size, 1]
    ## Mask_Img_3bands:   [B, Used_Img_size, Used_Img_size, 3]
    ## tgt_masked_BN_1ly: [B, Used_Img_size, Used_Img_size, 1]
    return [ref_BN_Img, tgt_masked_BN_Img, ref_masked_BN_Img, Mask_Img, Mask_Img_3bands,
            tgt_masked_BN_1ly
            ]

  def get_config(self):
    config = super().get_config()
    config.update({
      'Used_Img_size': self.Used_Img_size,
      'data_format': self.data_format,
      'dtype': self._dtype_policy, #self.dtype,
      'debug_print': self.debug_print
      })
    return config


