from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from .layers.partial_merge_layer import PartialMergeLayer
from .layers.reftgtencodelyr import RefTgtEncodeLyr
from .layers.reference_target_decoder_layer import DecodeLayer
from .layers.data_preprocess_layer_v2 import missing_pixel_reconstruct_data_preprocess_layer_v2
from .layers.lambda_layers import (tensor_copy_layer, slice_copy_layer, out_stack_layer)
from .layers.mask_batch_means import rescale_based_on_masked_img_layer
from .utils.string_2_function import (
  string_to_activation,
  string_to_regularizer
)

## create the SAPC2 MPR model (with funtional API)
def create_MPR_Model_fnAPI(
  used_img_size, 
  custom_dtype, 
  data_format,
  debug_print,
  parameter_dict
  ):
  
  float32_dp = tf.keras.mixed_precision.experimental.Policy("float32")
  
  ### ----------------------------------------------------------------------- ###
  # layer inits
  if data_format == 'channels_first':
    channel_axis = 1
  else:
    channel_axis = -1
    
  ### ----------------------------------------------------------------------- ###
  preprocess_layer = missing_pixel_reconstruct_data_preprocess_layer_v2(
    used_img_size, 
    data_format, 
    dtype=float32_dp, #custom_dtype,
    debug_print=debug_print#, 
    #input_shape=input_shape
    )

  ### ----------------------------------------------------------------------- ###
  ## layer for making a copy of in_ref as the first TBMBD_Img
  TBMBD_Img_copy_layer = tensor_copy_layer(ndim=4, dtype=float32_dp)  # custom_dtype

  ### ----------------------------------------------------------------------- ###
  ##
  FstSkConn_params = parameter_dict['first_skip_connection']
  FstSkConn_filters = FstSkConn_params['pmerge_filters']
  FstSkConn_kernel_shape = FstSkConn_params['pmerge_kernel_shape']

  FstSkConn_kernel_regularizer = string_to_regularizer(
    FstSkConn_params,
    regularization_function_key='pmerge_kernel_regularization_function',
    name_prefix='pmerge_kernel_')

  FstSkConn_use_bias = FstSkConn_params['use_bias']

  #FstSkConn_GPMerge_PConv_activation = string_to_activation(FstSkConn_params['gpmerge_activation'])
  FstSkConn_PMerge_activation = None
  #FstSkConn_GPMerge_PConv_use_batch_normalization = True
  FstSkConn_PMerge_use_batch_normalization = True
  FstSkConn_compress_ratio = FstSkConn_params['compress_ratio'] # 16.0/3.0

  FstSkConn_LayerName = FstSkConn_params['layer_name']
  
  FstSkConn_PartialMergeLyr = PartialMergeLayer(
    filters=FstSkConn_filters,
    kernel_size=FstSkConn_kernel_shape,
    #
    data_format=data_format,  # 'channels_last',
    activation=FstSkConn_PMerge_activation,
    use_batch_normalization=FstSkConn_PMerge_use_batch_normalization,
    use_bias=FstSkConn_use_bias,
    #
    kernel_regularizer=FstSkConn_kernel_regularizer,
    #
    dtype=custom_dtype,
    name=FstSkConn_LayerName
  )

  ### ----------------------------------------------------------------------- ###
  num_encoders = len(parameter_dict['encoder_list'])
  
  encoder_layers = []
  ec_cp_layers_list = [] #ec_ts_copy_ins_list = []
  
  iiLyr = 0
  
  for encoder_params in parameter_dict['encoder_list']:
    
    if (iiLyr == 0):
      dtype_iiLyr = float32_dp #tf.keras.mixed_precision.experimental.Policy("float32")
    else:
      dtype_iiLyr = custom_dtype
  
    PConv_activation = string_to_activation(encoder_params)
    PConv_strides = int(round(encoder_params['input_spatial'] / encoder_params['output_spatial']))
    PConv_kernel_regularizer = string_to_regularizer(
      encoder_params,
      regularization_function_key = 'pconv_kernel_regularization_function',
      name_prefix='pconv_kernel_')
    PMerge_kernel_regularizer = string_to_regularizer(
      encoder_params,
      regularization_function_key = 'pmerge_kernel_regularization_function',
      name_prefix='pmerge_kernel_')
    GPMerge_PConv_activation = string_to_activation(encoder_params['gpmerge_activation'])
    if (iiLyr < num_encoders - 1):
      PMerge_activation = None
      PMerge_use_batch_normalization = True
      skip_residual_before_pmerge = True
    else:
      PMerge_activation = string_to_activation(encoder_params['gpmerge_activation']) # PRelu
      PMerge_use_batch_normalization = True #False #True #False
      skip_residual_before_pmerge = True
      print('last layer [{}] PMerge\'s activation {}, use_BN {}, skip_residual {}: '.format(
        iiLyr, PMerge_activation, PMerge_use_batch_normalization, skip_residual_before_pmerge
      ))
    compress_ratio = encoder_params['compress_ratio']

    encoder = RefTgtEncodeLyr(
      encoder_params['output_feature'],
      tuple(encoder_params['pconv_kernel_shape']),
      encoder_params['output_feature'],
      tuple(encoder_params['pmerge_kernel_shape']),
      data_format,
      PConv_strides=PConv_strides,
      PConv_activation=PConv_activation,
      PConv_kernel_regularizer=PConv_kernel_regularizer,
      PConv_use_batch_normalization=encoder_params['batch_normalization'],
      #
      PMerge_kernel_regularizer=PMerge_kernel_regularizer,
      GPMerge_PConv_activation=GPMerge_PConv_activation,
      PMerge_activation=PMerge_activation,
      #GPMerge_PConv_use_batch_normalization=GPMerge_PConv_use_batch_normalization,
      PMerge_use_batch_normalization=PMerge_use_batch_normalization,
      compress_ratio=compress_ratio,
      skip_residual_before_pmerge=skip_residual_before_pmerge,
      #
      name=encoder_params['encoder_name'],
      dtype=dtype_iiLyr,
      debug_print=debug_print
    )
    encoder_layers.append(encoder)

    print(iiLyr, "PConv_strides", PConv_strides, " | ", encoder_params)

    iiLyr = iiLyr + 1

  iiLyr = 0
  for encoder_params in parameter_dict['encoder_list']:
    ## full encoder and the last layer PCONV share the same procedure to generate output shape,
    ## that's why the two branches use only one block to determine the instance of tensor_copy function.
    ## Note: different layers still have different instances.
    # 0: Activated Output (Reference) [B, H/S, W/S, OutFeatures] or [B, OutFeatures, H/S, W/S]
    # 1: Partial Merged Output [B, H/S, W/S, OutFeatures] or [B, OutFeatures, H/S, W/S]
    # 2: Updated Mask (Target) [B, H/S, W/S, OutFeatures] or [B, OutFeatures, H/S, W/S]
    # 3: Activated Output (Target) [B, H/S, W/S, OutFeatures] or [B, OutFeatures, H/S, W/S]
    out_img_spatial_iiLyr = int(encoder_params['output_spatial'])
    ec_4D_copy_layer_iiLyr = tensor_copy_layer(ndim=4, dtype=custom_dtype)
    ec_cp_layers_list.append(ec_4D_copy_layer_iiLyr)

    #print(iiLyr, B_HdS_WdS_OF_iiLyr_dyn_shp_raw, ec_4D_copy_layer_iiLyr, ec_cp_layers_list[-1])

    iiLyr = iiLyr + 1
  
  ### ----------------------------------------------------------------------- ###
  num_decoders = len(parameter_dict['decoder_list'])
  
  decoder_layers = []
  dc_cp_layers_list = []
  
  iiLyrDe = 0
  
  for decoder_params in parameter_dict['decoder_list']:
  
    Merge_activation = string_to_activation(decoder_params)
    Merge_kernel_regularizer = string_to_regularizer(
      decoder_params,
      regularization_function_key='merge_kernel_regularization_function',
      name_prefix='merge_kernel_')
  
    USF = int(round( decoder_params['output_spatial'] / decoder_params['input_spatial']))

    Merge_use_batch_normalization_iLyr = decoder_params['batch_normalization']

    if (iiLyrDe < num_encoders - 1):
      dtype_iiLyrDe = custom_dtype
      use_BP_MSK = True
      UMLoss_weight = decoder_params['umloss_weight']
    else:
      dtype_iiLyrDe = tf.keras.mixed_precision.experimental.Policy("float32")
      use_BP_MSK = False
  
    decoder = DecodeLayer(
      USF,
      decoder_params['output_feature'],
      tuple(decoder_params['merge_kernel_shape']),
      Merge_data_format=data_format,
      Merge_activation=Merge_activation,
      Merge_kernel_regularizer=Merge_kernel_regularizer,
      Merge_use_batch_normalization=Merge_use_batch_normalization_iLyr,
      Merge_use_bias=True,
      use_BP_MSK=use_BP_MSK,
      UMLoss_weight=UMLoss_weight,
      name=decoder_params['decoder_name'],
      dtype=dtype_iiLyrDe,
      debug_print=debug_print
      )
  
    decoder_layers.append(decoder)
    
    ##
    out_img_spatial_iiLyr = int(decoder_params['output_spatial'])
    dc_4D_copy_layer_iiLyr = tensor_copy_layer(ndim=4, dtype=custom_dtype)
    dc_cp_layers_list.append(dc_4D_copy_layer_iiLyr)
    
    tf.print(iiLyrDe, "USF", USF, " | ", decoder_params)
    #print(iiLyrDe, "USF", USF, " | ", decoder_params, " || ", B_HmU_WmU_OF_iiLyr_dyn_shp_raw)
  
    iiLyrDe = iiLyrDe + 1
    
  ### ----------------------------------------------------------------------- ###
  output_linear_params = parameter_dict['output_linear']
  output_linear_layer_regularizer = string_to_regularizer(
    output_linear_params,
    regularization_function_key='output_linear_kernel_regularization_function',
    name_prefix='output_linear_kernel_')

  output_linear_layer = tf.keras.layers.Conv2D(
    filters = 1,
    kernel_size = (1,1),
    strides=(1, 1),
    padding='same',
    data_format=data_format,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=output_linear_layer_regularizer,
    bias_regularizer=output_linear_layer_regularizer,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    dtype="float32" #custom_dtype
  )

  ### ----------------------------------------------------------------------- ###
  # layer for adjusting the mean and std of the reconstructed image
  # based on those values in the unmasked regions of tgt and ref
  # dtype=float32 b/c there is a division operation inside
  mean_std_rescale_layer = rescale_based_on_masked_img_layer(
    spatial_dim_H=used_img_size, dtype="float32")
  
  ### ----------------------------------------------------------------------- ###
  # layer for making a copy of rescaled_image
  rescaled_img_slice_copy_layer = slice_copy_layer(ndim=3, dtype="float32")  # custom_dtype

  ### ----------------------------------------------------------------------- ###
  o_stack_layer = out_stack_layer(axis=channel_axis, dtype="float32")  # custom_dtype

  ### ----------------------------------------------------------------------- ###
  ###    forward pass                                                         ###
  ### ----------------------------------------------------------------------- ###
  
  ### ----------------------------------------------------------------------- ###
  # (
  #   0: 'ARD_LST_ref'        [B, H, W]
  #   1: 'ref_DOY'            [B]
  #   2: 'DOYDiff_TmR'        [B]
  #   3: 'ARD_LST_tgt_masked' [B, H, W]
  #   4: 'tgt_DOY'            [B]
  #   5: 'DOYDiff_RmT'        [B]
  #   6: 'Mask_Img'           [B, H, W]
  # )    
  ##
  
  ### ----------------------------------------------------------------------- ###
  input_0 = tf.keras.layers.Input(shape=(used_img_size, used_img_size), dtype=custom_dtype._variable_dtype)
  input_1 = tf.keras.layers.Input(shape=tuple(), dtype=custom_dtype._variable_dtype)
  input_2 = tf.keras.layers.Input(shape=tuple(), dtype=custom_dtype._variable_dtype)
  input_3 = tf.keras.layers.Input(shape=(used_img_size, used_img_size), dtype=custom_dtype._variable_dtype)
  input_4 = tf.keras.layers.Input(shape=tuple(), dtype=custom_dtype._variable_dtype)
  input_5 = tf.keras.layers.Input(shape=tuple(), dtype=custom_dtype._variable_dtype)
  input_6 = tf.keras.layers.Input(shape=(used_img_size, used_img_size), dtype=custom_dtype._variable_dtype)
  
  inputs = [input_0, input_1, input_2, input_3, input_4, input_5, input_6]
  
  ### ----------------------------------------------------------------------- ###
  # preprocessing layer
  ## inputs are tuples of raw mask image and raw pairs image from transformed dataset
  ## ref_BN_Img: [B, Used_Img_size, Used_Img_size, 3]
  ## tgt_BN_Img: [B, Used_Img_size, Used_Img_size, 3]
  ## ref_masked_BN_Img: [B, Used_Img_size, Used_Img_size, 3]
  ## Mask_Img:   [B, Used_Img_size, Used_Img_size, 1]
  ## Mask_Img_3bands: [B, Used_Img_size, Used_Img_size, 3]
  ## tgt_masked_BN_1ly: [B, Used_Img_size, Used_Img_size, 1]
  #ref_BN_Img, tgt_masked_BN_Img, Mask_Img, Mask_Img_3bands = preprocess_layer(inputs)
  ref_BN_Img, tgt_masked_BN_Img, ref_masked_BN_Img, Mask_Img, Mask_Img_3bands, tgt_masked_BN_1ly = \
    preprocess_layer(inputs)
  ### ----------------------------------------------------------------------- ###
  # the to-be-merged-by-decoder images from the encoder side
  ## len of TBMBD_Img_list: num_encoders
  TBMBD_Img_list = []
  BP_Img_list = []
  MSK_Img_list = []

  ### ----------------------------------------------------------------------- ###
  TBMBD_Img_0 = FstSkConn_PartialMergeLyr(
    [ref_BN_Img, tgt_masked_BN_Img, Mask_Img_3bands])
  TBMBD_Img_list.append(TBMBD_Img_0)

  BP_Img_list.append(tgt_masked_BN_1ly)
  MSK_Img_list.append(Mask_Img)
  
  ### ----------------------------------------------------------------------- ###
  # encoder layers
  PrevLyr_Ref_Img_list = [ref_BN_Img]
  PrevLyr_Tgt_Masked_Img_list = [tgt_masked_BN_Img]
  PrevLyr_Mask_Img_list = [Mask_Img_3bands]

  ## order of encoder inputs
  ## 0: TargetImage = inputs[0] # [B, H, W, C] or [B, C, H, W]
  ## 1: TargetMaskImage = inputs[1] # [B, H, W, 1 or C] or [B, 1 or C, H, W]
  ## 2: RefereceImage = inputs[2] # [B, H, W, C] or [B, C, H, W]
  
  ## order of encoder ouputs
  ## 0: Activated Output (Reference) [B, H/S, W/S, OutFeatures] or [B, OutFeatures, H/S, W/S]
  ## 1: Partial Merged Output [B, H/S, W/S, OutFeatures] or [B, OutFeatures, H/S, W/S]
  ## 2: Updated Mask (Target) [B, H/S, W/S, OutFeatures] or [B, OutFeatures, H/S, W/S]
  ## 3: Activated Output (Target) [B, H/S, W/S, OutFeatures] or [B, OutFeatures, H/S, W/S]
  
  ### ----------------------------------------------------------------------- ###
  
  iiLyr = 0
  
  for encoder_i in encoder_layers[0:num_encoders-1]:
    Acti_PConv_Ref_Img_i, PMerge_Img_i, Updated_Mask_Img_i, Acti_PConv_Tgt_Img_i = \
      encoder_i([PrevLyr_Tgt_Masked_Img_list[-1], PrevLyr_Mask_Img_list[-1],
                 PrevLyr_Ref_Img_list[-1]])  # , training=training

    ec_cp_iiLyr = ec_cp_layers_list[iiLyr]
    print("  using ec ", iiLyr, ec_cp_iiLyr)

    #print("using ec ", iiLyr, ec_ts_copy_ins_iiLyr)

    PrevLyr_Tgt_Masked_Img_list.append(Acti_PConv_Tgt_Img_i)
    PrevLyr_Mask_Img_list.append(Updated_Mask_Img_i)
    PrevLyr_Ref_Img_list.append(Acti_PConv_Ref_Img_i)

    TBMBD_Img_list.append(ec_cp_iiLyr(PMerge_Img_i))

    BP_Img_list.append(Acti_PConv_Tgt_Img_i)
    MSK_Img_list.append(Updated_Mask_Img_i)

    iiLyr = iiLyr + 1
  
  # # last layer (num_encoders - 1)
  # _, LastLyr_TWMerge_Img, _, _ = \
  #   encoder_layers[-1](
  #     [PrevLyr_Tgt_Masked_Img_list[-1], PrevLyr_Mask_Img_list[-1], PrevLyr_Ref_Img_list[-1]])
  #
  # # last ec batch normalization
  # LastLyr_TWMerge_Img = tf.keras.layers.BatchNormalization(
  #   trainable=True, epsilon=6e-8, axis=channel_axis, dtype=custom_dtype
  # )(LastLyr_TWMerge_Img)
  # # last ec activation layer
  # LastLyr_TWMerge_Img = tf.keras.layers.ReLU()(LastLyr_TWMerge_Img)


  ### ----------------------------------------------------------------------- ###
  # decoder layers
  ec_cp_iiLyr = ec_cp_layers_list[-1]
  PrevLyr_DecodedImg_list = [ec_cp_iiLyr(LastLyr_TWMerge_Img)]
  
  ## order of decoder inputs
  ## 0: InLowResImg = inputs[0] # [B, H, W, C_I] or [B, C_I, H, W]
  ## 1: MrgPConvImg = inputs[1] # [B, H*USF, W*USF, C_M] or [B, C_M, H*USF, W*USF]
  
  ## order of decoder outputs (1-element list)
  ## 0: DecodedImg # [B, H*USF, W*USF, OutFeatures] or [B, OutFeatures, H*USF, W*USF]
  
  iiLyrDe = 0
  
  for decoder_i in decoder_layers:
    decoder_res_list_i = decoder_i(
      #[PrevLyr_DecodedImg_list[-1], TBMBD_Img_list[num_encoders - 1 - iiLyrDe]],
      [PrevLyr_DecodedImg_list[-1],
       TBMBD_Img_list[num_encoders - 1 - iiLyrDe],
       BP_Img_list[num_encoders - 1 - iiLyrDe],
       MSK_Img_list[num_encoders - 1 - iiLyrDe]
       ],
      ) #training=training
  
    PrevLyr_DecodedImg_list.append(decoder_res_list_i)
    
    iiLyrDe = iiLyrDe + 1
  
  ### ----------------------------------------------------------------------- ###
  ## LastLyr_DecodedImg is the restored target image
  # [B, H, W, 1]
  dc_ts_copy_ins_last = dc_cp_layers_list[-1]
  LastLyr_DecodedImg = dc_ts_copy_ins_last(PrevLyr_DecodedImg_list[-1])
  
  # Add linear layer for output scaling.
  # [B, H, W, 1]
  linear_output_image = output_linear_layer(LastLyr_DecodedImg)
  # rescale image to unmask mean of the target.
  rescaled_image = mean_std_rescale_layer((linear_output_image, input_3, input_6))

  # [B, H, W]
  Restored_3DImg = rescaled_img_slice_copy_layer(rescaled_image)

  # [B, H, W, 2]
  out_ts = o_stack_layer([Restored_3DImg, input_0])
  
  fn_MPR_Model = tf.keras.Model(inputs=inputs, outputs=out_ts)
  
  return fn_MPR_Model 
  












