from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys

import tensorflow as tf

def construct_layer_shapes(input_shape, first_output_shape, data_format):
  
    if data_format == 'channels_first':
        spatial_x_axis = -2
        channel_axis = 1
    else:
        channel_axis = -1
        spatial_x_axis = 1
    
    #Encoder
    # Note input_shape[1] must be divisible by power of 2.
    ## TODO: add assert here?
    num_encoder_layers = tf.cast(
      tf.math.round(tf.math.log(tf.cast(input_shape[1], tf.float32)) / tf.math.log(tf.constant(2.0)) - 1), tf.int32)
    
    spatial_dimension_list_encoder = tf.ones((num_encoder_layers + 1), dtype = tf.int32) * input_shape[spatial_x_axis]
    out_feature_list_encoder = tf.ones((num_encoder_layers), dtype = tf.int32)

    range_tensor0 = tf.range(num_encoder_layers +1, dtype = tf.int32)
    range_tensor1 = tf.range(1, num_encoder_layers +1, dtype = tf.int32)
    
    denominator_vector = tf.pow(2, range_tensor0)
    
    spatial_dimension_list_encoder = spatial_dimension_list_encoder / denominator_vector
    
    out_feature_list_encoder = first_output_shape * tf.pow(2, range_tensor1 - 1)
    out_feature_list_encoder = tf.concat([[input_shape[channel_axis]], out_feature_list_encoder], axis = -1)
    
    out_feature_list_encoder = tf.clip_by_value(out_feature_list_encoder, 1, 512)
    
    #decoder
    out_feature_list_decoder = tf.reverse(out_feature_list_encoder, [-1])
    out_feature_list_decoder = tf.concat([
        out_feature_list_decoder[0:-1], tf.constant([1], dtype = tf.int32)],
        axis = -1)

    spatial_dimension_list_decoder = tf.reverse(spatial_dimension_list_encoder, [-1])

    return spatial_dimension_list_encoder, out_feature_list_encoder, \
           spatial_dimension_list_decoder, out_feature_list_decoder

# Num_layers here is one less than in the other params, because there is no extra for input
def construct_pconv_kernels(encoder_first_kernel_shape, num_layers, min_shape = 3):
    
    #Check to make sure kernels have valid shape
    assert encoder_first_kernel_shape[0] % 2 == 1, "encoder kernel must use an odd shape"
    
    # for encoders, the first layer is equal to the first shape, after that subtract 2 at each layer, an clip to 3
    # create range equal to num_layers
    range_tensor = tf.range(num_layers, dtype = tf.int32)
    #make its dimensions match the kernel by stacking on itself
    range_tensor = tf.stack([range_tensor, range_tensor], axis = 1)
    
    #create a constant tensor with the shape of num_layers and value of kernel shape
    encoder_kernels = tf.constant(encoder_first_kernel_shape[0], dtype = tf.int32, shape = (num_layers, 2))
    
    #subtract 2 * depth 
    encoder_kernels = encoder_kernels - range_tensor*2

    #clip to min, and also max, 
    encoder_kernels = tf.clip_by_value(encoder_kernels, min_shape, encoder_first_kernel_shape[0])
    
    return encoder_kernels

def construct_merge_kernels(merge_kernel_shape, num_layers):
    # decoder, is a constant shape
    merge_kernels = tf.constant(merge_kernel_shape[0], dtype = tf.int32, shape = (num_layers, 2))
    return merge_kernels

#construct_layer_kernels((7,7), (3,3), 5)

# Num_layers here is one less than in the other params, because there is no extra for input
def batch_normalization_list(use_bn,
                             num_layers,
                             first_layer_bn = False,
                             last_ec_layer_bn = True,
                             ):
    
    batch_normalization_list_encoders = [use_bn] * num_layers
    batch_normalization_list_decoders = batch_normalization_list_encoders.copy()
    batch_normalization_list_encoders[0] = first_layer_bn
    batch_normalization_list_encoders[-1] = last_ec_layer_bn

    ### CMS_DBG ###
    #batch_normalization_list_decoders[-1] = False # reference: PCONV paper
    ###############

    return batch_normalization_list_encoders, batch_normalization_list_decoders

#batch_normalization_list(True, 5)

# Num_layers here is one less than in the other params, because there is no extra for input
def activation_function_list(
    encoder_activation,
    decoder_activation,
    num_layers,
    encoder_activation_kwarg_dict={},
    decoder_activation_kwarg_dict={},
    last_ec_layer_act=True
    ):
    
    encoder_list = [encoder_activation] * num_layers
    decoder_list = [decoder_activation] * num_layers

    encoder_act_kwarg_list = [encoder_activation_kwarg_dict] * num_layers
    decoder_act_kwarg_list = [decoder_activation_kwarg_dict] * num_layers
    
    if not last_ec_layer_act:
        encoder_list[-1] = None
        encoder_act_kwarg_list[-1] = {}

    ### CMS_DBG ###
    #encoder_list[-1] = None
    # decoder_list[-1] = None
    # decoder_act_kwarg_list[-1] = {}
    ###############

    return encoder_list, decoder_list, encoder_act_kwarg_list, decoder_act_kwarg_list


#now, create a dictionary. 
#each top level entry will be a layer, reprsented as a dictionary.
#each of its properties will come from all the above lists. 

def create_params(input_shape, first_output_shape, init_pmerge_filters,
    data_format,
    encoder_first_kernel_shape, merge_kernel_shape, use_bn,
    encoder_activation, encoder_gpmerge_activation, decoder_activation,
    encoder_activation_kwarg_dict=None,
    encoder_gpmerge_activation_kwarg_dict={'init_alpha': 0.3},
    decoder_activation_kwarg_dict={'alpha': 0.3},
    last_ec_layer_act = True,
    first_layer_bn = False, last_ec_layer_bn=True,
    min_shape = 3,
    regularization_factor = 1e-5,
    compress_ratio=1.0,
    umloss_weight=1.0,
    SE_reduction_ratio = 16,
    SE_kernel_initializer_ec = 'he_normal',
    SE_kernel_initializer_dc = 'glorot_uniform',
    SE_kernel_initializer = 'glorot_uniform'
    ):
    
    #create the needed parameter lists
    spatial_dimension_list_encoder, \
    out_feature_list_encoder, \
    spatial_dimension_list_decoder, \
    out_feature_list_decoder = \
      construct_layer_shapes(input_shape, first_output_shape, data_format)
    
    num_layers = spatial_dimension_list_decoder.shape[0] - 1
    
    encoder_pconv_kernels = construct_pconv_kernels(encoder_first_kernel_shape, num_layers)
    encoder_pconv_kernel_regularization_function = 'l2'
    encoder_pconv_kernel_l1_regularization_factor = regularization_factor #1e-8 
    encoder_pconv_kernel_l2_regularization_factor = regularization_factor #1e-8
    
    encoder_merge_kernels = construct_merge_kernels(merge_kernel_shape, num_layers)
    encoder_merge_kernel_regularization_function = 'l2'
    encoder_merge_kernel_l1_regularization_factor = regularization_factor #1e-8 
    encoder_merge_kernel_l2_regularization_factor = regularization_factor #1e-8
    
    encoder_se_kernel_regularization_function = 'l2'
    encoder_se_kernel_l1_regularization_factor = regularization_factor #1e-8
    encoder_se_kernel_l2_regularization_factor = regularization_factor #1e-8


    # top_merge_kernel_regularization_function = 'l2'
    # top_merge_kernel_l1_regularization_factor = regularization_factor #1e-8
    # top_merge_kernel_l2_regularization_factor = regularization_factor #1e-8

    decoder_merge_kernels = construct_merge_kernels(merge_kernel_shape, num_layers)
    decoder_merge_kernel_regularization_function = 'l2'
    decoder_merge_kernel_l1_regularization_factor = regularization_factor #1e-8 
    decoder_merge_kernel_l2_regularization_factor = regularization_factor #1e-8
    
    decoder_se_kernel_regularization_function = 'l2'
    decoder_se_kernel_l1_regularization_factor = regularization_factor #1e-8
    decoder_se_kernel_l2_regularization_factor = regularization_factor #1e-8

    # do not restrict/regularize the kernel of the output_linear_kernel because
    # the kernel starts from 1.0 and should move freely to larger (positive) values.
    output_linear_kernel_regularization_function = None #'l2'
    output_linear_kernel_l1_regularization_factor = 0.0 #regularization_factor #1e-8
    output_linear_kernel_l2_regularization_factor = 0.0 #regularization_factor #1e-8

    
    encoder_bn, decoder_bn =  batch_normalization_list(use_bn, num_layers, first_layer_bn, last_ec_layer_bn)
    
    encoder_activation, decoder_activation, encoder_activation_kwarg_dict_list, decoder_activation_kwarg_dict_list = \
        activation_function_list(
            encoder_activation,
            decoder_activation,
            num_layers,
            encoder_activation_kwarg_dict=encoder_activation_kwarg_dict,
            decoder_activation_kwarg_dict=decoder_activation_kwarg_dict,
            last_ec_layer_act=last_ec_layer_act
            )

    #Could be a list that we append to in the U structure.
    parameter_dict = {}
    
    #dict will contain two lists, one for each side
    parameter_dict['encoder_list'] = []
    parameter_dict['decoder_list'] = []
    
    for i in range(num_layers):
        
        # name the encoder with the layer number
        encoder_name = "encoder_{}".format(i)

        #at that part of the dict, insert the following object
        parameter_dict['encoder_list'].append({
            
            #name
            "encoder_name": encoder_name,
            
            #define input spatial and feature shape
            "input_spatial" : spatial_dimension_list_encoder[i].numpy().tolist(),
            "input_feature" : out_feature_list_encoder[i].numpy().tolist(),
            
            #define output spatial and feature shape
            "output_spatial" : spatial_dimension_list_encoder[i+1].numpy().tolist(),
            "output_feature" : out_feature_list_encoder[i+1].numpy().tolist(),
            
            # define kernel, bn and activation in the same manner
            "pconv_kernel_shape": encoder_pconv_kernels[i].numpy().tolist(),
            "pconv_kernel_regularization_function": encoder_pconv_kernel_regularization_function,
            "pconv_kernel_l1_regularization_factor": encoder_pconv_kernel_l1_regularization_factor, 
            "pconv_kernel_l2_regularization_factor": encoder_pconv_kernel_l2_regularization_factor,

            "pmerge_kernel_shape": encoder_merge_kernels[i].numpy().tolist(),
            "pmerge_kernel_regularization_function": encoder_merge_kernel_regularization_function,
            "pmerge_kernel_l1_regularization_factor": encoder_merge_kernel_l1_regularization_factor,
            "pmerge_kernel_l2_regularization_factor": encoder_merge_kernel_l2_regularization_factor,
            "batch_normalization" : encoder_bn[i],
            #
            "activation_function" : encoder_activation[i],
            "activation_parameter_dict": encoder_activation_kwarg_dict_list[i],
            "regularization_factor": regularization_factor,
            #
            "gpmerge_activation": {
                "activation_function": encoder_gpmerge_activation,
                "activation_parameter_dict": encoder_gpmerge_activation_kwarg_dict,
                "regularization_factor": regularization_factor
            },
            "compress_ratio": compress_ratio,
            
            # Squeeze and Excitation
            "se_reduction_ratio": SE_reduction_ratio,
            ##
            "se_kernel_initializer_ec": SE_kernel_initializer_ec,
            "se_kernel_initializer_dc": SE_kernel_initializer_dc,
            "se_kernel_initializer": SE_kernel_initializer,
            ##
            "se_kernel_regularization_function": encoder_se_kernel_regularization_function,
            "se_kernel_l1_regularization_factor": encoder_se_kernel_l1_regularization_factor,
            "se_kernel_l2_regularization_factor": encoder_se_kernel_l2_regularization_factor
            #,
        })

        # name the encoder with the layer number
        decoder_name = "decoder_{}".format(i)
        
        #at that part of the dict, insert the following object
        parameter_dict['decoder_list'].append({
            
            #name
            "decoder_name": decoder_name,
            
            #define input spatial and feature shape
            "input_spatial" : spatial_dimension_list_decoder[i].numpy().tolist(),
            "input_feature" : out_feature_list_decoder[i].numpy().tolist(),
            
            #define output spatial and feature shape
            "output_spatial" : spatial_dimension_list_decoder[i+1].numpy().tolist(),
            "output_feature" : out_feature_list_decoder[i+1].numpy().tolist(),
            
            # define kernel, bn and activation in the same manner
            "merge_kernel_shape" : decoder_merge_kernels[i].numpy().tolist(),
            "merge_kernel_regularization_function": decoder_merge_kernel_regularization_function,
            "merge_kernel_l1_regularization_factor": decoder_merge_kernel_l1_regularization_factor,
            "merge_kernel_l2_regularization_factor": decoder_merge_kernel_l2_regularization_factor,
            "batch_normalization" : decoder_bn[i],
            #
            "activation_function" : decoder_activation[i],
            "activation_parameter_dict": decoder_activation_kwarg_dict_list[i],
            "regularization_factor": regularization_factor,
            #
            "psconv_activation": {
                "activation_function": encoder_activation[num_layers - 1 - i],
                "activation_parameter_dict": encoder_activation_kwarg_dict_list[num_layers - 1 - i],
                "regularization_factor": regularization_factor
            },
            'umloss_weight': umloss_weight,

            # Squeeze and Excitation
            "se_reduction_ratio": SE_reduction_ratio,
            ##
            "se_kernel_initializer_ec": SE_kernel_initializer_ec,
            "se_kernel_initializer_dc": SE_kernel_initializer_dc,
            "se_kernel_initializer": SE_kernel_initializer,
            ##
            "se_kernel_regularization_function": decoder_se_kernel_regularization_function,
            "se_kernel_l1_regularization_factor": decoder_se_kernel_l1_regularization_factor,
            "se_kernel_l2_regularization_factor": decoder_se_kernel_l2_regularization_factor
            # ,
        })

    # Linear layer (1x1 convolution)
    parameter_dict['output_linear'] = {
        "output_linear_kernel_regularization_function": output_linear_kernel_regularization_function,
        "output_linear_kernel_l1_regularization_factor": output_linear_kernel_l1_regularization_factor,
        "output_linear_kernel_l2_regularization_factor": output_linear_kernel_l2_regularization_factor
    }

    # intial partial merge layer (init_pmerge)
    parameter_dict['first_skip_connection'] = {
        "layer_name": "FirstSkipConnectionLayer",
        
        "pmerge_filters": init_pmerge_filters, #1, #init_pmerge_filters, #8,
        "pmerge_kernel_shape": [3,3],
        "pmerge_kernel_regularization_function": encoder_merge_kernel_regularization_function,
        "pmerge_kernel_l1_regularization_factor": encoder_merge_kernel_l1_regularization_factor,
        "pmerge_kernel_l2_regularization_factor": encoder_merge_kernel_l2_regularization_factor,
        
        "use_bias": True,
        
        "gpmerge_activation": {
            "activation_function": encoder_gpmerge_activation,
            "activation_parameter_dict": encoder_gpmerge_activation_kwarg_dict,
            "regularization_factor": regularization_factor
        },
        "compress_ratio": init_pmerge_filters * 2.0 / 3.0, #8.0/3.0, #compress_ratio,
        
    }
    
    # top merge layer
    spatial_dimension_last_encoder = spatial_dimension_list_encoder[-1].numpy().tolist()
    out_feature_last_encoder = out_feature_list_encoder[-1].numpy().tolist()
    
    parameter_dict['top_merge'] = {
        "layer_name": "top_merge",

        # define input spatial and feature shape
        "input_spatial": spatial_dimension_last_encoder,
        "input_feature": out_feature_last_encoder,

        # define output spatial and feature shape
        "output_spatial": spatial_dimension_last_encoder, # 2
        "output_feature": out_feature_last_encoder,

        # define kernel, bn and activation in the same manner
        "top_merge_kernel_shape": [(int(round(spatial_dimension_last_encoder))+1)*2,
                                   int(round(spatial_dimension_last_encoder))], #[6,2]
        "top_merge_kernel_regularization_function": encoder_pconv_kernel_regularization_function,
        "top_merge_kernel_l1_regularization_factor": encoder_pconv_kernel_l1_regularization_factor,
        "top_merge_kernel_l2_regularization_factor": encoder_pconv_kernel_l2_regularization_factor,

        "top_merge_pointwise_kernel_regularization_function": encoder_pconv_kernel_regularization_function,
        "top_merge_pointwise_kernel_l1_regularization_factor": encoder_pconv_kernel_l1_regularization_factor,
        "top_merge_pointwise_kernel_l2_regularization_factor": encoder_pconv_kernel_l2_regularization_factor,

        #"padding": 'valid',
        "use_bias": True,
        "batch_normalization": True, #False,
        "activation_function": decoder_activation[0], #None, # LeakyReLU
        "activation_parameter_dict": decoder_activation_kwarg_dict_list[0],
    
        # Squeeze and Excitation
        "SE_reduction_ratio": SE_reduction_ratio,
        ##
        "se_kernel_initializer_ec": SE_kernel_initializer_ec,
        "se_kernel_initializer_dc": SE_kernel_initializer_dc,
        "se_kernel_initializer": SE_kernel_initializer,
        ##
        "se_kernel_regularization_function": encoder_se_kernel_regularization_function,
        "se_kernel_l1_regularization_factor": encoder_se_kernel_l1_regularization_factor,
        "se_kernel_l2_regularization_factor": encoder_se_kernel_l2_regularization_factor
        # ,
    }

    return parameter_dict

