import os, sys
import argparse
import json
import numpy as np
import copy
from collections import deque

def get_args():
  """Argument parser.
  Returns:
    Dictionary of arguments.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--parameterDictionary', '-paramdict',
    type=json.loads,
    required=True,
    action='append',
    help='a json string containing parameterName, interpretationMethod, and valueArray')
  parser.add_argument(
    '--export_json_FWN', '-ejsonfwn',
    default='~/tmp/tmp_hp_json.txt',
    type=str,
    help='full path+name of the exported json text file.')
  
  ret_args, _ = parser.parse_known_args()
  ret_args = vars(ret_args)
  print("ret_args", ret_args)
  for ikey, ivalue in ret_args.items():
    if ivalue is not None:
      print(ikey, ivalue)
  return ret_args

def get_hyperparameter_names(in_dict):
  hp_dict_arr = in_dict['parameterDictionary']
  #num_hps = len(hp_dict_arr)
  hp_dicts = {}
  hp_names = []
  for i_hp_dict in hp_dict_arr:
    # {"parameterName":"INITIAL_LEARNING_RATE",
    #  "interpretationMethod":"DISCRETE,DEPENDENT,LEARNING_RATE",
    #  "valueArray":[3.0,4.0]}
    parameterName_i = i_hp_dict["parameterName"]
    hp_names.append(parameterName_i)
    hp_dicts[parameterName_i]=copy.deepcopy(i_hp_dict)
    
  print("hp_names: ", hp_names)
  return hp_names, hp_dicts
  
def parse_one_hyperparameter_dict(tgt_hp_dict):
  tgt_interpretationMethod_arr = tgt_hp_dict["interpretationMethod"].split(',')
  tgt_valueArray = tgt_hp_dict["valueArray"]
  if (tgt_interpretationMethod_arr[0] == 'UNIT_LINEAR_SCALE'):
    tgt_valueArray_full = np.arange(tgt_valueArray[0], tgt_valueArray[2]+tgt_valueArray[1], tgt_valueArray[1])
    return tgt_valueArray_full
  elif (tgt_interpretationMethod_arr[0] == 'DISCRETE'):
    tgt_valueArray_full = tgt_valueArray
    return tgt_valueArray_full
  else:
    raise ValueError("unsupported interpretationMethod {}".format(tgt_interpretationMethod_arr[0]))
  
def generate_hyperparameter_array(hp_dicts, tgt_hp_name):
  # {'parameterName': 'LEARNING_RATE',
  #  'interpretationMethod': 'UNIT_LINEAR_SCALE',
  #  'valueArray': [0.00018, 6e-05, 0.00036]}
  # {"parameterName":"INITIAL_LEARNING_RATE",
  #  "interpretationMethod":"DISCRETE,DEPENDENT,LEARNING_RATE",
  #  "valueArray":[3.0,4.0]}
  tgt_hp_dict = hp_dicts[tgt_hp_name]
  tgt_interpretationMethod_arr = tgt_hp_dict["interpretationMethod"].split(',')
  tgt_valueArray = tgt_hp_dict["valueArray"]

  tgt_valueArray_full = parse_one_hyperparameter_dict(tgt_hp_dict)
  
  return tgt_valueArray_full
  
def is_dependent_hp(hp_dicts, tgt_hp_name):
  tgt_hp_dict = hp_dicts[tgt_hp_name]
  tgt_interpretationMethod_arr = tgt_hp_dict["interpretationMethod"].split(',')
  if (len(tgt_interpretationMethod_arr) == 1):
    return False
  elif (tgt_interpretationMethod_arr[1] == 'DEPENDENT'):
    return True
  else:
    raise ValueError("Error: 2nd element in tgt_interpretationMethod_arr is not 'DEPENDENT'")
  
  
def create_nested_hp_loops(hp_dicts_wFullArr, ll_hp_names, prev_hp_name=None, prev_hp_value=None):
  if (len(ll_hp_names) == 0):
    return []
  cur_hp_name = ll_hp_names.popleft()
  cur_valueArray_full = hp_dicts_wFullArr[cur_hp_name]['valueArray_full']

  if hp_dicts_wFullArr[cur_hp_name]['is_dependent_hp']:
    if (cur_hp_name == 'INITIAL_LEARNING_RATE'):
      cur_valueArray_full = [prev_hp_value / float(iCurVal) for iCurVal in cur_valueArray_full]
    else:
      raise ValueError("unsupported dependent hp.")
  
  ret_str_arr = []
  for iVal in cur_valueArray_full:
    iValStr = "\"{}\":{:.8e}".format(cur_hp_name, iVal)
    if (len(ll_hp_names)>0):
      ll_hp_names_cpy = copy.deepcopy(ll_hp_names)
      nextLvl_str_arr = create_nested_hp_loops(
        hp_dicts_wFullArr, ll_hp_names_cpy, prev_hp_name=cur_hp_name, prev_hp_value=iVal)
      for jnextLvl_str in nextLvl_str_arr:
        ijValStr = "{},{}".format(iValStr, jnextLvl_str)
        ret_str_arr.append(ijValStr)
    else:
      ret_str_arr.append(iValStr)
  
  return ret_str_arr

# if __name__ == "__main__":
#
#   args = get_args()
#   export_json_FWN = args['export_json_FWN']
#   hp_names, hp_dicts=get_hyperparameter_names(args)
#
#   hp_dicts_wFullArr = copy.deepcopy(hp_dicts)
#
#   ll_hp_names = deque()
#   ll_DEP_hp_names = deque()
#   for ii, i_hp_name in enumerate(hp_names):
#     i_tgt_valueArray_full = generate_hyperparameter_array(hp_dicts_wFullArr, i_hp_name)
#     i_is_dependent_hp = is_dependent_hp(hp_dicts_wFullArr, i_hp_name)
#     hp_dicts_wFullArr[i_hp_name]['valueArray_full'] = i_tgt_valueArray_full
#     hp_dicts_wFullArr[i_hp_name]['is_dependent_hp'] = i_is_dependent_hp
#
#     if (i_is_dependent_hp == False):
#       ll_hp_names.append(i_hp_name)
#     else:
#       ll_DEP_hp_names.append(i_hp_name)
#
#   if (len(ll_DEP_hp_names) > 0):
#     for iDEPhp in ll_DEP_hp_names:
#       i_DEPhp_dict = hp_dicts_wFullArr[iDEPhp]
#       i_DEPhp_interpretationMethod_arr = i_DEPhp_dict["interpretationMethod"].split(',')
#       i_DEPhp_dependent_hp_name = i_DEPhp_interpretationMethod_arr[2]
#       ll_hp_names.insert(ll_hp_names.index(i_DEPhp_dependent_hp_name) + 1, iDEPhp)
#
#   raw_dict_str_arr = create_nested_hp_loops(hp_dicts_wFullArr, ll_hp_names, prev_hp_name=None, prev_hp_value=None)
#   if (len(raw_dict_str_arr) > 0):
#     dict_str_arr = ["{{{}}}".format(iStr) for iStr in raw_dict_str_arr]
#     with open(export_json_FWN, 'w') as FH1:
#       for iStr in dict_str_arr:
#         FH1.write(iStr + "\n")
#
#   else:
#     dict_str_arr = []
#
#   print('end of program')
#
  