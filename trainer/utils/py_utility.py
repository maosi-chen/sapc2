import os

# check and create the path if not exists
def check_and_create_file_path(in_path, in_Path_desc_str=None):
  if (in_Path_desc_str is None):
    Path_desc_str = ""
  else:
    Path_desc_str = in_Path_desc_str
  try:
    os.stat(in_path)
    print("{} Path [{}] exists.".format(Path_desc_str, in_path))
  except:
    os.makedirs(in_path)
    print("{} Path [{}] was created.".format(Path_desc_str, in_path))

