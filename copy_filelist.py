import os
import shutil

folder_in= '/mnt/data/ShapeNet.ConvONet'
folder_out = '/mnt/data/ShapeNet.ConvONet.filelist'

folders = os.listdir(folder_in)
for folder in folders:
  curr_folder = os.path.join(folder_in, folder)
  des_folder = os.path.join(folder_out, folder)
  if os.path.isdir(curr_folder):
    if not os.path.exists(des_folder):
      os.makedirs(des_folder)

    for filename in ['train.lst', 'test.lst', 'val.lst']:
      curr_file = os.path.join(curr_folder, filename)
      des_file = os.path.join(des_folder, filename)
      shutil.copy(curr_file, des_file)
