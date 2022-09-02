import os
from pathlib import Path

file_dir = '/data/diffusion_data/val/dataset/labels'
def getFlist(path):
    for root, dirs, files in os.walk(file_dir):
        print('root_dir:', root)  
        print('sub_dirs:', dirs)   
        print('files:', files)     
    return files
file_names = getFlist(file_dir)

path = '/data/diffusion_data/val/dataset'
dirs = Path(path)
if not dirs.is_dir():
    os.makedirs(path)

file_path = path + '/' + 'val.lst'
file = open(file_path, 'w')
try:

    for file_name in file_names:
        file_name = file_name.split('.')[0]
        print(file_name)
        file.writelines(file_name+'\n')
except IOError:
    print('error')
file.close()

