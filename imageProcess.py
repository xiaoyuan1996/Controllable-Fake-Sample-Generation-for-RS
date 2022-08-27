from PIL import Image
import os
import random
import cv2
import data.util as Util
import numpy as np
def main_process(files,out_path):
    os.makedirs(out_path, exist_ok=True)
    for file in files:
        file_name = os.path.basename(file)
        final_path = os.path.join(out_path,file_name)
        Util.image_process(file,final_path)


label_path = '/data/diffusion_data/val/labels'
image_path = '/data/diffusion_data/val/images'
out_lable = '/data/diffusion_data/val/processed/labels'
out_image = '/data/diffusion_data/val/processed/images'
sr_path = Util.get_paths_from_images(label_path)
hr_path = Util.get_paths_from_images(image_path)
main_process(sr_path,out_lable)
main_process(hr_path,out_image)