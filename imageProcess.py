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
def reszie_process(files,out_path):
    os.makedirs(out_path, exist_ok=True)
    for file in files:
        file_name = os.path.basename(file)
        final_path = os.path.join(out_path,file_name)
        img = Image.open(file).convert("RGB")
        img_new = img.resize((256, 256), Image.ANTIALIAS)
        print(np.shape(img_new))
        img_new.save(final_path)



label_path = '/data/diffusion_data/infer/false_256_220830_020020/results/sr_save'
image_path = '/data/diffusion_data/infer/false_256_220830_020020/results/hr_save'
out_lable = '/data/diffusion_data/infer/infer_256/sr_save'
out_image = '/data/diffusion_data/infer/infer_256/hr_save'
sr_path = Util.get_paths_from_images(label_path)
hr_path = Util.get_paths_from_images(image_path)
# main_process(sr_path,out_lable)
# main_process(hr_path,out_image)
reszie_process(sr_path,out_lable)
reszie_process(hr_path,out_image)