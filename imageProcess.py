
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
def val_generate(files,out_path):
    os.makedirs(out_path, exist_ok=True)
    count = 0
    for file in files:
        file_name = os.path.basename(file)
        suffix = file_name.split('.')[1]
        img = Image.open(file).convert("RGB")
        H, W, C = np.shape(img)
        if(H>W):
            p1 = W//512 + 1
            W_new = int(W/p1)
            for i in range(1,p1):
                 box1 = (0, (i-1)*W_new, H, i*W_new)
                 img_first = img.crop(box1)
                 p2 = H//W_new + 1
                 H_new = int(H/p2)
                 for j in range(1, p2):
                     box2 = ((j-1)*H_new, 0, j*H_new,  W_new)
                     img_second = img_first.crop(box2)
                     print(np.shape(img_second))
                     img_new = img_second.resize((256, 256), Image.ANTIALIAS)
                     count = count + 1
                     save_path = os.path.join(out_path,str(count)+'.'+suffix)
                     img_new.save(save_path)
        else:
            p1 = H // 512 + 1
            H_new = int(H / p1)
            p2 = W // H_new + 1
            W_new = int(W / p2)
            for i in range(1, p1):
                 box1 = ((i - 1) * H_new,0 ,i * H_new,W)
                 img_first = img.crop(box1)
                 for j in range(1, p2):
                     box2 = (0, (j - 1) * W_new, H_new, j * W_new)
                     img_second = img_first.crop(box2)
                     print(np.shape(img_second))
                     img_new = img_second.resize((256, 256), Image.ANTIALIAS)
                     count = count + 1
                     save_path = os.path.join(out_path, str(count) + '.' + suffix)
                     img_new.save(save_path)
    print("End Running")
def remove_file(image_path):
    files = Util.get_paths_from_images(image_path)
    for file in files:
        img = Image.open(file).convert("RGB")
        if(np.sum(img)<10000):
            os.remove(file)
            print(file)
def remove_images(image_path):
    files = Util.get_paths_from_images(image_path)
    txt_file = "/data/diffusion_data/val/dataset/val.lst"
    with open(txt_file) as file_object:
        list = file_object.read().splitlines()
    print(list)
    for file in files:
        file_name = os.path.basename(file)
        name = file_name.split('.')[0]
        #print(name)
        if name not in list:
            os.remove(file)
            print(file)



label_path = '/data/diffusion_data/val/images'
image_path = '/data/diffusion_data/val/labels'
list_path = "/data/diffusion_data/val/val.lst"
# label_path = '/data/diffusion_data/infer/false_256_220830_020020/results/sr_save'
# image_path = '/data/diffusion_data/infer/false_256_220830_020020/results/hr_save'
out_lable = '/data/diffusion_data/val/test/labels'
out_image = '/data/diffusion_data/val/test/images'
# sr_path = Util.get_paths_from_images(label_path)
# hr_path = Util.get_paths_from_images
img_ids = [i_id.strip() for i_id in open(list_path)]
files= []
for name in img_ids:
    img_file = os.path.join(image_path, "%s.png" % name)
    label_file = os.path.join(label_path, "%s._instance_color_RGB.png" % name)
    files.append({
        "img": img_file,
        "label": label_file,
        "name": name
    })
# main_process(sr_path,out_lable)
# main_process(hr_path,out_image)
# reszie_process(sr_path,out_lable)
# reszie_process(hr_path,out_image)
val_generate(files["label"],out_lable)
val_generate(files["img"],out_image)
# remove_file(out_image)
#remove_images(out_lable)