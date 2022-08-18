import torch
import cv2
from PIL import Image
import util as Util
import numpy as np
from transform import RandomCrop
import os
#import core.metrics as Metrics
def test():
    randomcrop = RandomCrop(256)
    hr_path = "/data/diffusion_data/dataset/false_generate/hr_256/P0406.png"
    sr_path = "/data/diffusion_data/dataset/false_generate/sr_32_256/P0406_instance_color_RGB.png"
    image_HR = cv2.imread(hr_path)
    # print(self.hr_path[index])
    img_HR = cv2.cvtColor(image_HR, cv2.COLOR_BGR2RGB)
    image_SR = cv2.imread(sr_path)
    img_SR = cv2.cvtColor(image_SR, cv2.COLOR_BGR2RGB)
    sample = {'SR': img_SR, 'HR': img_HR}
    sample = randomcrop(sample)
    img1 = Image.fromarray(sample['HR'],"RGB")
    img2 = Image.fromarray(sample['HR'], "RGB")
    dir = "/data/diffusion_data/dataset/test"
    os.makedirs(dir,exist_ok=True)
    path1 = os.path.join(dir,"P0406.png")
    path2 = os.path.join(dir, "P0406_instance_color_RGB.png")
    img1.save(path1)
    img2.save(path2)

    [sample['SR'], sample['HR']] = Util.transform_augment(
        [sample['SR'], sample['HR']], split='train', min_max=(-1, 1))
    # img_SR = Image.open(self.sr_path[index]).convert("RGB")
    #hr_img = Metrics.tensor2img(sample['HR'])  # uint8
    #lr_img = Metrics.tensor2img(sample['HR'])  # uint8
if __name__ == "__main__":
    test()
