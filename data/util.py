import os
import torch
import torchvision
import random
import numpy as np
from PIL import Image
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.tif']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)
def add_noise(img,mean =0 ,var = 2):
    img = np.array(img)
    shape = img.shape
    image = np.ones(shape,dtype = np.uint8)
    image = image*10
    noise = np.random.normal(mean,var,shape)
    out = image + noise
    print("addNoise:",np.max(out),np.min(out))
    return out

def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img

def image_process(path,out_path):
    img = Image.open(path).convert("RGB")
    H, W, C = np.shape(img)
    while (H>400 or W>400):
        H = H//2
        W = W//2
        #print(H,W)
    p1 = H//32
    p2 = W//32
    H = 32*(p1+1)
    W = 32*(p2+1)
    img = img.resize((H,W),Image.ANTIALIAS)
    print(np.shape(img))
    img.save(out_path)

def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 0, 1)))).float()
    # to range min_max
    img = img*(min_max[1] - min_max[0]) + min_max[0]
    return img


# implementation by numpy and torch
# def transform_augment(img_list, split='val', min_max=(0, 1)):
#     imgs = [transform2numpy(img) for img in img_list]
#     imgs = augment(imgs, split=split)
#     ret_img = [transform2tensor(img, min_max) for img in imgs]
#     return ret_img


# implementation by torchvision, detail in https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/issues/14
totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()
def transform_augment(img_list, split='val', min_max=(0, 1)):
    #print(np.min(img_list[0]), np.max(img_list[0]))
    imgs = [totensor(img) for img in img_list]
    #print(torch.min(imgs[0]), torch.max(imgs[0]))
    if split == 'train':
        imgs = torch.stack(imgs, 0)
        #print(imgs.shape)
        #imgs = torchvision.transforms.ToPILImage(imgs)
        #print(imgs)
        imgs = hflip(imgs)
        imgs = torch.unbind(imgs, dim=0)
    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
    return ret_img
