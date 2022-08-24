from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import cv2
import data.util as Util
import numpy as np
from data.transform import RandomCrop

class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        self.randomcrop = RandomCrop(r_resolution)

        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot, readonly=True, lock=False,
                                 readahead=False, meminit=False)
            # init the datalen
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'img' or datatype == 'random':
            # self.sr_path = Util.get_paths_from_images(
            #     '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            # self.hr_path = Util.get_paths_from_images(
            #     '{}/hr_{}'.format(dataroot, r_resolution))
            self.hr_path = Util.get_paths_from_images('{}/hr_256'.format(dataroot))
            self.sr_path = Util.get_paths_from_images('{}/sr_32_256'.format(dataroot))
            if self.need_LR:
                self.lr_path = Util.get_paths_from_images(
                    '{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        if self.datatype == 'lmdb':
            with self.env.begin(write=False) as txn:
                hr_img_bytes = txn.get(
                    'hr_{}_{}'.format(
                        self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                sr_img_bytes = txn.get(
                    'sr_{}_{}_{}'.format(
                        self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                if self.need_LR:
                    lr_img_bytes = txn.get(
                        'lr_{}_{}'.format(
                            self.l_res, str(index).zfill(5)).encode('utf-8')
                    )
                # skip the invalid index
                while (hr_img_bytes is None) or (sr_img_bytes is None):
                    new_index = random.randint(0, self.data_len-1)
                    hr_img_bytes = txn.get(
                        'hr_{}_{}'.format(
                            self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    sr_img_bytes = txn.get(
                        'sr_{}_{}_{}'.format(
                            self.l_res, self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    if self.need_LR:
                        lr_img_bytes = txn.get(
                            'lr_{}_{}'.format(
                                self.l_res, str(new_index).zfill(5)).encode('utf-8')
                        )
                img_HR = Image.open(BytesIO(hr_img_bytes)).convert("RGB")
                img_SR = Image.open(BytesIO(sr_img_bytes)).convert("RGB")
                if self.need_LR:
                    img_LR = Image.open(BytesIO(lr_img_bytes)).convert("RGB")
        elif self.datatype == 'random':
            # image_HR = cv2.imread(self.hr_path[index])
            # #print(self.hr_path[index])
            # img_HR = cv2.cvtColor(image_HR,cv2.COLOR_BGR2RGB)
            # #print(self.hr_path[index],np.min(img_HR))
            # image_SR = cv2.imread(self.sr_path[index])
            # img_SR = cv2.cvtColor(image_SR,cv2.COLOR_BGR2RGB)
            # sample = {'SR': img_SR, 'HR': img_HR}
            # sample = self.randomcrop
            image_HR = Image.open(self.hr_path[index]).convert("RGB")
            image_SR = Image.open(self.sr_path[index]).convert("RGB")
            H, W, C = np.shape(image_HR)
            if H > self.r_res + 10 and W > self.r_res +10:
                start_x = np.random.randint(0, H-self.r_res)
                start_y = np.random.randint(0, W-self.r_res)
                box = (start_y, start_x, start_y + self.r_res, start_x + self.r_res)
                img_HR = image_HR.crop(box)
                img_SR = image_SR.crop(box)
                #img_HR = image_HR.resize((self.r_res, self.r_res))
                #img_SR = image_SR.resize((self.r_res, self.r_res))
            else:
                img_HR = image_HR.resize((self.r_res, self.r_res))
                img_SR = image_SR.resize((self.r_res, self.r_res))
            #img_SR = Image.open(self.sr_path[index]).convert("RGB")
            if self.need_LR:
                img_LR = Image.open(self.lr_path[index]).convert("RGB")

                # image_LR = cv2.imread(self.lr_path[index])
                # img_LR = cv2.cvtColor(image_LR,cv2.COLOR_BGR2RGB)
                # sample = {'SR': img_SR, 'HR': img_HR, 'LR': img_LR}
                # sample = self.randomcrop(sample)

                #img_LR = Image.open(self.lr_path[index]).convert("RGB")
        else:
            img_HR = Image.open(self.hr_path[index]).convert("RGB")
            img_SR = Image.open(self.sr_path[index]).convert("RGB")
            #sample = {'SR': img_SR, 'HR': img_HR}
            #sample = self.randomcrop(sample)
            if self.need_LR:
                img_LR = Image.open(self.lr_path[index]).convert("RGB")
                #sample = {'SR': img_SR, 'HR': img_HR, 'LR': img_LR}
                #img_LR = Image.open(self.lr_path[index]).convert("RGB")
        if self.need_LR:
            [img_HR, img_SR, img_LR] = Util.transform_augment(
                [img_HR, img_SR, img_LR], split=self.split, min_max=(-1, 1))
            return {'LR': img_LR, 'HR':img_HR, 'SR': img_SR, 'Index': index}
        else:
            [img_HR,img_SR] = Util.transform_augment(
                [img_HR, img_SR], split=self.split, min_max=(-1, 1))
            #print(sample['HR'])
            print(img_HR.shape)
            return {'HR': img_HR, 'SR': img_SR, 'Index': index}
