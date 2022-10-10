from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import cv2
import data.util as Util
import numpy as np
from data.transform import RandomCrop
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
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
        elif datatype == 'img':
            self.sr_path = Util.get_paths_from_images(
                '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            self.hr_path = Util.get_paths_from_images(
                '{}/hr_{}'.format(dataroot, r_resolution))
            if self.need_LR:
                self.lr_path = Util.get_paths_from_images(
                    '{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
        elif datatype == 'infer' or datatype == 'infer_to128'or datatype == 'infer_noise':
            self.sr_path = Util.get_paths_from_images(
                '{}/labels'.format(dataroot))
            self.hr_path = Util.get_paths_from_images(
                '{}/images'.format(dataroot))
            if self.need_LR:
                self.lr_path = Util.get_paths_from_images(
                    '{}/lr_save'.format(dataroot))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
        elif datatype == 'test_infer':
            self.sr_path = Util.get_paths_from_images(
                '{}/sr_save'.format(dataroot))
            self.hr_path = Util.get_paths_from_images(
                '{}/hr_save'.format(dataroot))
            if self.need_LR:
                self.lr_path = Util.get_paths_from_images(
                    '{}/lr_save'.format(dataroot))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
        elif datatype == 'random' or datatype == 'change' or datatype == 'crop' or datatype == 'multiple' or datatype == 'noise' or datatype == 'large_scale':
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
        elif datatype == 'segmentation':
            self.hr_path = Util.get_paths_from_images('{}/images')
            self.sr_path = Util.get_paths_from_images('{}/rgb_labels')
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
        background = None

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
            image_HR = Image.open(self.hr_path[index % self.dataset_len]).convert("RGB")
            #image_F = Image.open(self.hr_path[index % self.dataset_len]).convert("L")
            image_SR = Image.open(self.sr_path[index % self.dataset_len]).convert("RGB")
            H, W, C = np.shape(image_HR)
            if H > self.r_res + 10 and W > self.r_res + 10:
                start_x = np.random.randint(0, H - self.r_res)
                start_y = np.random.randint(0, W - self.r_res)
                box = (start_y, start_x, start_y + self.r_res, start_x + self.r_res)
                img_HR = image_HR.crop(box)
                #img_F = image_F.crop(box)
                img_SR = image_SR.crop(box)
                #background = background_compute(img_F)
                # print(np.max(img_SR),np.min(img_SR))

            else:
                img_HR = image_HR.resize((self.r_res, self.r_res))
                img_SR = image_SR.resize((self.r_res, self.r_res))
            #print(np.max(img_SR),np.min(img_SR))
            if self.need_LR:
                img_LR = Image.open(self.lr_path[index]).convert("RGB")

                # image_LR = cv2.imread(self.lr_path[index])
                # img_LR = cv2.cvtColor(image_LR,cv2.COLOR_BGR2RGB)
                # sample = {'SR': img_SR, 'HR': img_HR, 'LR': img_LR}
                # sample = self.randomcrop(sample)

                #img_LR = Image.open(self.lr_path[index]).convert
        elif self.datatype == 'infer_to128':
            image_HR = Image.open(self.hr_path[index % self.dataset_len]).convert("RGB")
            image_SR = Image.open(self.sr_path[index % self.dataset_len]).convert("RGB")
            img_HR = image_HR.resize((self.r_res, self.r_res))
            img_SR = image_SR.resize((self.r_res, self.r_res))
        elif self.datatype == 'infer_noise':
            image_HR = Image.open(self.hr_path[index % self.dataset_len]).convert("RGB")
            image_SR = Image.open(self.sr_path[index % self.dataset_len]).convert("RGB")
            img_HR = image_HR.resize((self.r_res, self.r_res))
            img_SR = image_SR.resize((self.r_res, self.r_res))
            if (np.max(img_SR) == 0):
                img_SR = Util.add_noise(img_SR)
        elif self.datatype == 'large_scale':
            image_HR = Image.open(self.hr_path[index % self.dataset_len]).convert("RGB")
            image_SR = Image.open(self.sr_path[index % self.dataset_len]).convert("RGB")
            H, W, C = np.shape(image_HR)
            if H > self.l_res + 10 and W > self.l_res + 10:
                start_x = np.random.randint(0, H - self.l_res)
                start_y = np.random.randint(0, W - self.l_res)
                box = (start_y, start_x, start_y + 512, start_x + 512)
                img_HR = image_HR.crop(box)
                img_SR = image_SR.crop(box)
                img_HR = img_HR.resize((self.r_res, self.r_res))
                img_SR = img_SR.resize((self.r_res, self.r_res))
                # print(np.max(img_SR),np.min(img_SR))

            else:
                img_HR = image_HR.resize((self.r_res, self.r_res))
                img_SR = image_SR.resize((self.r_res, self.r_res))
            # print(np.max(img_SR), np.min(img_SR))
            if self.need_LR:
                img_LR = Image.open(self.lr_path[index]).convert("RGB")

        elif self.datatype == 'crop':
            image_HR = Image.open(self.hr_path[index % self.dataset_len]).convert("RGB")
            image_SR = Image.open(self.sr_path[index % self.dataset_len]).convert("RGB")
            H, W, C = np.shape(image_HR)
            if H > 512 + 10 and W > 512 + 10:
                start_x = np.random.randint(0, H - 512)
                start_y = np.random.randint(0, W - 512)
                box = (start_y, start_x, start_y + 512, start_x + 512)
                img_HR = image_HR.crop(box)
                img_SR = image_SR.crop(box)
                img_HR = img_HR.resize((self.r_res, self.r_res))
                img_SR = img_SR.resize((self.r_res, self.r_res))
                # print(np.max(img_SR),np.min(img_SR))

            else:
                img_HR = image_HR.resize((self.r_res, self.r_res))
                img_SR = image_SR.resize((self.r_res, self.r_res))
            #print(np.max(img_SR), np.min(img_SR))
            if self.need_LR:
                img_LR = Image.open(self.lr_path[index]).convert("RGB")
        elif self.datatype == 'change':
            # image_HR = cv2.imread(self.hr_path[index])
            # #print(self.hr_path[index])
            # img_HR = cv2.cvtColor(image_HR,cv2.COLOR_BGR2RGB)
            # #print(self.hr_path[index],np.min(img_HR))
            # image_SR = cv2.imread(self.sr_path[index])
            # img_SR = cv2.cvtColor(image_SR,cv2.COLOR_BGR2RGB)
            # sample = {'SR': img_SR, 'HR': img_HR}
            # sample = self.randomcrop
            image_HR = Image.open(self.hr_path[index % self.dataset_len]).convert("RGB")
            image_SR = Image.open(self.sr_path[index % self.dataset_len]).convert("RGB")
            H, W, C = np.shape(image_HR)
            scale = random.randint(self.r_res/2,self.r_res*2)
            print("scale:",scale)
            if H > scale + 10 and W > scale + 10:
                start_x = np.random.randint(0, H - scale)
                start_y = np.random.randint(0, W - scale)
                box = (start_y, start_x, start_y + scale, start_x + scale)
                img_HR = image_HR.crop(box)
                img_SR = image_SR.crop(box)
                img_HR = img_HR.resize((self.r_res, self.r_res))
                img_SR = img_SR.resize((self.r_res, self.r_res))
                #print(np.max(img_SR),np.min(img_SR))

            else:
                img_HR = image_HR.resize((self.r_res, self.r_res))
                img_SR = image_SR.resize((self.r_res, self.r_res))
            if(np.max(img_SR) == 0):
                s = np.random.randint(0,10)
                if s>=5 :
                    value = np.random.randint(0, 20, size=[self.r_res, self.r_res, 3])
                    img_SR = value
                    print("random:",np.mean(img_SR))


            # img_SR = Image.open(self.sr_path[index]).convert("RGB")
            if self.need_LR:
                img_LR = Image.open(self.lr_path[index]).convert("RGB")
        elif self.datatype == 'noise':
            # image_HR = cv2.imread(self.hr_path[index])
            # #print(self.hr_path[index])
            # img_HR = cv2.cvtColor(image_HR,cv2.COLOR_BGR2RGB)
            # #print(self.hr_path[index],np.min(img_HR))
            # image_SR = cv2.imread(self.sr_path[index])
            # img_SR = cv2.cvtColor(image_SR,cv2.COLOR_BGR2RGB)
            # sample = {'SR': img_SR, 'HR': img_HR}
            # sample = self.randomcrop
            image_HR = Image.open(self.hr_path[index % self.dataset_len]).convert("RGB")
            image_SR = Image.open(self.sr_path[index % self.dataset_len]).convert("RGB")
            H, W, C = np.shape(image_HR)
            if H > self.r_res + 10 and W > self.r_res + 10:
                start_x = np.random.randint(0, H - self.r_res)
                start_y = np.random.randint(0, W - self.r_res)
                box = (start_y, start_x, start_y + self.r_res, start_x + self.r_res)
                img_HR = image_HR.crop(box)
                img_SR = image_SR.crop(box)
                #print(np.max(img_SR),np.min(img_SR))

            else:
                img_HR = image_HR.resize((self.r_res, self.r_res))
                img_SR = image_SR.resize((self.r_res, self.r_res))
            if(np.max(img_SR) == 0):
                img_SR = Util.add_noise(img_SR)
                # print(img_SR.dtype)
                # Hr_dtpye = np.array(img_HR)
                # print(Hr_dtpye.dtype)
            # img_SR = Image.open(self.sr_path[index]).convert("RGB")
            if self.need_LR:
                img_LR = Image.open(self.lr_path[index]).convert("RGB")
        elif self.datatype == 'multiple':
            # image_HR = cv2.imread(self.hr_path[index])
            # #print(self.hr_path[index])
            # img_HR = cv2.cvtColor(image_HR,cv2.COLOR_BGR2RGB)
            # #print(self.hr_path[index],np.min(img_HR))
            # image_SR = cv2.imread(self.sr_path[index])
            # img_SR = cv2.cvtColor(image_SR,cv2.COLOR_BGR2RGB)
            # sample = {'SR': img_SR, 'HR': img_HR}
            # sample = self.randomcrop
            image_HR = Image.open(self.hr_path[index % self.dataset_len]).convert("RGB")
            image_SR = Image.open(self.sr_path[index % self.dataset_len]).convert("RGB")
            H, W, C = np.shape(image_HR)
            scale = random.randint(self.r_res/2,self.r_res*2)
            print("scale:",scale)
            if H > scale + 10 and W > scale + 10:
                start_x = np.random.randint(0, H - scale)
                start_y = np.random.randint(0, W - scale)
                box = (start_y, start_x, start_y + scale, start_x + scale)
                img_HR = image_HR.crop(box)
                img_SR = image_SR.crop(box)
                img_HR = img_HR.resize((self.r_res, self.r_res))
                img_SR = img_SR.resize((self.r_res, self.r_res))
                #print(np.max(img_SR),np.min(img_SR))

            else:
                img_HR = image_HR.resize((self.r_res, self.r_res))
                img_SR = image_SR.resize((self.r_res, self.r_res))
            # img_SR = Image.open(self.sr_path[index]).convert("RGB")
            if self.need_LR:
                img_LR = Image.open(self.lr_path[index]).convert("RGB")
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
            # if not background:
            #     img_SR = np.concatenate(img_SR,background)
            #print(sample['HR'])
            return {'HR': img_HR, 'SR': img_SR, 'Index': index}
