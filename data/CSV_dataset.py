import pandas as pd                         #导入pandas包
import os
import data.util as Util
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as transforms
import math
def csv_read(path,dataroot = ""):
    df = pd.DataFrame(pd.read_csv(path,header=0))           	#读取csv文件
    print(df.shape)
    file_list = df['file_name']
    path_list = []
    for file in file_list:
        file_path = os.path.join(dataroot,file)
        path_list.append(file_path)
    hour_list = df['hour']
    minute_list = df['minute']
    bbox_list = df['bbox_det']
    collection_list = (hour_list-1)*60 + minute_list

    print(path_list[0],hour_list[0],minute_list[0],bbox_list[0],collection_list[0])
    #print(df.columns)
    #print(df.head())
    #print(data)
def concat_csv(path1,path2,newpath):
    data1 = pd.read_csv(path1,header=0)
    data2 = pd.read_csv(path2,header=0)
    data1.to_csv(newpath, encoding="utf_8_sig", index=False, header=False, mode='a+')
    data2.to_csv(newpath, encoding="utf_8_sig", index=False, header=False, mode='a+')
def FFT_value(A = 1,B = 1,t = 0):
    value = A*math.cos(0.5*t) + B*math.cos(2*t)
    return value

transforms_ = [
        #transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
class CSVDataset(Dataset):
    def __init__(self,path,dataroot, r_resolution=128,data_len = -1):
        df = pd.DataFrame(pd.read_csv(path, header=0))
        self.file_list = df['file_name']
        self.r_res = r_resolution
        self.path_list = []
        self.data_len = data_len
        for file in self.file_list:
            file_path = os.path.join(dataroot, file)
            self.path_list.append(file_path)
        self.hour_list = df['hour']
        self.minute_list = df['minute']
        self.bbox_list = df['bbox_det']
        self.dataset_len = len(self.file_list)
        #self.collection_list = (self.hour_list - 1) * 60 + self.minute_list
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)
        self.transform = transforms.Compose(transforms_)

    def __getitem__(self, index):
        img = Image.open(self.path_list[index % self.dataset_len]).convert('RGB')
        img_HR = img.resize((self.r_res, self.r_res))
        hour = self.hour_list[index]
        minute = self.minute_list[index]
        #value_list = torch.tensor([hour,minute])
        value_list = [hour, minute]
        item_image = self.transform(img_HR)

        return {'HR': item_image, 'clock_data': value_list, 'Index': index}

    def __len__(self):
        return self.data_len
path = '/data/diffusion_data/dataset/all_final.csv'
# path2 = 'D:\workapp\download\coco\coco_final.csv'
# newpath = 'D:\workapp\download\all_final.csv'
dataroot = '/data/diffusion_data/dataset/clock_dataset'
