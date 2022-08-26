import random
import os
import numpy as np
import data.util as Util
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ISImageDataset(Dataset):
    def __init__(self, root, transforms_=None):
        self.transform = transforms.Compose(transforms_)

        self.files = Util.get_paths_from_images(root)

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)]).convert('RGB')
        item_image = self.transform(img)
        return item_image

    def __len__(self):
        return len(self.files)
