import random

import numpy as np


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        #print(type(sample['HR']))
        h, w = sample['HR'].shape[:2]
        ch = min(h, self.output_size[0])
        cw = min(w, self.output_size[1])

        h_space = h - self.output_size[0]
        w_space = w - self.output_size[1]

        if w_space > 0:
            cont_left = 0
            img_left = random.randrange(w_space + 1)
        else:
            cont_left = random.randrange(-w_space + 1)
            img_left = 0

        if h_space > 0:
            cont_top = 0
            img_top = random.randrange(h_space + 1)
        else:
            cont_top = random.randrange(-h_space + 1)
            img_top = 0

        key_list = sample.keys()
        for key in key_list:
            img = sample[key]
            img_crop = np.zeros((self.output_size[0], self.output_size[1], 3), np.float32)
            img_crop[cont_top:cont_top + ch, cont_left:cont_left + cw] = \
                img[img_top:img_top + ch, img_left:img_left + cw]
            print("key:"+key)
            sample[key] = img_crop

        return sample