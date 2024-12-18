from unimatch.dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2


class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=256, id_path=None, nsample=None, rt_mask=False):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.rt_mask = rt_mask

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open('splits/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item): 
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        if self.name == "Polyp":
            with open(os.path.join(self.root, id.split(' ')[1]), 'rb') as f:
                mask = Image.open(f)
                mask = mask.convert('1')
                mask = Image.fromarray(np.array(mask, dtype=np.uint8))
        else:
            mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))

        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask, id
        
        img, mask = resize(img, mask, (0.5, 2))
        ignore_value = 254 if self.mode == 'train_u' else 255
        # ignore_value = 0
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        img_w, img_s = deepcopy(img), deepcopy(img)

        if self.mode == 'train_l':
            return normalize(img, mask)


        if random.random() < 0.8:
            img_s = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s)
        img_s = transforms.RandomGrayscale(p=0.2)(img_s)
        img_s = blur(img_s, p=0.5)
        cutmix_box = obtain_cutmix_box(img_s.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s, ignore_mask = normalize(img_s, ignore_mask)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255
        
        if not self.rt_mask:
            return normalize(img_w), img_s, ignore_mask, cutmix_box
        else:
            return normalize(img_w), img_s, ignore_mask, cutmix_box, mask

    def __len__(self):
        return len(self.ids)
