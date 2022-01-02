from skimage.util.dtype import img_as_bool
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn, optim
from torchvision import transforms
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import numpy as np
from meaning import *
import random

SIZE = 256
class ColorizationDataset(Dataset):
    def __init__(self, paths, split='train'):
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((SIZE, SIZE),  Image.BICUBIC),
                transforms.RandomHorizontalFlip(), # A little data augmentation!
            ])
        elif split == 'val':
            self.transforms = transforms.Resize((SIZE, SIZE),  Image.BICUBIC)

        self.max, self.min = self.__norm_param__()
        self.split = split
        self.size = SIZE
        self.paths = paths
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32") # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1. # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110. # Between -1 and 1
        num = random.randint(10, 21)
        if(num < 21):
            mean_ab = meaning(ab, num)
        else:
            mean_ab = torch.zeros_like(ab)
        
        
        return {'L': L, 'ab': ab, 'mask' : mean_ab}
    
    def __len__(self):
        return len(self.paths)

    def __norm_param__(self):
        img = np.mgrid[0:256, 0:256, 0:256].astype(np.float32)
        img /= 255
        min_param = dict()
        max_param = dict()

        img_lab = rgb2lab(img.T)

        img_L = img_lab[:,:,:,0]
        img_a = img_lab[:,:,:,1]
        img_b = img_lab[:,:,:,2]

        min_param['L'] = np.min(img_L)
        min_param['a'] = np.min(img_a)
        min_param['b'] = np.min(img_b)

        max_param['L'] = np.max(img_L)
        max_param['a'] = np.max(img_a)
        max_param['b'] = np.max(img_b)

        return max_param,  min_param

def make_dataloaders(batch_size=4, n_workers=4, pin_memory=True, **kwargs): # A handy function to make our dataloaders
    dataset = ColorizationDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader