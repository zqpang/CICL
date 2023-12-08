from __future__ import absolute_import
import pdb
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image
import torch
from skimage.io import imsave

from torchvision import transforms


a = [2,3,4,5,6,7,10,11]


def numpy_trans(img):
    img = img.detach().numpy()
    img = img.transpose(1,2,0)
    return img



def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img



def read_parsing_result(img_path):
    """Keep reading image until succeed.
        This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def white(img, parsing_result):
    
    parsing_result = parsing_result.unsqueeze(0).repeat(3, 1, 1)
    img[(parsing_result == 0)] = 255

    return img



def random_col(img, parsing_result):
    
    img = np.array(img, dtype=np.uint8).transpose(2, 0, 1)
    
    parsing_result = torch.tensor(np.asarray(parsing_result, dtype=np.uint8))
    
    
    for i in a:
        img[0][(parsing_result == i)] = random.random()*96
        img[1][(parsing_result == i)] = random.random()*96
        img[2][(parsing_result == i)] = random.random()*96
    
    img = white(img, parsing_result)
    
    return np.array(img).transpose(1, 2, 0)





def random_erasing(img, parsing_result):
    
    img = np.array(img, dtype=np.uint8).transpose(2, 0, 1)
    
    parsing_result = torch.tensor(np.asarray(parsing_result, dtype=np.uint8))

    for i in a:
        channel = random.randint(0,2)
        img[channel][(parsing_result == i)] = 0
    
    img = white(img, parsing_result)
    
    return np.array(img).transpose(1, 2, 0)



def random_aug(img, parsing_result):
    
    img = np.array(img, dtype=np.uint8).transpose(2, 0, 1)
    
    parsing_result = torch.tensor(np.asarray(parsing_result, dtype=np.uint8))
    
    
    for i in a:
        index = np.random.permutation(3)
        img[index[0]][(parsing_result == i)] = img[index[1]][(parsing_result == i)]
        img[index[1]][(parsing_result == i)] = img[index[2]][(parsing_result == i)]
        
        img[index[2]][(parsing_result == i)] = random.random()*96
        
    img = white(img, parsing_result)
    
    return np.array(img).transpose(1, 2, 0)




def black(img, parsing_result):
    
    img = np.array(img, dtype=np.uint8).transpose(2, 0, 1)
      
    
    parsing_result = torch.tensor(np.asarray(parsing_result, dtype=np.uint8)).repeat(3, 1, 1)
    
    for i in a:
        img[(parsing_result == i)] = 0
    
    img[(parsing_result == 0)] = 255
    
    return img.transpose(1, 2, 0)



class Preprocessor(Dataset):
    def __init__(self, dataset, train = True, root=None, transform1=None, transform2=None, black=False):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform1 = transform1
        self.transform2 = transform2
        self.train = train
        self.black = black
        
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, clothesid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = read_image(fpath)
        
        if self.train:
            fname_mask = fname.split('/')
            if 'ltcc' in fname_mask:
                mask_path = '/home/zhiqi/dataset/ltcc/parsing'
                fname_mask = osp.join(mask_path, fname_mask[-1])
                img_mask = read_parsing_result(fname_mask)
                
            if 'celebreidlight' in fname_mask:
                mask_path = '/home/zhiqi/datasets/celebreidlight/parsing'
                fname_mask = osp.join(mask_path, fname_mask[-1][:-3] + 'png')
                img_mask = read_parsing_result(fname_mask)
            elif 'celebreidlight' not in fname_mask and 'celebreid' in fname_mask:
                mask_path = '/home/zhiqi/datasets/celebreid/parsing'
                fname_mask = osp.join(mask_path, fname_mask[-1][:-3] + 'png')
                img_mask = read_parsing_result(fname_mask)
                
            if 'prcc' in fname_mask:
                mask_path = '/home/zhiqi/dataset/prcc/parsing'
                fname_mask = osp.join(mask_path, fname_mask[-2],fname_mask[-1][:-3] + 'png')
                img_mask = read_parsing_result(fname_mask)
                
            if 'VC_Clothes' in fname_mask:
                mask_path = '/home/zhiqi/dataset/VC_Clothes/parsing'
                fname_mask = osp.join(mask_path,fname_mask[-1][:-3] + 'png')
                img_mask = read_parsing_result(fname_mask)
        
        
        if self.transform1 is not None:
            img1 = self.transform1(img)
        
        if self.train:
            
            img_rand = random_col(img, img_mask)

            img_rand = Image.fromarray(img_rand)
            img_rand = self.transform1(img_rand)
            
            img_black = black(img, img_mask)
            
            img_black = Image.fromarray(img_black)
            img_black = self.transform1(img_black)
            
                                  
            
        else:
            img_rand = img1
            img_black = img1
            
        return img1, img_rand, img_black, fname, pid, clothesid, camid, index
        
