# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:29:03 2022

@author: maijieai
"""
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import os
import glob
import h5py

class srDataset(Dataset):
    def __init__(self, lr_dir,hr_dir, mode='train', transform=None):
        super(srDataset, self).__init__()
        self.mode = mode
        self.transform = transform
        self.lr_path, self.hr_path = self.get_pathlist (lr_dir,hr_dir)
            
    
    def get_pathlist ( self, lr1,hr1):
        file_lr=h5py.File(lr1,'r')
        lr_path = file_lr['image']
        file_hr=h5py.File(hr1,'r')
        hr_path = file_hr['image']

        return lr_path, hr_path

    def __len__(self):
        return len(self.lr_path)

    def __getitem__(self, idx):
        img_lr = self.lr_path[idx,:,:,:]
        img_lr = cv2.cvtColor(img_lr,cv2.COLOR_RGB2BGR) 
        img_hr =self.hr_path[idx,:,:,:]
        img_hr = cv2.cvtColor(img_hr,cv2.COLOR_RGB2BGR) 
        img_lr = self.transform(img_lr)
        img_hr = self.transform(img_hr)
    
        return img_lr,img_hr
class srDataset_4c(Dataset):
    def __init__(self, lr_dir,hr_dir, mode='train', transform=None):
        super(srDataset_4c, self).__init__()
        self.mode = mode
        self.transform = transform
        self.lr_path, self.hr_path = self.get_pathlist (lr_dir,hr_dir)
            
    
    def get_pathlist ( self, lr1,hr1):
        file_lr=h5py.File(lr1,'r')
        lr_path = file_lr['image']
        file_hr=h5py.File(hr1,'r')
        hr_path = file_hr['image']

        return lr_path, hr_path

    def __len__(self):
        return len(self.lr_path)

    def __getitem__(self, idx):
        img_lr = self.lr_path[idx,:,:,:]
        #print ( img_lr.shape)
        img_lr = cv2.cvtColor(img_lr,cv2.COLOR_RGBA2BGRA) # cv.COLOR_RGBA2BGRA
        img_hr =self.hr_path[idx,:,:,:]
        img_hr = cv2.cvtColor(img_hr,cv2.COLOR_RGBA2BGRA) 
        #print ( img_lr.shape)
        #print ( img_hr.shape)
        img_lr = self.transform(img_lr)
        img_hr = self.transform(img_hr)
        
    
        return img_lr,img_hr

if __name__ == "__main__":
    hr_path = r'C:\e\SR_datasets\SR\processed\gan_data\hr_hdf5_file.h5'
    lr_path =  r'C:\e\SR_datasets\SR\processed\gan_data\lr_hdf5_file.h5'

    data_transform = transforms.Compose([
    transforms.ToTensor()
    ])

    sr_data = srDataset( lr_dir = lr_path,hr_dir = hr_path, mode='train', transform=data_transform)
    
    print(sr_data[5])  # 此处为__getitem__的用法
    print(len(sr_data))  # 此处为__len__的用法
    