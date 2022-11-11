# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:25:48 2022

@author: maijieai
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from torch.autograd import Variable
import glob

def detransform( img,device ):
    mean = [0.5,0.5,0.5]
    std = [0.5,0.5,0.5]
    
    
    mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1).to(device)
    std = torch.tensor(std).unsqueeze(1).unsqueeze(1).to(device)

    imgt = img *std + mean
    
    imgn = imgt.cpu().detach().numpy().squeeze(0)
    imgn *= 225
    imgn = np.float32(imgn).transpose(1,2,0)
    print ( imgn.shape)
    return imgn


def psnr ( imgr,imgt,ycbcr=False, shave=0):
    if ycbcr:
        a = np.float32(imgr)
        b = np.float32(imgt)
        a = sc.rgb2ycbcr(a / 255)[:, :, 0]
        b = sc.rgb2ycbcr(b / 255)[:, :, 0]
    else:
        a = np.array(imgr).astype(np.float32)
        b = np.array(imgt).astype(np.float32)
        
    if shave:
        a = a[shave:-shave, shave:-shave]
        b = b[shave:-shave, shave:-shave]
    
    mse = np.mean((a - b) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return np.minimum(100.0, 20 * np.math.log10(PIXEL_MAX) - 10 * np.math.log10(mse))

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = np.math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * np.math.log10(255.0 / rmse)
    


def predict_simage ( model,img_path,img_path2,transforms1,device):
    img = Image.open(img_path)
    target = Image.open(img_path2)
    imgt = transforms1(img).unsqueeze(0)
    imgt = imgt.to(device)
    
    imgn = np.array (img)
    target = np.array ( target)
    #print ( imgn.shape)
    
    pre = model(imgt)
    pren = pre.cpu().detach().numpy().squeeze(0)
    pren *= 225
    pren = np.float32(pren).transpose(1,2,0)
    #pre_t = detransform(pre,device)
    #print ( imgn)
    #print ( pre_t)
    pre_s = pren.astype(np.uint8)
    pre_s1 = Image.fromarray ( pre_s)
    pre_s1.save ("test_image.jpg")
    #print ( pren.shape)
    p1 = psnr (target,pren)
    p3 = PSNR ( pren,target)
    #p4 = PSNR (imgn,target)
    #p2 = psnr ( target,imgn )
    print ( p1)
    print ( p3)
    #print ( p4)
    #print ( p2)
    
def predict_simager ( model,img_path,img_path2,transforms1,device):
    img = Image.open(img_path)
    target = Image.open(img_path2)
    imgt = transforms1(img).unsqueeze(0)
    imgt = imgt.to(device)
    
    imgn = np.array (img)
    target = np.array ( target)
    #print ( imgn.shape)
    
    pre = model(imgt)
    pren = pre.cpu().detach().numpy().squeeze(0)
    pren *= 225
    pren = np.float32(pren).transpose(1,2,0)
    #transpose(1,2,0)
    #pre_t = detransform(pre,device)
    
    pre_s = pren.astype(np.uint8)
    #cv2.imwrite ("testimg1.png",pre_s)
    
    pre_s1 = Image.fromarray ( pre_s,'RGBA')
    pre_s1.save ("test_image.png")
    
    p1 = psnr (target,pren)
    p3 = PSNR ( pren,target)
    #p4 = PSNR (imgn,target)
    #p2 = psnr ( target,imgn )
    print ( p1)
    print ( p3)
    #print ( p4)
    #print ( p2)        
    