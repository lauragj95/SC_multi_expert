# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:43:56 2024

@author: Laura
"""
import glob
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset

import cv2 
import os
import json 
# from PIL import Image
from skimage.io import imread
# from pretrain.utils import PatchDataAugmentationDINO
from source.transformations import random_hv_flip,color_jitter,Fliplr_image,Flipud_image,Rot90_image,Rot180_image,ColorHEDAugmentation

def get_Gleason_imgs(dir):
        
        img_files = os.listdir(f'{dir}/imgs_crop')
        imgs = []
        names = []
        for file in img_files:
            file_name = file.split('.')[0]
            imgs.append(imread(f'{dir}/imgs_crop/{file_name}.jpg'))
            names.append(file_name)
        
        return imgs,names


def get_Gleason(dir,expert):
        files = os.listdir(f'{dir}/maps_crop/{expert}')
        img_files = os.listdir(f'{dir}/imgs_crop')
        labels = []
        names = []
        for file in img_files:
            file_name = file.split('.')[0]
            if file_name + '.png' in files:
                label = imread(f'{dir}/maps_crop/{expert}/{file_name}.png')
                # if 2 in np.unique(label):
                #     label[label==2] = 1
                # if 3 in np.unique(label):
                #     label[label==3] = 2
                # if 4 in np.unique(label):
                #     label[label==4] = 3
                # if 5 in np.unique(label):
                #     label[label==5] = 4
                labels.append(label)
            else:
                labels.append(np.zeros((4096,4096),dtype=int)-1)
            names.append(file_name)
        
        return labels,names

def get_PANDA(dir):
        img_files = os.listdir(f'{dir}/train_imgs')
        imgs = []
        labels = []
        names = []
        for file in img_files:
            file_name = file.split('.')[0]
            label = imread(f'{dir}/train_masks/{file_name}.png')
            # if 2 in np.unique(label):
            #     label[label==2] = 1
            # if 3 in np.unique(label):
            #     label[label==3] = 2
            # if 4 in np.unique(label):
            #     label[label==4] = 3
            # if 5 in np.unique(label):
            #     label[label==5] = 4
            labels.append(label)
            
            imgs.append(imread(f'{dir}/train_imgs/{file_name}.png'))
            names.append(file_name)
        
        return imgs,labels,names    

def get_Gleason_consensus(dir):
        img_files = os.listdir(f'{dir}/imgs_crop')
        imgs = []
        labels = []
        names = []
        for file in img_files:
            file_name = file.split('.')[0]
            label = imread(f'{dir}/fused_ann/{file_name}.png')
            if 2 in np.unique(label):
                label[label==2] = 1
            if 3 in np.unique(label):
                label[label==3] = 2
            if 4 in np.unique(label):
                label[label==4] = 3
            if 5 in np.unique(label):
                label[label==5] = 4
            labels.append(label)
            
            imgs.append(imread(f'{dir}/imgs_crop/{file_name}.jpg'))
            names.append(file_name)
        
        return imgs,labels,names
    
def get_Gleason1(dir,n_fold, json_file, expert, train=True):
        kfold = json.load(open(json_file))
        imgs = []
        labels = []
        names = []
        if train:
            train_imgs = kfold[f'expert_{expert}']['train']
            files = os.listdir(f'{dir}/maps_crop/{expert}')
            for file in files:
                file_name = file.split('.')[0]
                if file in train_imgs:
                    label = imread(f'{dir}/maps_crop/{expert}/{file}')
                    if 2 in np.unique(label):
                        label[label==2] = 1
                    if 3 in np.unique(label):
                        label[label==3] = 2
                    if 4 in np.unique(label):
                        label[label==4] = 3
                    if 5 in np.unique(label):
                        label[label==5] = 4
                    imgs.append(imread(f'{dir}/imgs_crop/{file_name}.jpg'))
                    labels.append(label)
                    names.append(file_name)
                            
                

        else:
            valid_imgs = kfold[f'expert_{expert}']['valid']
            files = os.listdir(f'{dir}/maps_crop/{expert}')

            for file in files:
                file_name = file.split('.')[0]
                if file in valid_imgs:
                    label = imread(f'{dir}/maps_crop/{expert}/{file}')
                    if 2 in np.unique(label):
                        label[label==2] = 1
                    if 3 in np.unique(label):
                        label[label==3] = 2
                    if 4 in np.unique(label):
                        label[label==4] = 3
                    if 5 in np.unique(label):
                        label[label==5] = 4
                    imgs.append(imread(f'{dir}/imgs_crop/{file_name}.jpg'))
                    labels.append(label)
                    names.append(file_name)
        return imgs, labels,names


    
class GleasonDataset(Dataset):
    def __init__(self, X, train):
        self.img = X[0]
        self.label = X[1]
        self.train = train
        
    def __getitem__(self, index):
        x = self.img[index]
        y = self.label[index]
        color_jitter = transforms.ColorJitter(brightness=(0,26),
                                hue=(-0.3,0.3),
                                saturation=(0,0.2),
                                contrast=(0.75,1.25))
        if self.train:
            # data_transforms = transforms.Compose([
            #     transforms.RandomHorizontalFlip(),
            #     transforms.RandomVerticalFlip(),
            #     transforms.RandomRotation((90,90)),
            #     transforms.RandomRotation((180,180)),
            #     transforms.RandomRotation((270,270))])
            data_transforms = transforms.Compose([
                Fliplr_image(),Flipud_image(),Rot90_image(),Rot180_image()
            ]) 
            ksize = np.random.randint(0, 3, size=(2,))
            ksize = tuple((ksize * 2 + 1).tolist())
            img_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomApply([color_jitter], p=0.8),
                                                # ColorHEDAugmentation(),
                                                   transforms.GaussianBlur(kernel_size=ksize),
                                                   transforms.ToTensor()
                                                  ])
            
        else:
            img_transforms = transforms.Compose([transforms.ToTensor()])
            # transform =  PatchDataAugmentationDINO(
            #     (0.14, 1),
            #     (0.05,0.4),
            #     8,
            # )


        
        if self.train:
            [x,y] = data_transforms([x,y])
        x = img_transforms(x)   
        y = np.array(y)
        y=torch.from_numpy(y)
        y = y.type(torch.LongTensor)

        return index,x,y


    def __len__(self):
        return len(self.img)
    


class GleasonDatasetImages(Dataset):
    def __init__(self, X, train):
        self.img = X[0]
        self.train = train
        
    def __getitem__(self, index):
        x = self.img[index]

        img_transforms = transforms.Compose([transforms.ToTensor()])
        x = img_transforms(x)   

        return index,x


    def __len__(self):
        return len(self.img)
    
# data = get_Gleason1("D:/datasets/Gleason19", 0, 'kfolds.json', 1, True)
# train_dataset = GleasonDataset(data,False)

# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=5, shuffle=False,
#     num_workers=1, pin_memory=True, drop_last=True)

# get_Gleason('/home/laura/Documents/dataset/Gleason19/',1)