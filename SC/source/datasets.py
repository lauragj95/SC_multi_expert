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
import os
import json 
from skimage.io import imread


def get_Gleason(dir,expert=None,mode="img"):
        """
        Load Gleason dataset images and/or labels.

        Args:
            dir (str): Base directory.
            expert (str, optional): Expert folder name under maps_crop. Required if mode='labels' or 'both'.
            mode (str): One of ['imgs', 'labels', 'both'].
                - 'img'   -> return images, names
                - 'label' -> return labels, names
                - 'both'   -> return (images, labels, names)

        Returns:
            Depending on mode:
                imgs, names
                labels, names
        """
        img_files = os.listdir(f'{dir}/imgs_crop')
        names = [file.split(".")[0] for file in img_files]

        imgs = []
        labels = []
        if mode in ["img", "both"]:
            imgs = [imread(f"{dir}/imgs_crop/{name}.jpg") for name in names]

        if mode in ["label", "both"]:
            if expert is None:
                raise ValueError("expert must be provided when mode='labels' or 'both'")
            files = os.listdir(f"{dir}/maps_crop/{expert}")
            for name in names:
                if f"{name}.png" in files:
                    labels.append(imread(f"{dir}/maps_crop/{expert}/{name}.png"))
                else:
                    labels.append(np.zeros((4096, 4096), dtype=int) - 1)
        
        if mode == "img":
            return imgs, names
        elif mode == "label":
            return labels, names
        elif mode == "both":
            return imgs, labels, names
        else:
            raise ValueError("mode must be 'img', 'label', or 'both'")



def get_PANDA(dir,expert='original',json_file=None,fold = "fold_0"):
        """
        Loads images and corresponding expert mask labels from the PANDA dataset.
        Args:
            dir (str): Path to the dataset directory.
            expert (str, optional): Name of the expert whose masks to use. Defaults to 'original'.
            json_file (str, optional): Name of the JSON file containing k-fold split information. If None, all images in 'train_imgs' are used. Defaults to None.
            fold (str, optional): Fold identifier to use from the JSON file. Defaults to "fold_0".
        Returns:
            tuple: A tuple containing:
                - imgs (list): List of loaded training images as numpy arrays.
                - labels (list): List of corresponding expert mask images as numpy arrays.
                - names (list): List of image file names (without extension).
        """
        
        if json_file is not None: 
            kfold = json.load(open(f'{dir}/{json_file}'))
            img_files = kfold[fold]['train']
        else:
            img_files = [f.split(".")[0] for f in os.listdir(f"{dir}/train_imgs")]
            
        imgs = []
        labels = []
        names = []
        for file in img_files:
            labels.append(imread(f'{dir}/expert_masks/{expert}/{file}.png'))
            imgs.append(imread(f'{dir}/train_imgs/{file}.png'))
            names.append(file)
        
        return imgs,labels,names    

 

def get_BCSS(dir,expert=None,mode="img"):
        """
        Load BCSS dataset images and/or labels.

        Args:
            dir (str): Base directory.
            expert (str, optional): Expert folder name under maps_crop. Required if mode='labels' or 'both'.
            mode (str): One of ['imgs', 'labels', 'both'].
                - 'img'   -> return images, names
                - 'label' -> return labels, names
                - 'both'   -> return (images, labels, names)

        Returns:
            Depending on mode:
                imgs, names
                labels, names
        """
        img_files = os.listdir(f'{dir}/complete_imgs')
        names = [file.split(".")[0] for file in img_files]

        remove = ['A0BG_BH', 'A0DA_A7', 'A0SK_A1', 'A0YM_A2', 'A1AY_AR', 'A1LS_E2',
       'A26Y_C8', 'A5D6_OL', 'A66I_OL', 'A73Y_LL']
        
        
        labels = []
        imgs = []
        if mode in ["img", "both"]:
            imgs = [imread(f"{dir}/complete_imgs/{file}") for file in img_files]
        if mode in ["label", "both"]:
            if expert is None:
                raise ValueError("expert must be provided when mode='labels' or 'both'")
            expert_files = os.listdir(f'{dir}/complete_masks/{expert}')
            for file in img_files:
                if file.split('.')[0] in remove:
                    if file in expert_files:
                        label = imread(f'{dir}/complete_masks/{expert}/{file}')
                        labels.append(label)
                    else:
                        labels.append(np.zeros((4096,4096),dtype=int)+10)
    
        if mode == "img":
            return imgs, names
        elif mode == "label":
            return labels, names
        elif mode == "both":
            return imgs, labels, names
        else:
            raise ValueError("mode must be 'img', 'label', or 'both'")


    
class CustomDataset(Dataset):
    def __init__(self, X, get_label=True):
        self.img = X[0]
        self.get_label = get_label
        if self.get_label:
            self.label = X[1]
        
    def __getitem__(self, index):
        x = self.img[index]
        y = self.label[index]
   

        img_transforms = transforms.Compose([transforms.ToTensor()])
        x = img_transforms(x)   

        if self.get_label:
            y = np.array(y)
            y=torch.from_numpy(y)
            y = y.type(torch.LongTensor)

            return index,x,y
        else:
            return index,x


    def __len__(self):
        return len(self.img)
    
