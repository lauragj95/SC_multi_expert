import os
import torch
import numpy as np
import pandas as pd
import cv2
import random
import json

from torch.utils import data

import albumentations as albu
import utils.globals as globals
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from utils.preprocessing import get_preprocessing_fn_without_normalization


def get_training_augmentation():
    aug_config = globals.config['data']['augmentation']
    if aug_config['use_augmentation']:
        train_transform = [
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.RandomRotate90(p=0.5),

            albu.Blur(blur_limit=aug_config['gaussian_blur_kernel'], p=0.5),
            albu.RandomBrightnessContrast(brightness_limit=aug_config['brightness_limit'],
                                          contrast_limit=aug_config['contrast_limit'],
                                          p=0.5),
            albu.HueSaturationValue(hue_shift_limit=aug_config['hue_shift_limit'],
                                    sat_shift_limit=aug_config['sat_shift_limit'],
                                    p=0.5)
        ]
        composed_transform = albu.Compose(train_transform)
    else:
        composed_transform = None
    return composed_transform


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def to_tensor1(x, **kwargs):
    return torch.from_numpy(x).type(torch.FloatTensor)

def to_tensor2(x, **kwargs):
    return torch.from_numpy(x).type(torch.LongTensor)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor2),
    ]
    return albu.Compose(_transform)

def get_preprocessingProb(preprocessing_fn):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor1),
    ]
    return albu.Compose(_transform)

# =============================================
class ProbDataset(torch.utils.data.Dataset):
    """Custom Dataset. Read images, apply augmentation and preprocessing transformations.
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. normalization, shape manipulation, etc.)
    """
    def __init__(
            self,
            images_dir,
            masks_dir,
            augmentation=None,
            preprocessing=None
    ):
        label_folder = os.path.join(globals.config['data']['path'], 'masks/Train/sc')
        # self.ids = [file for file in os.listdir(label_folder) if 'eval' in file]
        self.ids = os.listdir(label_folder)
        self.images_fps = [os.path.join(images_dir, image_id.replace('.npy','.png')) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.class_no = globals.config['data']['class_no']
        self.class_values = [1,2,3]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.load(self.masks_fps[i])
        mask = mask.transpose(1,2,0)



        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            if mask.shape[0] != self.class_no:  
                mask = np.transpose(mask, (2, 0, 1)) 

            
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            # mask = mask.transpose(0,2,1)


        return image, mask, self.ids[i], 0

    def __len__(self):
        return len(self.ids)

    def set_class_values(self, class_no):
        if globals.config['data']['ignore_last_class']:
            class_values = list(range(class_no + 1))
        else:
            class_values = list(range(class_no))
        return class_values
    
class CustomDataset(torch.utils.data.Dataset):
    """Custom Dataset. Read images, apply augmentation and preprocessing transformations.
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. normalization, shape manipulation, etc.)
    """
    def __init__(
            self,
            images_dir,
            masks_dir,
            augmentation=None,
            preprocessing=None,
            train = False
    ):
        if train:
            label_folder = os.path.join(globals.config['data']['path'], 'masks/Train/sc')
            self.ids = [file.replace('.npy','.png') for file in os.listdir(label_folder)]
        else:
            self.ids =  os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.class_no = globals.config['data']['class_no']
        self.class_values = self.set_class_values(self.class_no)
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        mask[mask==5]=0
        mask[mask==4]=0
        # extract certain classes from mask (e.g. cars)
        # print(self.class_values)
        # masks = [np.unique(mask == v) for v in self.class_values]
        # mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask, self.ids[i], 0

    def __len__(self):
        return len(self.ids)

    def set_class_values(self, class_no):
        if globals.config['data']['ignore_last_class']==0:
            class_values = list(range(1,class_no+1 ))
        else:
            class_values = list(range(class_no))
        return class_values


class PandaProbDataset(torch.utils.data.Dataset):
    """Custom Dataset. Read images, apply augmentation and preprocessing transformations.
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. normalization, shape manipulation, etc.)
    """
    def __init__(
            self,
            dir,
            mode='train',
            exp='sc',
            fold=0,
            augmentation=None,
            preprocessing=None
    ):

        kfold = json.load(open(f'{dir}kfolds.json'))
        names = kfold[f'fold_{fold}'][mode]

        if mode == 'train':
            data_dir = f'{dir}overlap/'
        else:
            data_dir = dir
        self.ids = []
        for file in os.listdir(f'{data_dir}imgs'):
            split_file = file.split('_')
            if split_file[0]+'_'+split_file[1] in names:
                self.ids.append(file)
        self.images_fps = [os.path.join(data_dir, 'imgs', image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(data_dir, 'masks',exp, image_id.replace('.png','.npy')) for image_id in self.ids]
        self.class_no = globals.config['data']['class_no']
        self.class_values = [1,2,3,4,5]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.load(self.masks_fps[i])
        
        # mask = mask.transpose(2,0,1)

        # extract certain classes from mask (e.g. cars)
        # masks = [np.unique(mask == v) for v in self.class_values]
        # mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            if mask.shape[0] != self.class_no:  
                mask = np.transpose(mask, (2, 0, 1)) 

            
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            # mask = mask.transpose(0,2,1)


        return image, mask, self.ids[i], 0

    def __len__(self):
        return len(self.ids)

    def set_class_values(self, class_no):
        if globals.config['data']['ignore_last_class']:
            class_values = list(range(class_no + 1))
        else:
            class_values = list(range(class_no))
        return class_values
    
class PandaDataset(torch.utils.data.Dataset):
    """Custom Dataset. Read images, apply augmentation and preprocessing transformations.
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. normalization, shape manipulation, etc.)
    """
    def __init__(
            self,
            dir,
            augmentation=None,
            preprocessing=None,
            mode = 'train',
            exp = 'original',
            fold = 1
    ):
        kfold = json.load(open(f'{dir}kfolds.json'))
        names = kfold[f'fold_{fold}'][mode]

        if mode == 'train':
            data_dir = f'{dir}overlap/'
        else:
            data_dir = dir
        self.ids = []
        for file in os.listdir(f'{data_dir}imgs'):
            split_file = file.split('_')
            if split_file[0]+'_'+split_file[1] in names:
                self.ids.append(file)
        self.images_fps = [os.path.join(data_dir, 'imgs', image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(data_dir, 'masks',exp, image_id) for image_id in self.ids]
        self.class_no = globals.config['data']['class_no']
        self.class_values = self.set_class_values(self.class_no)
        self.augmentation = augmentation
        self.mode = mode
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        if self.mode!='train':
            mask[mask==2]=1
            mask[mask==3]=2
            mask[mask==4]=3
            mask[mask==5]=0

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask, self.ids[i], 0

    def __len__(self):
        return len(self.ids)

    def set_class_values(self, class_no):
        if globals.config['data']['ignore_last_class']==0:
            class_values = list(range(1,class_no+1 ))
        else:
            class_values = list(range(class_no))
        return class_values

class GleasonProbDataset(torch.utils.data.Dataset):
    """Custom Dataset. Read images, apply augmentation and preprocessing transformations.
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. normalization, shape manipulation, etc.)
    """
    def __init__(
            self,
            dir,
            exp='sc',
            augmentation=None,
            preprocessing=None
    ):

        self.ids = os.listdir(f'{dir}/imgs')

        self.images_fps = [os.path.join(dir, 'imgs', image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(dir, 'masks',exp, image_id.replace('.png','.npy')) for image_id in self.ids]
        self.class_no = globals.config['data']['class_no']
        self.class_values = [1,2,3,4,5]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.load(self.masks_fps[i])

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            if mask.shape[0] != self.class_no:  
                mask = np.transpose(mask, (2, 0, 1)) 

            
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            # mask = mask.transpose(0,2,1)


        return image, mask, self.ids[i], 0

    def __len__(self):
        return len(self.ids)

    def set_class_values(self, class_no):
        if globals.config['data']['ignore_last_class']:
            class_values = list(range(class_no + 1))
        else:
            class_values = list(range(class_no))
        return class_values
    

class GleasonDataset(torch.utils.data.Dataset):
    """Custom Dataset. Read images, apply augmentation and preprocessing transformations.
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. normalization, shape manipulation, etc.)
    """
    def __init__(
            self,
            dir,
            augmentation=None,
            preprocessing=None,
            mode = 'validation',            
            ann = 'path1',

    ):

        data_dir = dir
        self.ids = os.listdir(f'{data_dir}/masks/{ann}')
            
        self.images_fps = [os.path.join(data_dir, 'imgs',mode, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(data_dir, 'masks',ann, image_id) for image_id in self.ids]
        self.class_no = globals.config['data']['val']['class_no']
        self.class_values = self.set_class_values(self.class_no)
        self.augmentation = augmentation
        self.mode = mode
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        mask[mask==3]=2
        mask[mask==4]=3
        mask[mask==5]=4

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask, self.ids[i], 0

    def __len__(self):
        return len(self.ids)

    def set_class_values(self, class_no):
        if globals.config['data']['ignore_last_class']==0:
            class_values = list(range(1,class_no+1 ))
        else:
            class_values = list(range(class_no))
        return class_values
    
class InferenceDataset(torch.utils.data.Dataset):
    """Custom Dataset. Read images, apply augmentation and preprocessing transformations.
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. normalization, shape manipulation, etc.)
    """
    def __init__(
            self,
            images_dir,
            masks_dir,
            kfold_file = '/datadisk/datasets/PANDA/patches/kfolds.json',
            augmentation=None,
            preprocessing=None,
            data = 'Panda',
            class_no = 5,
            ignore_class=0,
            fold = 1
    ):

        self.data = data
        if self.data == 'Panda':
            kfold = json.load(open(kfold_file))
            names = kfold[f'fold_{fold}']['test']

            self.ids = []
            for file in os.listdir(images_dir):
                split_file = file.split('_')
                if split_file[0]+'_'+split_file[1] in names:
                    self.ids.append(file)
    
        else:
            self.ids =  os.listdir(masks_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.class_no = class_no
        self.class_values = self.set_class_values(self.class_no)
        self.ignore_class = ignore_class
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        if self.data == 'Gleason19' or self.data=='Panda':
            mask[mask==2]=1
            mask[mask==3]=2
            mask[mask==4]=3
            mask[mask==5]=4
        elif self.data == 'Arvaniti':
            mask[mask==3]=2
            mask[mask==4]=3
            mask[mask==5]=4
        else:
            mask[mask==5]=0
            mask[mask==4]=0


        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask, self.ids[i], 0

    def __len__(self):
        return len(self.ids)

    def set_class_values(self, class_no,ignore_class=0):
        if ignore_class==0:
            class_values = list(range(1,class_no+1 ))
        else:
            class_values = list(range(class_no))
        return class_values



def get_data_supervised():
    config = globals.config
    batch_size = config['model']['batch_size']
    normalization = config['data']['normalization']
    prob = config['data']['prob']
    dataset = config['data']['dataset']
    annotators = []

    if normalization:
            encoder_name = config['model']['encoder']['backbone']
            encoder_weights = config['model']['encoder']['weights']
            preprocessing_fn = get_preprocessing_fn(encoder_name, pretrained=encoder_weights)
    else:
        preprocessing_fn = get_preprocessing_fn_without_normalization()

    if dataset=='panda':
        train_folder = config['data']['path']
        if prob:
            preprocessing = get_preprocessingProb(preprocessing_fn)
            train_dataset = PandaProbDataset(train_folder, exp=config['data']['exp'],augmentation=get_training_augmentation(),
                                        preprocessing = preprocessing)

        else:
            preprocessing = get_preprocessing(preprocessing_fn)
            train_dataset = PandaDataset(train_folder, augmentation=get_training_augmentation(),
                                        preprocessing = preprocessing)
        preprocessing = get_preprocessing(preprocessing_fn)
        validate_dataset = PandaDataset(train_folder,mode='valid', preprocessing = preprocessing)

        test_dataset = PandaDataset(train_folder, mode='test',preprocessing = preprocessing)

        trainloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
        validateloader = data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size,
                                        drop_last=False)
        testloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size,
                                    drop_last=False)
        

    elif dataset == 'gleason19':
        train_folder = config['data']['path']
        val_folder = config['data']['val']['path']
        test_folder = config['data']['test']['path']

        if prob:
            preprocessing = get_preprocessingProb(preprocessing_fn)
            train_dataset = GleasonProbDataset(train_folder, exp=config['data']['exp'],augmentation=get_training_augmentation(),
                                        preprocessing = preprocessing)

        else:
            
            preprocessing = get_preprocessing(preprocessing_fn)
            train_dataset = GleasonDataset(train_folder, augmentation=get_training_augmentation(),
                                        preprocessing = preprocessing)
        preprocessing = get_preprocessing(preprocessing_fn)

        validate_dataset = PandaDataset(val_folder,mode='valid', preprocessing = preprocessing)
        test_dataset = PandaDataset(test_folder, mode='test',preprocessing = preprocessing)
        trainloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
        validateloader = data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size,
                                        drop_last=False)
        testloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size,
                                    drop_last=False)
 


        

    else:
        train_image_folder = os.path.join(config['data']['path'], config['data']['train']['images'])
        train_label_folder = os.path.join(config['data']['path'], config['data']['train']['masks'])
        val_image_folder = os.path.join(config['data']['path'], config['data']['val']['images'])
        val_label_folder = os.path.join(config['data']['path'], config['data']['val']['masks'])
        test_image_folder = os.path.join(config['data']['path'], config['data']['test']['images'])
        test_label_folder = os.path.join(config['data']['path'], config['data']['test']['masks'])

        if prob:
            preprocessing = get_preprocessingProb(preprocessing_fn)
            train_dataset = ProbDataset(train_image_folder, train_label_folder, augmentation=get_training_augmentation(),
                                        preprocessing = preprocessing)

        else:
            preprocessing = get_preprocessing(preprocessing_fn)
            train_dataset = CustomDataset(train_image_folder, train_label_folder, augmentation=get_training_augmentation(),
                                        preprocessing = preprocessing,train=True)
        preprocessing = get_preprocessing(preprocessing_fn)
        validate_dataset = CustomDataset(val_image_folder, val_label_folder, preprocessing = preprocessing)


        test_dataset = CustomDataset(test_image_folder, test_label_folder, preprocessing = preprocessing)

        trainloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
        validateloader = data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size,
                                        drop_last=False)
        testloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size,
                                    drop_last=False)

    return trainloader, validateloader, testloader, annotators

