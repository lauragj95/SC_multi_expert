# -*- coding: utf-8 -*-
"""
Created on Mon May 13 10:19:14 2024

@author: Laura
"""

import os 
from source.models import HIPT
import torch
from source import dataset_gleason
from skimage.io import imsave,imread
import matplotlib.pyplot as plt
import numpy as np

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(device)
    
checkpoint_256 = './checkpoints/vit_256_small_dino_fold_4.pt'
checkpoint_4k = './checkpoints/vit_4096_xs_dino_fold_4.pt'
checkpoint_segmentation = './runs/segmentation/epoch_069.pt'
data_dir = "D:/datasets/Gleason19"
out_dir = "./results/1"
n_fold = 0
json_file = 'kfolds.json'
expert = 1
batch_size = 1
model = HIPT(num_classes = 5, pretrain_vit_patch = checkpoint_256, pretrain_vit_region = checkpoint_4k)
model.relocate(gpu_id = 0)
model.to(device)

checkpoint = torch.load(checkpoint_segmentation)
state_dict = checkpoint.state_dict()
model.load_state_dict(state_dict)
model.eval()

train_data = dataset_gleason.get_Gleason1(data_dir, n_fold, json_file, expert, train = True)

train_dataset = dataset_gleason.GleasonDataset(train_data, False)

loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False,
    num_workers=0)

outs = []
outs_prob = []
labels = []
imgs = []
for idx, img, label in loader:
    # labels.append(label.squeeze(0))
    imgs.append(img.squeeze(0))
    img = img.to(device)
    out = model(img)
    # print(out.shape)
    outs_prob.append(torch.nn.Softmax(dim=1)(out).squeeze(0).cpu().detach().numpy())
    out = torch.argmax(out,dim=1).squeeze(0).cpu().detach().numpy()
    outs.append(out)
    # imsave(f'{out_dir}/{idx}.png')

m = 1
max_map = np.zeros(outs_prob[m][0,:,:].shape)
for i in range(max_map.shape[0]):
    for j in range(max_map.shape[1]):
        value = outs_prob[m][:,i,j]
        max_value = np.max(list(value))
        order_values = list(value)
        order_values.sort(reverse=True)
        second_max = order_values[1]
        max_map[i,j] = max_value - second_max

plt.figure(figsize=(10, 10))
plt.imshow(max_map,cmap='viridis')
plt.colorbar()
plt.show()

plt.figure(figsize=(10, 10))
plt.imshow(imgs[m].squeeze(0).permute(1,2,0))

out1 = outs[m]
# out1[out1>0] = 1
plt.figure(figsize=(10, 10))
plt.imshow(out1)

plt.figure(figsize=(10, 10))
plt.imshow(train_data[1][m])
    
for i in range(len(outs)):
    plt.figure(figsize=(10, 10))
    plt.imshow(train_data[0][i])
    
    plt.figure(figsize=(10, 10))
    plt.imshow(outs[i])

    plt.figure(figsize=(10, 10))
    plt.imshow(train_data[1][i])

expert = imread(f'{data_dir}/maps_crop/6/slide001_core004.png')
plt.figure(figsize=(10, 10))
plt.imshow(expert)
# cl = []
# for i in range(len(train_data[1])):
#     print(train_data[1][i].shape)
#     if train_data[1][i].shape[0] != 4096 or train_data[1][i].shape[1]!=4096:
#         print(train_data[2][i])
#     cl.extend(np.unique(out))

# classes = np.unique(cl)
    

