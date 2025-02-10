# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 09:33:30 2024

@author: Laura
"""

import torch
import torch.backends.cudnn as cudnn
import numpy as np
from source.dataset_gleason import get_PANDA, GleasonDataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from source.vision_transformer import vit_small, vit4k_xs
from source.utils import update_state_dict
import random
import matplotlib
import matplotlib.patches as mpatches   
import copy 
import pandas as pd
from SC_maps_utils import get_patch_centroids,calculate_distances_core_noncore,get_patches,\
                        calculate_nearest_neighbors_non_core,recontruct_image_from_patches,\
                            calculate_distances,\
                                get_core_expert, get_weights,\
                                    get_metrics_NoDBSCAN,get_min_samples,compute_sc2
from skimage.io import imsave
import write_results

if torch.cuda.is_available():
    device = torch.device('cuda')
    cudnn.deterministic = True
    cudnn.benchmark = True
    print(device)
    
random.seed(0)
dir = '/home/laura/Documents/dataset/PANDA/'
img = ''
patch_size = 256
region_size = 4096
mini_patch_size = 16
checkpoint_256 = 'checkpoints/vit_256_small_dino_fold_4.pt'
checkpoint_4k = 'checkpoints/vit_4096_xs_dino_fold_4.pt'




#### DATA LOADER ####  
train_data = get_PANDA(dir)
train_dataset = GleasonDataset(train_data, False)
loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1, shuffle=False,
    num_workers=0)


#### SET MODELS ####
vit_patch = vit_small(
    img_size=patch_size,
    patch_size=mini_patch_size,
    embed_dim=384,
    mask_attn=False,
    num_register_tokens=0,
)

vit_region = vit4k_xs(
    img_size=region_size,
    patch_size=patch_size,
    input_embed_dim=384,
    output_embed_dim=192,
    mask_attn=False
)

state_dict = torch.load(checkpoint_256, map_location="cpu")
checkpoint_key = "teacher"
if checkpoint_key is not None and checkpoint_key in state_dict:
    state_dict = state_dict[checkpoint_key]
# remove `module.` prefix
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
# remove `backbone.` prefix induced by multicrop wrapper
state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
state_dict, msg = update_state_dict(vit_patch.state_dict(), state_dict)
vit_patch.load_state_dict(state_dict, strict=False)
for name, param in vit_patch.named_parameters():
    param.requires_grad = False
vit_patch.to(device)
vit_patch.eval()

state_dict = torch.load(checkpoint_4k, map_location="cpu")
if checkpoint_key is not None and checkpoint_key in state_dict:
    state_dict = state_dict[checkpoint_key]
# remove `module.` prefix
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
# remove `backbone.` prefix induced by multicrop wrapper
state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
state_dict, msg = update_state_dict(
    vit_region.state_dict(), state_dict
)
vit_region.load_state_dict(state_dict, strict=False)
for name, param in vit_region.named_parameters():
    param.requires_grad = False
vit_region.to(device)
vit_region.eval()


#### GET FEATURES ####
features = []
labels = []
for _,img,label in loader:
        img = img.to(device)
        label = label.to(device)
        # feat = model(img)
        feat,label = get_patches(img,label,vit_patch,vit_region,patch_size,region_size)
        features.extend(feat.cpu().detach().numpy())
        labels.append(label.cpu().detach().numpy())



features_aux = []
labels_aux = []
idx_aux = []
for i,lbl in enumerate(labels):
    feat = features[i]
    for j,label in enumerate(lbl):
            features_aux.append(feat[j])
            idx_aux.append(f'{i}_{j}')
            labels_aux.append(np.unique(label))

#### PCA ####
pca = PCA(n_components=0.9)
principalComponents = pca.fit_transform(features_aux)
explained_variance = pca.explained_variance_ratio_
total_variance = sum(list(explained_variance))*100


knn_features = []
knn_labels = []
knn_idx = []
for i,lbl in enumerate(labels_aux):
    if (2 in lbl and 3 in lbl) or (2 in lbl and 5 in lbl) or (3 in lbl and 4 in lbl) or (3 in lbl and 5 in lbl) or (4 in lbl and 5 in lbl):
        pass
    else:
        # if len(lbl)==1 and lbl[0]==0:
        #     pass
        # else:
        knn_features.append(principalComponents[i])
        knn_idx.append(idx_aux[i])
        if len(lbl)==1:
            
            # print(lbl)
            knn_labels.append(lbl[0])
        else:
            # if 3 in lbl or 4 in lbl or 5 in lbl:
                # print(lbl)
            knn_labels.append(lbl[-1])


###### DIVIDE BY CLASSES #####
cl_0 = {}
cl_1 = {}
cl_2 = {}
cl_3 = {}
cl_4 = {}
cl_5 = {}
for i, idx in enumerate(knn_idx):
    div_index = idx.split('_')
    expert_ann = labels[int(div_index[0])][int(div_index[1])]
    cl,count = np.unique(expert_ann,return_counts=True)
    cl_counts = dict(zip(cl,count))
    if len(cl)==1 and cl[0]==0:
        cl_0[idx] = knn_features[i]
    elif knn_labels[i] == 1 :
        if cl_counts[1]/(256**2)>0.15 and 0 not in cl:
            cl_1[idx] = knn_features[i]
    elif knn_labels[i] == 2:
        if cl_counts[2]/(256**2)>0.15 and 0 not in cl:
            cl_2[idx] = knn_features[i]
    elif knn_labels[i] == 3:
        if cl_counts[3]/(256**2)>0.15 and 0 not in cl:
            cl_3[idx] = knn_features[i] 
    elif knn_labels[i] == 4:
        if cl_counts[4]/(256**2)>0.15 and 0 not in cl:
            cl_4[idx] = knn_features[i] 
    elif knn_labels[i] == 5:
        if cl_counts[5]/(256**2)>0.15 and 0 not in cl:
            cl_5[idx] = knn_features[i] 
        


###### DISTANCES #######
dist0 = calculate_distances(cl_0)
dist1 = calculate_distances(cl_1)
dist2 = calculate_distances(cl_2)
dist3 = calculate_distances(cl_3)
dist4 = calculate_distances(cl_4)
dist5 = calculate_distances(cl_5)




######## DBSCAN - ORIGINAL #########

dbscan1 = DBSCAN(eps=np.percentile(dist1,10), min_samples=get_min_samples(cl_1,dist1)).fit(list(cl_1.values()))
centroids_1 = dbscan1.components_
labels_1 = dbscan1.labels_

dbscan2 = DBSCAN(eps=np.percentile(dist2,10), min_samples=get_min_samples(cl_2,dist2)).fit(list(cl_2.values()))
centroids_2 = dbscan2.components_
labels_2 = dbscan2.labels_

dbscan3 = DBSCAN(eps=np.percentile(dist3,10), min_samples=get_min_samples(cl_3,dist3)).fit(list(cl_3.values()))
centroids_3 = dbscan3.components_
labels_3 = dbscan3.labels_

dbscan4 = DBSCAN(eps=np.percentile(dist4,10), min_samples=get_min_samples(cl_4,dist4)).fit(list(cl_4.values()))
centroids_4 = dbscan4.components_
labels_4 = dbscan4.labels_

dbscan5 = DBSCAN(eps=np.percentile(dist5,10), min_samples=get_min_samples(cl_5,dist5)).fit(list(cl_5.values()))
centroids_5 = dbscan5.components_
labels_5 = dbscan5.labels_

######## ORIGINAL #########



sc1_orig,min_dist1_orig,outliers_sc1_orig,dists1_orig,sc2_orig,min_dist2_orig,\
outliers_sc2_orig,dists2_orig,sc3_orig,min_dist3_orig,outliers_sc3_orig,dists3_orig,\
sc4_orig,min_dist4_orig,outliers_sc4_orig,dists4_orig,sc5_orig,min_dist5_orig,\
outliers_sc5_orig,dists5_orig,means_orig,stds_orig =  get_metrics_NoDBSCAN([cl_1,cl_2,cl_3,cl_4,cl_5])

sc_orig2,eucl_distance_orig = compute_sc2(means_orig,[[],[],[],[],[]])
weights_orig = get_weights(sc1_orig,min_dist1_orig,outliers_sc1_orig,dists1_orig,
                           sc2_orig,min_dist2_orig,outliers_sc2_orig,dists2_orig,
                           sc3_orig,min_dist3_orig,outliers_sc3_orig,dists3_orig,
                           sc4_orig,min_dist4_orig,outliers_sc4_orig,dists4_orig,
                           sc5_orig,min_dist5_orig,outliers_sc5_orig,dists5_orig
                           )


##### INTRODUCE NOISE ######
## Mix 2-3 ##

random.seed(0)
cl2_aux_rand = copy.deepcopy(list(cl_2.items()))
cl3_aux_rand = copy.deepcopy(list(cl_3.items()))
random.shuffle(cl2_aux_rand)
random.shuffle(cl3_aux_rand)
change2_3 = dict(cl2_aux_rand[:int(0.2*len(cl_2))])
change3_2 = dict(cl3_aux_rand[:int(0.2*len(cl_3))])

modif_labels1 = copy.deepcopy(labels)
for change in change2_3.keys():
    idx = change.split('_')
    modif_labels1[int(idx[0])][int(idx[1])][modif_labels1[int(idx[0])][int(idx[1])]==2] = 3

for change in change3_2.keys():
    idx = change.split('_')
    modif_labels1[int(idx[0])][int(idx[1])][modif_labels1[int(idx[0])][int(idx[1])]==3] = 2  


cl2_exp1 = copy.deepcopy(cl_2)
for change in change2_3.keys():
    cl2_exp1.pop(change)
    
cl3_exp1 = copy.deepcopy(cl_3)
for change in change3_2.keys():
    cl3_exp1.pop(change)

cl3_exp1.update(change2_3)
cl2_exp1.update(change3_2)



sc1_exp1,min_dist1_exp1,outliers_sc1_exp1,dists1_exp1,sc2_exp1,min_dist2_exp1,\
outliers_sc2_exp1,dists2_exp1,sc3_exp1,min_dist3_exp1,outliers_sc3_exp1,dists3_exp1\
,sc4_exp1,min_dist4_exp1,outliers_sc4_exp1,dists4_exp1,sc5_exp1,min_dist5_exp1,\
outliers_sc5_exp1,dists5_exp1,means_exp1,stds_exp1 = get_metrics_NoDBSCAN(
    [cl_1,cl2_exp1,cl3_exp1,cl_4,cl_5])

sc_exp1_2,eucl_distance_exp1 = compute_sc2(means_exp1,[[],[],[],[],[]])

weights_exp1 = get_weights(sc1_exp1,min_dist1_exp1,outliers_sc1_exp1,dists1_exp1,
                           sc2_exp1,min_dist2_exp1,outliers_sc2_exp1,dists2_exp1,
                           sc3_exp1,min_dist3_exp1,outliers_sc3_exp1,dists3_exp1,
                           sc4_exp1,min_dist4_exp1,outliers_sc4_exp1,dists4_exp1,
                           sc5_exp1,min_dist5_exp1,outliers_sc5_exp1,dists5_exp1)

## Mix 3-4 ##
random.seed(10)
cl3_aux_rand = copy.deepcopy(list(cl_3.items()))
cl4_aux_rand = copy.deepcopy(list(cl_4.items()))
random.shuffle(cl4_aux_rand)
random.shuffle(cl3_aux_rand)
change4_3 = dict(cl4_aux_rand[:int(0.2*len(cl_4))])
change3_4 = dict(cl3_aux_rand[:int(0.2*len(cl_3))])

modif_labels2 = copy.deepcopy(labels)
for change in change4_3.keys():
    idx = change.split('_')
    modif_labels2[int(idx[0])][int(idx[1])][modif_labels2[int(idx[0])][int(idx[1])]==4] = 3

for change in change3_4.keys():
    idx = change.split('_')
    modif_labels2[int(idx[0])][int(idx[1])][modif_labels2[int(idx[0])][int(idx[1])]==3] = 4  


cl4_exp2 = copy.deepcopy(cl_4)
for change in change4_3.keys():
    cl4_exp2.pop(change)
    
cl3_exp2 = copy.deepcopy(cl_3)
for change in change3_4.keys():
    cl3_exp2.pop(change)

cl3_exp2.update(change4_3)
cl4_exp2.update(change3_4)

sc1_exp2,min_dist1_exp2,outliers_sc1_exp2,dists1_exp2,sc2_exp2,min_dist2_exp2,\
outliers_sc2_exp2,dists2_exp2,sc3_exp2,min_dist3_exp2,outliers_sc3_exp2,dists3_exp2,\
    sc4_exp2,min_dist4_exp2,outliers_sc4_exp2,dists4_exp2,sc5_exp2,min_dist5_exp2,\
outliers_sc5_exp2,dists5_exp2,means_exp2,stds_exp2 = get_metrics_NoDBSCAN(
    [cl_1,cl_2,cl3_exp2,cl4_exp2,cl_5]) 

sc_exp2_2,eucl_distance_exp2 = compute_sc2(means_exp2,[[],[],[],[],[]])

weights_exp2 = get_weights(sc1_exp2,min_dist1_exp2,outliers_sc1_exp2,dists1_exp2,sc2_exp2,min_dist2_exp2,
outliers_sc2_exp2,dists2_exp2,sc3_exp2,min_dist3_exp2,outliers_sc3_exp2,dists3_exp2,
sc4_exp2,min_dist4_exp2,outliers_sc4_exp2,dists4_exp2,sc5_exp2,min_dist5_exp2,
outliers_sc5_exp2,dists5_exp2)


## Mix 2-3 3-4 ##
random.seed(47)
cl2_aux_rand = copy.deepcopy(list(cl_2.items()))
cl3_aux_rand = copy.deepcopy(list(cl_3.items()))
cl4_aux_rand = copy.deepcopy(list(cl_4.items()))
random.shuffle(cl2_aux_rand)
random.shuffle(cl4_aux_rand)
random.shuffle(cl3_aux_rand)
change2_3 = dict(cl2_aux_rand[:int(0.15*len(cl_2))])
change4_3 = dict(cl4_aux_rand[:int(0.15*len(cl_4))])
change3_2 = dict(cl3_aux_rand[:int(0.15*len(cl_3))])
change3_4 = dict(cl3_aux_rand[int(0.15*len(cl_3)):int(0.3*len(cl_3))])
                
                
modif_labels = copy.deepcopy(labels)
for change in change4_3.keys():
    idx = change.split('_')
    modif_labels[int(idx[0])][int(idx[1])][modif_labels[int(idx[0])][int(idx[1])]==4] = 3

for change in change3_4.keys():
    idx = change.split('_')
    modif_labels[int(idx[0])][int(idx[1])][modif_labels[int(idx[0])][int(idx[1])]==3] = 4  

for change in change2_3.keys():
    idx = change.split('_')
    modif_labels[int(idx[0])][int(idx[1])][modif_labels[int(idx[0])][int(idx[1])]==2] = 3

for change in change3_2.keys():
    idx = change.split('_')
    modif_labels[int(idx[0])][int(idx[1])][modif_labels[int(idx[0])][int(idx[1])]==3] = 2  


cl2_exp3 = copy.deepcopy(cl_2)
for change in change2_3.keys():
    cl2_exp3.pop(change)
    
cl4_exp3 = copy.deepcopy(cl_4)
for change in change4_3.keys():
    cl4_exp3.pop(change)
    
cl3_exp3 = copy.deepcopy(cl_3)
for change in change3_4.keys():
    cl3_exp3.pop(change)
for change in change3_2.keys():
    cl3_exp3.pop(change)
    
cl3_exp3.update(change4_3)
cl3_exp3.update(change2_3)
cl2_exp3.update(change3_2)
cl4_exp3.update(change3_4)




sc1_exp3,min_dist1_exp3,outliers_sc1_exp3,dists1_exp3,sc2_exp3,min_dist2_exp3,\
outliers_sc2_exp3,dists2_exp3,sc3_exp3,min_dist3_exp3,outliers_sc3_exp3,dists3_exp3,\
sc4_exp3,min_dist4_exp3,outliers_sc4_exp3,dists4_exp3,sc5_exp3,min_dist5_exp3,\
outliers_sc5_exp3,dists5_exp3,means_exp3,stds_exp3 = get_metrics_NoDBSCAN(
    [cl_1,cl2_exp3,cl3_exp3,cl4_exp3,cl_5])

sc_exp3_2,eucl_distance_exp3 = compute_sc2(means_exp3,[[],[],[],[],[]])

weights_exp3 = get_weights(sc1_exp3,min_dist1_exp3,outliers_sc1_exp3,dists1_exp3,sc2_exp3,min_dist2_exp3,
outliers_sc2_exp3,dists2_exp3,sc3_exp3,min_dist3_exp3,outliers_sc3_exp3,dists3_exp3,
sc4_exp3,min_dist4_exp3,outliers_sc4_exp3,dists4_exp3,sc5_exp3,min_dist5_exp3,
outliers_sc5_exp3,dists5_exp3)




sc_orig = [sc1_orig,sc2_orig,sc3_orig,sc4_orig,sc5_orig]
sc_exp1 = [sc1_exp1,sc2_exp1,sc3_exp1,sc4_exp1,sc5_exp1]
sc_exp2 = [sc1_exp2,sc2_exp2,sc3_exp2,sc4_exp2,sc5_exp2]
sc_exp3 = [sc1_exp3,sc2_exp3,sc3_exp3,sc4_exp3,sc5_exp3]
sc_v1 = [sc_orig,sc_exp1,sc_exp2,sc_exp3]



sc_v2 = [sc_orig2,sc_exp1_2,sc_exp2_2,sc_exp3_2]


means = [means_orig,means_exp1,means_exp2,means_exp3]
stds = [stds_orig,stds_exp1,stds_exp2,stds_exp3]




edistances_orig = [dists1_orig,dists2_orig,dists3_orig,dists4_orig,dists5_orig]
edistances_exp1 = [dists1_exp1,dists2_exp1,dists3_exp1,dists4_exp1,dists5_exp1]
edistances_exp2 = [dists1_exp2,dists2_exp2,dists3_exp2,dists4_exp2,dists5_exp2]
edistances_exp3 = [dists1_exp3,dists2_exp3,dists3_exp3,dists4_exp3,dists5_exp3]
e_distance = [edistances_orig,edistances_exp1,edistances_exp2,edistances_exp3]


soft = torch.nn.Softmax()
percentage_orig = []
for w in weights_orig:
    percentage_orig.append(soft(torch.from_numpy(np.array(w).astype('float'))).cpu().detach().numpy())

percentage_exp1 = []
for w in weights_exp1:
    percentage_exp1.append(soft(torch.from_numpy(np.array(w).astype('float'))).cpu().detach().numpy())

percentage_exp2 = []
for w in weights_exp2:
    percentage_exp2.append(soft(torch.from_numpy(np.array(w).astype('float'))).cpu().detach().numpy())

percentage_exp3 = []
for w in weights_exp3:
    percentage_exp3.append(soft(torch.from_numpy(np.array(w).astype('float'))).cpu().detach().numpy())




weights = [weights_orig,weights_exp1,weights_exp2,weights_exp3]
percentage = [percentage_orig,percentage_exp1,percentage_exp2,percentage_exp3]



write_results.write_euclidean_distances(means,stds,'results/euclidean_distance_all.csv')
write_results.write_edistances(e_distance,'results/edistance_all.csv')
write_results.write_sc1(sc_v1,'results/sc1_all.csv')
write_results.write_sc2(sc_v2,'results/sc2_all.csv')
write_results.write_weights(weights,percentage,'results/weights_all.csv')