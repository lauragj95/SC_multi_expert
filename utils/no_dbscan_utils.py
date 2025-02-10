
from self_consistency.sc import compute_sc,compute_sc2
from utils.distance_utils import calculate_distances,calculate_distances_diff_cl
import numpy as np
import torch
from self_consistency.weights import compute_weights


def get_metrics(cl):
    scs = []
    sc2s = []
    means = {}
    stds = {}
    eucl_distances = []
    min_dists = []
    e_distss = []

    for i in range(len(cl)):
        dist_intra = calculate_distances(cl[i])
        means[f'{i+1}_{i+1}'] = sum(dist_intra)/(len(cl)**2)
        stds[f'{i+1}_{i+1}'] = np.std(dist_intra)
        for j in range(len(cl)):
            if j!=i:
                dist_inter = calculate_distances_diff_cl(cl[i],cl[j])
                if f'{i+1}_{j+1}' or f'{j+1}_{i+1}' not in means.keys():
                    means[f'{i+1}_{j+1}'] = np.mean(dist_inter)
                    stds[f'{i+1}_{j+1}'] = np.std(dist_inter)

    for i in range(len(cl)):
        
        means_inter = []
        means_intra = []
        means_intra.append(means[f'{i+1}_{i+1}'])
        for j in range(len(cl)):
            if j!=i:
                means_inter.append(means[f'{i+1}_{j+1}']) 
            if f'{j+1}_{j+1}' != f'{i+1}_{i+1}':
                means_intra.append(means[f'{j+1}_{j+1}']) 
        sc2,eucl_distance = compute_sc2(means[f'{i+1}_{i+1}'],means_inter,np.ones(len(cl[i])))
        sc2s.append(sc2)
        eucl_distances.append(eucl_distance)
        sc,min_dist,_,e_dists = compute_sc(means_inter,means_intra,np.ones(len(cl[i])))
        scs.append(sc)
        min_dists.append(min_dist)
        e_distss.append(e_dists)
    

    return scs,sc2s,eucl_distances,min_dists,e_distss,means,stds


def get_weights(scs,dists):
    weights = []
    soft = torch.nn.Softmax()
    weights.append([10,0,0,0,0,0])
    for i,sc in enumerate(scs):
    
        weights.append(compute_weights(0,[],dists[i],sc,i+1,dbscan=False))
    
    percentage = []
    for w in weights:
        percentage.append(soft(torch.from_numpy(np.array(w).astype('float'))).cpu().detach().numpy())   

    return weights,percentage