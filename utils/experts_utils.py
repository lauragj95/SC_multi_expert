import random
from skimage.io import imsave
from utils.patching_utils import recontruct_image_from_patches
import copy
from utils.distance_utils import calculate_distances

def save_expert(modif_labels,train_imgs_names,exp):
    for idx,im in enumerate(train_imgs_names):
        patches = [modif_labels[idx][i] for i in range(0,256)]
        reconstructed_img = recontruct_image_from_patches(patches)
        imsave(f'/home/laura/Documents/dataset/PANDA/expert_masks/{exp}/{im}.png', reconstructed_img.astype('uint8'))


def create_expert(expert, cls, cl_list, labels, img_names, change_ratio, seed,save=False):
    random.seed(seed) 
    cl_exps = {}
    changes = {}
    dists = {}
    modif_labels = copy.deepcopy(labels)
    for cl_orig,cl_modif in cls.items():
        cl_aux_rand = copy.deepcopy(list(cl_list[cl_orig].items()))
        random.shuffle(cl_aux_rand)
        cl_exps[cl_orig] = copy.deepcopy(cl_list[cl_orig])
        for i,cl in enumerate(cl_modif):
            if i==0:
                if change_ratio > 0.5 :
                    change = dict(cl_aux_rand[int(change_ratio*len(cl_list[cl_orig])):])
                else:
                    change = dict(cl_aux_rand[:int(change_ratio*len(cl_list[cl_orig]))])
            else:
                change = dict(cl_aux_rand[int(change_ratio*len(cl_list[cl_orig])):int(2*change_ratio*len(cl_list[cl_orig]))])
            changes[f'{cl_orig}_{cl}'] = change
            
            for ch in change.keys():
                cl_exps[cl_orig].pop(ch)
                idx = ch.split('_')
                modif_labels[int(idx[0])][int(idx[1])][modif_labels[int(idx[0])][int(idx[1])]==cl_orig] = cl
           
    
    if save:
        save_expert(modif_labels,img_names,expert)

              
    for cl_orig,cl_modif in cls.items():
        for cl in cl_modif:
            cl_exps[cl].update(changes[f'{cl_orig}_{cl}'])
    for cl_orig in cls.keys(): 
        dists[cl_orig]=calculate_distances(cl_exps[cl_orig])
    return cl_exps,dists