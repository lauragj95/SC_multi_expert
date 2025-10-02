import random
from skimage.io import imsave
from utils.patching_utils import recontruct_image_from_patches
import copy
from utils.distance_utils import calculate_distances

def save_expert(modif_labels,train_imgs_names,exp,output_dir):
    """
    Saves reconstructed expert-labeled images to the specified output directory.

    Args:
        modif_labels (list): A list where each element contains the modified label patches for an image.
        train_imgs_names (list): A list of image names corresponding to the training images.
        exp (str): The expert identifier or name, used to organize output directories.
        output_dir (str): The base directory where the reconstructed images will be saved.

    Notes:
        - Assumes that each entry in `modif_labels` contains at least 256 patches.
        - Uses `recontruct_image_from_patches` to reconstruct the image from its patches.
        - Images are saved in PNG format under the path: {output_dir}/{exp}/{im}.png.
    """
    for idx,im in enumerate(train_imgs_names):
        patches = [modif_labels[idx][i] for i in range(0,256)]
        reconstructed_img = recontruct_image_from_patches(patches)
        imsave(f'{output_dir}/{exp}/{im}.png', reconstructed_img.astype('uint8'))


def create_expert(expert, cls, cl_list, labels, img_names, change_ratio, seed,save=False,output_dir=None):
    """
    Creates an expert by modifying class labels according to a specified change ratio and class mapping.
    Args:
        expert (str or int): Identifier for the expert being created.
        cls (dict): Dictionary mapping original class labels to lists of target class labels for modification.
        cl_list (dict): Dictionary mapping class labels to dictionaries of sample indices and associated data.
        labels (list or np.ndarray): Original labels for the dataset, structured as a list or array.
        img_names (list): List of image names corresponding to the dataset samples.
        change_ratio (float): Ratio (between 0 and 1) of samples to be changed for each class.
        seed (int): Random seed for reproducibility.
        save (bool, optional): Whether to save the modified labels and expert data. Defaults to False.
        output_dir (str, optional): Directory where the expert data should be saved if `save` is True.
    Returns:
        tuple: 
            - cl_exps (dict): Dictionary of class experts after modification.
            - dists (dict): Dictionary of calculated distances for each class expert.
    Side Effects:
        If `save` is True, saves the modified labels and expert data to the specified `output_dir`.
    Notes:
        - The function randomly selects a subset of samples from each class to modify, based on `change_ratio`.
        - The labels of the selected samples are changed from the original class to the target class as specified in `cls`.
        - The function ensures reproducibility by setting the random seed.
    """
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
        save_expert(modif_labels,img_names,expert,output_dir)

              
    for cl_orig,cl_modif in cls.items():
        for cl in cl_modif:
            cl_exps[cl].update(changes[f'{cl_orig}_{cl}'])
    for cl_orig in cls.keys(): 
        dists[cl_orig]=calculate_distances(cl_exps[cl_orig])
    return cl_exps,dists