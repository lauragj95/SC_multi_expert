import numpy as np


def compute_weights(outlier_class, outliers,dists,sc_cl, cl,dbscan=True):
    """
    Compute weights for clustering or outlier detection based on distances and outlier information.
    Args:
        outlier_class (float): A value indicating the class's outlier status (typically 0 or 1).
        outliers (list or np.ndarray): List or array indicating outlier status for each sample.
        dists (list or np.ndarray): List or array of distances for each sample.
        sc_cl (float): Self-consistency value to be inserted at the index `cl`.
        cl (int): Index at which to insert the self-consistency value.
        dbscan (bool, optional): If True, compute weights using DBSCAN logic; otherwise, use a simpler weighting scheme. Defaults to True.
    Returns:
        list: List of computed weights with the self-consistency value inserted at the specified index.
    """
    weights = []
    for i in range(len(dists)):
        if dbscan:
            weight = (np.max(dists) - dists[i]) * (1-outlier_class) * outliers[i] 
        else:
            weight = (np.max(dists) - dists[i])
        weights.append(weight)
    weights.insert(cl,sc_cl)
    
    return weights 
