import numpy as np


def compute_weights(outlier_class, outliers,dists,sc_cl, cl,dbscan=True):
    weights = [0]
    for i in range(len(dists)):
        if dbscan:
            weight = (np.max(dists) - dists[i]) * (1-outlier_class) * outliers[i] 
        else:
            weight = (np.max(dists) - dists[i])
        weights.append(weight)
    weights.insert(cl,sc_cl)
    
    return weights 