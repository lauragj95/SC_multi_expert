import numpy as np



def calculate_max_threshold(original,uncertainty_map):
    
    diff = original -np.argmax(uncertainty_map,axis=0) 
    diff[diff!=0] = 1
    diff[original==0] = -5
    cl,counts = np.unique(diff,return_counts=True)
    cl_dict = dict(zip(cl.astype(int),counts))
    good_count = cl_dict[0]
    if 1 in cl_dict.keys():
        bad_count = cl_dict[1]
    else:
        bad_count = 0
    diff[diff==0] = -5

    npr = {}
    final={}
    ppr = {}

    prob = 0.1
    while prob <=0.9:

        threshold = np.max(uncertainty_map,axis=0).copy()
        threshold[np.max(uncertainty_map,axis=0)>prob] = 0
        threshold[np.max(uncertainty_map,axis=0)<prob] = 1

        
        diff_a = diff - threshold 
        cl,counts = np.unique(diff_a,return_counts=True)
        cl_dict = dict(zip(cl.astype(int),counts))
        if 0 in cl_dict.keys():
            TN = cl_dict[0] # where pixels are WELL detected as bad
        else:
            TN = 0
        if -6 in cl_dict.keys():
            FN = cl_dict[-6]
        else:
            FN = 0
        
        FP = bad_count - TN
        TP = good_count - FN 

        npr[prob] = FN/(TP+FN)
        ppr[prob] = FP/(TN+FP+0.01)
        final[prob] = max(npr[prob],ppr[prob])
        prob+=0.01
    return final