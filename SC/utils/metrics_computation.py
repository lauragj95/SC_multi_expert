import numpy as np



def calculate_max_threshold(original,uncertainty_map,notall=False):
    """
    Computes threshold-based metrics for evaluating the performance of an uncertainty map against a ground truth mask.
    This function iterates over a range of probability thresholds and, for each threshold, calculates two metrics:
    - `final_max`: The maximum of the normalized false negative rate (npr) and the normalized false positive rate (ppr).
    - `final_sum`: The sum of npr and ppr.
    The function is designed to help select an optimal threshold for uncertainty-based segmentation or classification tasks.
    Args:
        original (np.ndarray): Ground truth mask or label array.
        uncertainty_map (np.ndarray): Array representing uncertainty values for each pixel or element.
        notall (bool, optional): If True, sets elements in `original` to 0 where the maximum uncertainty is at index 0. Default is False.
    Returns:
        tuple:
            final_max (dict): Dictionary mapping each threshold (float) to the maximum of npr and ppr at that threshold.
            final_sum (dict): Dictionary mapping each threshold (float) to the sum of npr and ppr at that threshold.
    """
    if notall:
        original[np.argmax(uncertainty_map,axis=0)==0] = 0
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
    final_max={}
    final_sum={}
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
        final_max[prob] = max(npr[prob],ppr[prob])

        final_sum[prob] = npr[prob]+ppr[prob]
        prob+=0.01
    return final_max,final_sum