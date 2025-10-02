import numpy as np


from sklearn.metrics import accuracy_score,brier_score_loss,f1_score,precision_score,recall_score

def dice_coef_binary(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

def dice_coef_multilabel(y_true, y_pred,class_no):
    dice_per_class = []
    for index in range(1,class_no):
        dice_per_class.append(dice_coef_binary(y_true == index, y_pred == index))

    return np.array(dice_per_class)

def f1_multilabel(y_true, y_pred,class_no):
    f1_per_class = []
    for index in range(1,class_no):

        f1_per_class.append(f1_score(y_true == index, y_pred == index,zero_division=1))

    return np.array(f1_per_class)

def prec_multilabel(y_true, y_pred,class_no):
    prec_per_class = []
    for index in range(1,class_no):

        prec_per_class.append(precision_score(y_true == index, y_pred == index,zero_division=1))

    return np.array(prec_per_class)

def recall_multilabel(y_true, y_pred,class_no):
    recall_per_class = []
    for index in range(1,class_no):

        recall_per_class.append(recall_score(y_true == index, y_pred == index,zero_division=1))

    return np.array(recall_per_class)

def brier_multilabel(y_true, y_pred,weights,class_no):
    brier_per_class = []
    for index in range(1,class_no):

        brier_per_class.append(weights[index]*brier_score_loss(y_true == index, y_pred == index))

    return np.array(brier_per_class).mean()

def acc_multilabel(y_true, y_pred,weights,class_no):
    acc_per_class = []
    for index in range(1,class_no):

        acc_per_class.append(weights[index]*accuracy_score(y_true == index, y_pred == index))

    return np.array(acc_per_class).mean()

def segmentation_scores(label_trues, label_preds, metric_names,class_no,class_names,weights,ignore_class):
    '''
    :param label_trues:
    :param label_preds:
    :param n_class:
    :return:
    '''
    results = {}


    assert len(label_trues) == len(label_preds)

    label_preds = np.array(label_preds, dtype='int8')
    label_trues = np.array(label_trues, dtype='int8')
   
    if ignore_class==0:
        label_preds = label_preds[label_trues!=0]
        label_trues = label_trues[label_trues!=0]

    dice_per_class = dice_coef_multilabel(label_trues, label_preds,class_no)
    f1_per_class = f1_multilabel(label_trues, label_preds,class_no)
    prec_per_class = prec_multilabel(label_trues, label_preds,class_no)
    rec_per_class = recall_multilabel(label_trues, label_preds,class_no)

    results['macro_dice'] = dice_per_class.mean()

    intersection = (label_preds == label_trues).sum(axis=None)
    sum_ = 2 * np.prod(label_preds.shape)
    results['micro_dice'] = ((2 * intersection + 1e-6) / (sum_ + 1e-6))
    for class_id in range(1,class_no):
        results['dice_class_' + str(class_id) + '_' + class_names[class_id]] = dice_per_class[class_id-1]
        results['f1_class_' + str(class_id) + '_' + class_names[class_id]] = f1_per_class[class_id-1]
        results['recall_class_' + str(class_id) + '_' + class_names[class_id]] = prec_per_class[class_id-1]
        results['prec_class_' + str(class_id) + '_' + class_names[class_id]] = rec_per_class[class_id-1]
    results['brier'] = brier_multilabel(label_trues,label_preds,weights,class_no)
    results['accuracy'] = acc_multilabel(label_trues, label_preds,weights,class_no)


    for metric in metric_names:
        assert metric in results.keys()

    return results