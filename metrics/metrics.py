import numpy as np
from scipy.stats import gmean
from sklearn.metrics import confusion_matrix, roc_auc_score


def calculate_expert_accuracy(y_pred, y, weights):
    '''
    Calculate three expert objectives.
    :param y_pred: Prediction results (cross-validation results during training)
    :param y: True labels
    :param weights: Weight of each class, used to calculate the third expert objective
    :return: Three expert objectives
    '''
    cm = confusion_matrix(y, y_pred)
    tp_per_class = cm.diagonal()
    s_per_class = cm.sum(axis=1)
    acc1 = np.sum(tp_per_class) / np.sum(s_per_class)  # Acc_ocd
    acc2 = np.mean(tp_per_class.astype(float) / s_per_class.astype(float))  # Acc_bcd
    acc3 = np.mean((tp_per_class.astype(float) / s_per_class.astype(float)) * weights)  # Acc_icd
    return round(acc1, 6), round(acc2, 6), round(acc3, 6)


def calculate_gmean_mauc(y_pred_proba, y):
    '''
    Calculate G-mean and MAUC.
    :param y_pred_proba:
    :param y: True labels
    :return: geometric_mean and auc_ovo_macro
    '''
    num_classes = len(np.unique(y))  # Number of classe
    if num_classes == 2:
        auc_ovo_macro = roc_auc_score(y, y_pred_proba[:, 1])
    else:
        auc_ovo_macro = roc_auc_score(y, y_pred_proba, multi_class="ovo", average="macro")
    y_pred = np.argmax(y_pred_proba, axis=1)
    cm = confusion_matrix(y, y_pred)
    recall_per_class = cm.diagonal() / cm.sum(axis=1)
    geometric_mean = gmean(recall_per_class)
    return round(geometric_mean, 6), round(auc_ovo_macro, 6)
