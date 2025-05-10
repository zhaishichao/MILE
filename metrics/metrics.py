import numpy as np
from scipy.stats import gmean
from sklearn.metrics import confusion_matrix, roc_auc_score


def calculate_expert_accuracy(y_pred, y, weights):
    '''
    计算三个专家目标
    :param y_pred: 预测结果（训练时为交叉验证的结果）
    :param y: 真实标签
    :param weights: 每个类的占的权重，用于计算第三个专家目标
    :return: 三个专家目标
    '''
    cm = confusion_matrix(y, y_pred)
    tp_per_class = cm.diagonal()  # 对角线元素表示每个类预测正确的个数，对角线求和，即所有预测正确的实例个数之和，计算Acc1
    s_per_class = cm.sum(axis=1)  # 对每个类求和，即所有预测的实例个数之和，计算Acc2和Acc3
    acc1 = np.sum(tp_per_class) / np.sum(s_per_class)  # Acc1
    acc2 = np.mean(tp_per_class.astype(float) / s_per_class.astype(float))  # Acc2
    acc3 = np.mean((tp_per_class.astype(float) / s_per_class.astype(float)) * weights)  # Acc3
    return round(acc1, 6), round(acc2, 6), round(acc3, 6)


def calculate_gmean_mauc(y_pred_proba, y):
    '''
    计算G-mean和MAUC
    :param y_pred_proba:
    :param y: 真实标签
    :return: geometric_mean和auc_ovo_macro
    '''
    num_classes = len(np.unique(y))  # 类别数量
    # 计算auc_ovo_macro
    if num_classes == 2:
        auc_ovo_macro = roc_auc_score(y, y_pred_proba[:, 1])
    else:
        auc_ovo_macro = roc_auc_score(y, y_pred_proba, multi_class="ovo", average="macro")
    y_pred = np.argmax(y_pred_proba, axis=1)
    cm = confusion_matrix(y, y_pred)

    recall_per_class = cm.diagonal() / cm.sum(axis=1)  # 计算每类召回率（每类正确预测个数 / 该类总数）
    geometric_mean = gmean(recall_per_class)  # geometric_mean
    return round(geometric_mean, 6), round(auc_ovo_macro, 6)
