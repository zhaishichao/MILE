import numpy as np
from sklearn.metrics import confusion_matrix


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
    s_per_class = cm.sum(axis=1) # 对每个类求和，即所有预测的实例个数之和，计算Acc2和Acc3
    acc1 = np.sum(tp_per_class) / np.sum(s_per_class)  # Acc1
    acc2 = np.mean(tp_per_class.astype(float) / s_per_class.astype(float))  # Acc2
    acc3 = np.mean((tp_per_class.astype(float) / s_per_class.astype(float)) * weights)  # Acc3
    return round(acc1, 6), round(acc2, 6), round(acc3, 6)