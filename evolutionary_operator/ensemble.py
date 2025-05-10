import numpy as np
from scipy import stats
from sklearn.base import clone

from utils import get_subset


def ensemble_individuals(individuals, estimator, x_train, y_train):
    '''
    由个体选择的实例子集训练分类器并集成
    :param individuals: 参与集成的个体
    :param estimator: 基分类器
    :param x_train: 原始特征数据
    :param y_train: 原始标签
    :return: 分类器集合
    '''
    ensembles = []
    # 遍历每个个体，获取子集和标签，并训练分类器
    for ind in individuals:
        x_sub, y_sub = get_subset(ind, x_train, y_train)
        estimator_clone = clone(estimator)
        estimator_clone.fit(x_sub, y_sub)  # 训练分类器
        ind.estimator = estimator_clone  # 将训练好的分类器赋值给个体的estimator
        ensembles.append(ind.estimator)  # 将分类器添加到ensembles列表中
    return ensembles


def vote_result_ensembles(ensembles, x_test, result='soft'):
    '''
    投票的方式，集成分类结果
    :param ensembles: 分类器集合
    :param x_test: 测试集
    :param result: soft为软标签，表示预测概率；hard为实际的预测标签
    :return: 预测结果
    '''
    y_pred_labels_ensembles = []
    y_pred_prob_labels_ensembles = []
    #  遍历每个个体，获取预测结果
    for estimator in ensembles:
        ind_pred = estimator.predict(x_test)
        ind_proba = estimator.predict_proba(x_test)
        y_pred_labels_ensembles.append(ind_pred)
        y_pred_prob_labels_ensembles.append(ind_proba)
    vote_pred = stats.mode(y_pred_labels_ensembles, axis=0, keepdims=False).mode.flatten()  # 按列投票，取每列中的众数作为最终分类结果
    vote_pred_prob = np.stack(y_pred_prob_labels_ensembles, axis=0)  # 堆叠为三维数组
    vote_pred_prob = np.mean(vote_pred_prob, axis=0)  # 沿第一个维度 (num_classifiers) 求平均
    if result == 'soft':
        return vote_pred_prob
    elif result == 'hard':
        return vote_pred
    else:
        raise ValueError('result must be "soft" or "hard"')  # 提示出错