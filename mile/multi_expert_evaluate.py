import numpy as np
from sklearn.model_selection import cross_val_predict

from metrics import calculate_expert_accuracy
from utils import get_subset


def objective_function(individual, weights_train):
    '''
    由个体的交叉验证结果计算三个专家目标
    :param individual: 个体
    :param weights_train: 每个类的权重，计算第三个目标
    :return: 三个专家目标
    '''
    y_sub, pred_proba = individual.y_sub_and_pred_proba  # 获取个体的实例选择标签和预测概率
    pred = np.argmax(pred_proba, axis=1)  # 获取个体的预测标签
    Acc1, Acc2, Acc3 = calculate_expert_accuracy(pred, y_sub, weights_train)  # 计算 Acc1、Acc2、Acc3
    return Acc1, Acc2, Acc3


def evaluate_individuals(individuals, estimator, x_train, y_train, n_splits, random_state, weights_train):
    '''
    多专家评估每一个个体
    :param individuals:个体
    :param estimator: 基分类器
    :param x_train: 训练集特征数据
    :param y_train: 训练集标签
    :param n_splits: k-fold的参数k
    :param random_seed: 随机种子
    :param weights_train: 每个类的权重参数
    :return: void
    '''
    for individual in individuals:
        if not individual.fitness.valid:  # 如果该个体的适应度没有计算过
            x_sub, y_sub = get_subset(individual, x_train, y_train)
            # y_pred_proba = k_fold_cross_validation(estimator=estimator, x=x_sub, y=y_sub, random_state=random_state,
            #                                        n_splits=n_splits, method='soft')  # 交叉验证得到软标签
            y_pred_proba = cross_val_predict(estimator, x_sub, y_sub, cv=n_splits, random_state=random_state,
                                             method='predict_proba')
            individual.y_sub_and_pred_proba = (y_sub, y_pred_proba)  # 保存个体的软标签和预测概率
            individual.fitness.values = objective_function(individual, weights_train)  # 计算个体的目标值


def evaluate_individual(individual, estimator, x_train, y_train, n_splits, random_state, weights_train):
    '''
    多专家评估每一个个体
    :param individual:个体
    :param estimator: 基分类器
    :param x_train: 训练集特征数据
    :param y_train: 训练集标签
    :param n_splits: k-fold的参数k
    :param random_seed: 随机种子
    :param weights_train: 每个类的权重参数
    :return: void
    '''
    if not individual.fitness.valid:  # 如果该个体的适应度没有计算过
        x_sub, y_sub = get_subset(individual, x_train, y_train)
        y_pred_proba = cross_val_predict(estimator, x_sub, y_sub, cv=n_splits, method='predict_proba')
        individual.y_sub_and_pred_proba = (y_sub, y_pred_proba)  # 保存个体的软标签和预测概率
        individual.fitness.values = objective_function(individual, weights_train)  # 计算个体的目标值
        return {"valid": True, "fitness": individual.fitness.values,
                "y_sub_and_pred_proba": individual.y_sub_and_pred_proba}
    return {"valid": False}
