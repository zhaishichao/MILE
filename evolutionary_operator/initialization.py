import random

import numpy as np
import array

from deap import creator, base, tools

from .evaluate import *
from .constraint import *
from .evolve import *


def init_by_zero(binary=0):
    '''
    二进制编码
    :param binary: 0或1
    :return: binary
    '''
    return binary


def random_init(population, y_train, ratio):
    '''
    随机初始化种群
    :param population: 要进行初始化的种群
    :param y_train: 原始的训练集
    :param ratio: 平衡的比例（即初始实例数量的占比）
    :return: population
    '''
    # 使用 numpy.unique 获取类别、计数以及每个类别对应的索引
    unique_elements, counts = np.unique(y_train, return_counts=True)
    num_instances = int(np.ceil(counts.min() * ratio))
    # 构造每个类别的索引列表
    class_indices = {element: np.where(y_train == element)[0] for element in unique_elements}
    for i in range(len(population)):
        # 对于每个类，随机选择 num_instances 个不同的索引，生成一个新的dict
        select_class_indices = {}
        for index, item in enumerate(class_indices.items()):
            random_number = random.randint(num_instances, counts[index])  # 在num_instances和counts中对应的实例数量之间随机生成一个数字
            selected_indices = np.random.choice(item[1], random_number, replace=False)  # 选择不重复的索引
            select_class_indices[item[0]] = selected_indices
        for element in unique_elements:
            for indexs in select_class_indices[element]:
                population[i][indexs] = 1  # 将对应位置的二进制编码设置为1
    return population


def init_toolbox(estimator, x_train, y_train, weights_train, constraints, random_state, n_splits=5):
    '''
    将MILE中涉及到的算子，封装到deap库中的toolbox里
    :param y_train: 训练集标签
    :return: toolbox
    '''
    len_ind = len(y_train)  # 个体长度
    creator.create("FitnessMaxAndMax", base.Fitness, weights=(1.0, 1.0, 1.0))  # 最大化目标
    creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMaxAndMax, pfc=None, estimator=None,
                   y_sub_and_pred_proba=None, gmean=None, mauc=None, cv=None)  # 个体
    toolbox = base.Toolbox()
    toolbox.register("gene", init_by_zero)  # 0-1编码，基因全部初始化为0或1
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.gene, n=len_ind)  # 个体初始化
    toolbox.register("init_pop", random_init, y_train=y_train,
                     ratio=0.9)  # 初始化为平衡数据集（实例个数为min*0.9）
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # 种群初始化
    toolbox.register("objective_function", objective_function, weights_train=weights_train)  # 目标函数
    toolbox.register("evaluate", evaluate_individuals, estimator=estimator, x_train=x_train, y_train=y_train,
                     n_splits=n_splits,
                     random_state=random_state, weights_train=weights_train)  # 评价个体
    toolbox.register("mate", tools.cxOnePoint)  # 单点交叉
    toolbox.register("mutate", binary_inversion)  # 二进制突变
    toolbox.register("select", selNSGA2)  # NSGA-II选择（同一等级基于PFC选择）
    toolbox.register("get_feasible_infeasible", get_feasible_infeasible, constraints=constraints)  # 获取种群的可行个体与不可行个体
    toolbox.register("remove_duplicates", remove_duplicates)  # 去重
    toolbox.register("selTournamentNDCD", selTournamentNDCD)  # 锦标赛选择（同一等级基于PFC选择）
    return toolbox
