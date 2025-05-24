import math
import random
from itertools import chain
from operator import attrgetter

import numpy as np
from deap import tools

# 参见 deap.tools.selTournamentNDCD/selNSGA2
# 这里对这两个算法进行了重写
# 主要是将crowding_dist拥挤距离更换为PFC
# 注意只是值的替换，变量名称仍未crowding_dist
# 这么做是为了便于调用deap库

def binary_inversion(individual, mutation_rate=0.2):
    num_genes = len(individual)  # 基因总数
    num_mutation = math.ceil(random.uniform(0.05, mutation_rate) * num_genes)  # 突变的总数 (math.ceil向上取整)
    indices = random.sample(range(num_genes), num_mutation)  # 在num_genes个基因中随机采样num_mutation个
    for index in indices:
        individual[index] ^= 1  # 将对应位置的二进制编码取反 (按位异或) 1^1->1, 0^1->1
    return individual,


def selTournamentNDCD(individuals, k, tournsize):
    """Select the best individual among *tournsize* randomly chosen
    individuals, *k* times. The list returned contains
    references to the input *individuals*.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param tournsize: The number of individuals participating in each tournament.
    :param fit_attr: The attribute of individuals to use as selection criterion
    :returns: A list of selected individuals.

    This function uses the :func:`~random.choice` function from the python base
    :mod:`random` module.
    """
    # 先做非支配排序，再根据选择支配等级进行选择
    chosen = []
    for i in range(k):
        aspirants = tools.selRandom(individuals, tournsize)  # 随机选择tournsize个个体
        pareto_fronts = tools.sortNondominated(aspirants, len(aspirants))  # 进行非支配排序
        tools.emo.assignCrowdingDist(pareto_fronts[0])
        pareto_first_front = sorted(pareto_fronts[0], key=attrgetter("fitness.crowding_dist"),
                                    reverse=True)  # 按拥挤度降序排列
        chosen.append(pareto_first_front[0])  # 选择第一个等级中拥挤度最大的
    return chosen


def selNSGA2(individuals, k, nd='standard'):
    """
    基于PFC的NSGAII算法
    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param nd: Specify the non-dominated algorithm to use: 'standard' or 'log'.
    :returns: A list of selected individuals.
    .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
       non-dominated sorting genetic algorithm for multi_objective
       optimization: NSGA-II", 2002.
    """
    if nd == 'standard':
        pareto_fronts = tools.sortNondominated(individuals, k)
    elif nd == 'log':
        pareto_fronts = tools.sortLogNondominated(individuals, k)
    else:
        raise Exception('selNSGA2: The choice of non-dominated sorting '
                        'method "{0}" is invalid.'.format(nd))

    assignCrowdingDist_PFC(individuals)
    chosen = list(chain(*pareto_fronts[:-1]))
    k = k - len(chosen)
    if k > 0:
        sorted_front = sorted(pareto_fronts[-1], key=attrgetter("fitness.crowding_dist"), reverse=True)
        chosen.extend(sorted_front[:k])

    return chosen


def assignCrowdingDist_PFC(individuals):
    """
    用PFC替换掉crowding_dist
    Assign a crowding distance to each individual's fitness. The
    crowding distance can be retrieve via the :attr:`crowding_dist`
    attribute of each individual's fitness.
    """
    if len(individuals) == 0:
        return
    failpat = []
    for ind in individuals:
        y_sub, y_pred_proba = ind.y_sub_and_pred_proba  # 训练子集和对应的预测软标签
        y_pred = np.argmax(y_pred_proba, axis=1)  # 分类器预测结果
        binary_sequence = [1 if y1 == y2 else 0 for y1, y2 in zip(y_sub, y_pred)]  # 0表示对应实例预测错误，1表示预测正确
        failpat.append(binary_sequence)

    accfailcred = [[0 for _ in range(len(individuals))] for _ in range(len(individuals))]  # 记录成对的错误预测次数
    # 计算成对的错误预测次数
    for i in range(len(individuals) - 1):
        num_error_i = sum(1 for x in failpat[i] if x == 0)
        for j in range(i + 1, len(individuals)):
            hamming_distance = sum(x != y for x, y in zip(failpat[i], failpat[j]))  # 计算汉明距离
            num_error_j = sum(1 for x in failpat[j] if x == 0)
            if (num_error_i + num_error_j) > 0:
                accfailcred[i][j] = 1.0 * hamming_distance / (num_error_i + num_error_j)
            else:
                accfailcred[i][j] = 1.0 * hamming_distance
            accfailcred[j][i] = accfailcred[i][j]

    row_sum_accfailcred = [sum(row) for row in accfailcred]  # 对每一行求和
    for i in range(len(individuals)):
        individuals[i].fitness.crowding_dist = 1.0 * row_sum_accfailcred[i] / (
                len(individuals) - 1)  # 使用PFC代替拥挤距离，但仍使用crowding_dist命名，目的是为了与deap内置的selNSGA2算法中使用的拥挤距离保持一致
