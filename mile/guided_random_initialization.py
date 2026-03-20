import random

import numpy as np


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
