from operator import attrgetter

def cv(individual, constraints):
    '''
    计算个体的违约程度 (Degree of constraint violation) 简称cv
    :param individual: 个体
    :param constraints: 约束条件（三个约束阈值）
    :return: 约束违反程度
    '''
    ind_fitness = individual.fitness.values
    if len(ind_fitness) != len(constraints):
        raise ValueError("约束条件和个体适应度无法匹配！")
    difference = [min(0, x - y) for x, y in zip(ind_fitness, constraints)]  # 判断个体三个专家指标是否都大于等于约束条件
    cv = sum(difference)  # 求0和cv中的最小值之和，cv=0，表示是一个可行个体
    individual.fitness.cv = cv  # 将cv值保存在个体中
    return cv


def get_feasible_infeasible(pop, constraints):
    '''
    :param pop: 种群
    :param constraints: 约束阈值
    :return: 可行解和不可行解
    '''
    index = []
    for i in range(len(pop)):
        if cv(pop[i], constraints) != 0:  # 判断个体适应度是否都满足约束条件
            index.append(i)  # 将不符合约束条件的个体的索引添加到index中
    feasible_pop = [ind for j, ind in enumerate(pop) if j not in index]  # 可行个体
    infeasible_pop = [ind for j, ind in enumerate(pop) if j in index]  # 不可行个体
    infeasible_pop = sorted(infeasible_pop, key=attrgetter("fitness.cv"), reverse=True)  # 对不可行个体按cv值降序排序
    return feasible_pop, infeasible_pop

def remove_duplicates(pop, penalty_factor=0.0):
    '''
    :param pop: 要操作的种群对象 array.array
    :param penalty_factor: 差异的惩罚因子（即基因重复率在0-penalty_factor区间内，均视为重复个体）
    :return: 去重后的pop
    '''
    n = len(pop)
    len_ind = len(pop[0])
    duplicates = []  # 用于记录重复对的索引

    for i in range(n):
        duplicate = ()
        for j in range(i + 1, n):
            # 计算pop[i]、pop[j]之间的汉明距离（两个二进制序列对应元素不相等的个数）
            hamming_distance = sum(x != y for x, y in zip(pop[i], pop[j]))
            if 1.0 * hamming_distance / len_ind <= penalty_factor:
                duplicate = duplicate + (j,)
        duplicates.append(duplicate)
    # 找到所有需要移除的索引
    to_remove = set()  # 只保留后出现的索引
    for duplicate in duplicates:
        to_remove.update(duplicate)  # update是用来更新set集合的
    return [pop[i] for i in range(len(pop)) if i not in to_remove], len(to_remove)
