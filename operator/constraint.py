from _operator import attrgetter


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
    cv = sum(difference)  # 求0和cv中的最小值之和，cv=0，表示是一个可行解
    individual.cv = cv  # 将cv值保存在个体中
    return cv


def get_feasible_infeasible(pop, constraints):
    index = []
    for i in range(len(pop)):
        if cv(pop[i], constraints) != 0:  # 判断个体适应度是否都满足约束条件
            index.append(i)  # 将不符合约束条件的个体的索引添加到index中
    feasible_pop = [ind for j, ind in enumerate(pop) if j not in index]  # 可行解
    infeasible_pop = [ind for j, ind in enumerate(pop) if j in index]  # 不可行解
    infeasible_pop = sorted(infeasible_pop, key=attrgetter("individual.cv"), reverse=True)  # 对不可行解按cv值降序排序
    return feasible_pop, infeasible_pop
