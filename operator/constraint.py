def cv(individual, constraints):
    '''
    计算个体的违约程度 (Degree of constraint violation) 简称cv
    :param individual: 个体
    :param constraints: 约束条件（三个约束阈值）
    :return: 约束违反程度和
    '''
    ind_fitness = individual.fitness.values
    if len(ind_fitness) != len(constraints):
        raise ValueError("约束条件和个体适应度无法匹配！")
    difference = [min(0, x - y) for x, y in zip(ind_fitness, constraints)]  # 判断个体三个专家指标是否都大于等于约束条件
    cv = sum(difference)  # 求0和cv中的最小值之和，cv=0，表示是一个可行解
    individual.cv = cv  # 将cv值保存在个体中
    return cv

