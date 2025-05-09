import array

from deap import creator, base, tools

from operator import init_by_zero, random_init, objective_function, evaluate_individuals


def init_toolbox(estimator, x_train, y_train, weights_train, constraints, n_splits=5, random_seed=42):
    '''
    将MILE中涉及到的算子，封装到deap库中的toolbox里
    :param y_train: 训练集标签
    :return: toolbox
    '''
    len_ind = len(y_train)  # 个体长度
    creator.create("FitnessMaxAndMax", base.Fitness, weights=(1.0, 1.0, 1.0))  # 最大化目标
    creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMaxAndMax, pfc=None, model=None,
                   y_sub_and_pred_proba=None, gmean=None, mauc=None, cv=None)  # 个体
    toolbox = base.Toolbox()
    toolbox.register("gene", init_by_zero)  # 0-1编码，基因全部初始化为0或1
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.gene, n=len_ind)  # 个体初始化
    toolbox.register("init_population", random_init, y_train=y_train,
                     ratio=0.9)  # 初始化为平衡数据集（实例个数为min*0.9）
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # 种群初始化

    toolbox.register("objective_function", objective_function, weights_train=weights_train)
    toolbox.register("evaluate", evaluate_individuals, estimator=estimator, x_train=x_train, y_train=y_train, n_splits=n_splits,
                     random_seed=random_seed, fitness_function=toolbox.objective_function)  # 评价个体
    return toolbox
