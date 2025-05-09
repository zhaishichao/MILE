from sklearn.base import clone
from sklearn.neural_network import MLPClassifier
import random
from config import BalanceScale
from evolutionary_operator import init_toolbox
from utils import pre_processing


class MILE():
    """Multi-Expert Ensemble with Instance Selection for Imbalanced Learning.
    Parameters
    ----------
    dataset: tuple
        A tuple containing feature data x and label y -> (x, y),
        x cannot contain a string, y must be an array of 0, 1, 2, 3...
    estimator : estimator
        The base estimator from which the ensemble is grown.
    random_state: int
        Determines the partitioning of the dataset
        and the subset of instances at random initialization.
    n_splits: int
        k折交叉验证中的k
    display_distribution: bool
        是否显示数据的分布信息
    """

    def __init__(self, file_path=None, estimator=None, random_state=42, n_splits=5, display_distribution=True, parameter=None):
        self.file_path = file_path
        self.estimator = estimator
        self.random_state = random_state
        self.n_splits = n_splits
        self.x_train, self.x_test, self.y_train, self.y_test, self.constraints, self.weights_train = pre_processing(
            self.n_splits, display_distribution, self.file_path, self.estimator, self.random_state)
        self.toolbox = init_toolbox(self.estimator, self.x_train, self.y_train, self.weights_train, self.constraints,
                                    self.n_splits, self.random_state)
        self.parameter=parameter

    def fit(self, ):
        ####################################种群的初始化###########################
        pop = self.toolbox.population(n=self.parameter.POPSIZE)  # 个体编码默认全为0
        pop = self.toolbox.init_pop(pop)  # 初始化种群中的个体
        self.toolbox.evaluate(pop)  # 计算个体的适应度
        ####################################种群的迭代#################################################
        for gen in range(0, self.parameter.NGEN):
            offspring = self.toolbox.selTournamentNDCD(pop, self.parameter.POPSIZE, tournsize=3)  # 锦标赛选择（1、先根据非支配排序的等级2、再根据拥挤距离）
            offspring = [self.toolbox.clone(ind) for ind in offspring]
            for i in range(0, len(offspring) - 1, 2):
                if random.random() <= self.parameter.CXPB:
                    offspring[i], offspring[i + 1] = self.toolbox.mate(offspring[i], offspring[i + 1])  # 单点交叉
                offspring[i] = self.toolbox.mutate(offspring[i], self.parameter.MR)[0]  # 二进制反转突变
                offspring[i + 1] = self.toolbox.mutate(offspring[i + 1], self.parameter.MR)[0]  # 二进制反转突变
                del offspring[i].fitness.values, offspring[i + 1].fitness.values
            #############################################################合并、去重#####################################################
            offspring = self.toolbox.individuals_constraints(offspring)  # 限制每个类至少有一个实例被选择
            pop = pop + offspring  # 种群的合并
            pop, _ = self.toolbox.remove_duplicates(pop)  # 去重
            while len(pop) < self.parameter.POPSIZE:  # 保证种群大小为POPSIZE
                add_individual = []
                num_add = self.parameter.POPSIZE - len(pop)
                for i in range(0, num_add):
                    index = random.randint(0, len(offspring) - 1)  # 在0-len(offspring)范围内随机产生一个索引
                    offspring[index] = self.toolbox.mutate(offspring[index], self.parameter.MR)[0]  # 选择index对应的个体进行突变
                    del offspring[index].fitness.values
                    add_individual.append(offspring[index])
                add_individual = self.toolbox.individuals_constraints(add_individual)  # 限制每个类至少有一个实例被选择
                pop = pop + add_individual  # 种群的合并
                pop, _ = self.toolbox.remove_duplicates(pop)  # 去重
            self.toolbox.evaluate(pop)  # 计算新种群适应度
            ###############################################得到pareto_fronts############################################
            feasible_pop, infeasible_pop = self.toolbox.get_feasible_infeasible(pop)  # 得到可行解与不可行解
            if len(feasible_pop) >= self.parameter.POPSIZE:
                pop = self.toolbox.select(feasible_pop, self.parameter.POPSIZE)
                ensembles = pop  # pop均为可行解，则集成pop中所有个体
            elif len(feasible_pop) > 0:
                pop = feasible_pop + infeasible_pop[:self.parameter.POPSIZE - len(feasible_pop)]  # 在不可行解中选取违约程度小的个体，保证pop数量为POPSIZE
                ensembles = feasible_pop  # 只集成可行解
            else:
                pop = feasible_pop + infeasible_pop[:self.parameter.POPSIZE - len(feasible_pop)]  # 加入不可行解中违约程度小的个体，保证pop数量为POPSIZE
                ensembles = [infeasible_pop[0]]  # 没有可行解，集成不可行解中第一个个体
            return ensembles


if __name__ == '__main__':
    IMBALANCED_DATASET_PATH = '../datasets/mat/'
    DATASET = BalanceScale  # 数据集名称（包含对应的参数配置）
    file_path = IMBALANCED_DATASET_PATH + DATASET.DATASETNAME
    name = BalanceScale.DATASETNAME.split('.')[0]
    model = MLPClassifier(hidden_layer_sizes=(DATASET.HIDDEN_SIZE,), max_iter=DATASET.MAX_ITER,
                          random_state=42, learning_rate_init=DATASET.LEARNING_RATE)
    mile = MILE(file_path=file_path, estimator=model, random_state=42)
