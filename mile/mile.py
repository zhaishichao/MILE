import numpy as np
import random
from sklearn.neural_network import MLPClassifier
from config import BalanceScale, MILEConfig
from evolutionary_operator import init_toolbox, ensemble_individuals, vote_result_ensembles
from metrics import calculate_gmean_mauc
from utils import pre_processing
import warnings
warnings.filterwarnings("ignore")  # 忽略警告


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

    def __init__(self, file_path=None, estimator=None, random_state=42, n_splits=5, display_distribution=True,
                 parameter=None):
        self.file_path = file_path
        self.estimator = estimator
        self.random_state = random_state
        self.n_splits = n_splits
        self.x_train, self.x_test, self.y_train, self.y_test, self.constraints, self.weights_train = pre_processing(
            self.n_splits, display_distribution, self.random_state, self.file_path, self.estimator)
        self.toolbox = init_toolbox(self.estimator, self.x_train, self.y_train, self.weights_train, self.constraints,
                                    self.random_state, self.n_splits)
        self.parameter = parameter
        self.ensemble_classifiers = None

    def fit(self):
        '''
        实例子集的搜索，获取分类器集合
        :return: void
        '''
        # 种群的初始化
        pop = self.toolbox.population(n=self.parameter.POPSIZE)
        pop = self.toolbox.init_pop(pop)  # 初始化种群中的个体
        self.toolbox.evaluate(pop)  # 评估初始种群
        # 实例子集进化搜索
        for gen in range(0, self.parameter.NGEN):
            # 选择、交叉、变异
            offspring = self.toolbox.selTournamentNDCD(pop, self.parameter.POPSIZE, tournsize=3)  # 1、先由非支配排序的等级 2、再由PFC
            offspring = [self.toolbox.clone(ind) for ind in offspring]
            for i in range(0, len(offspring) - 1, 2):
                if random.random() <= self.parameter.CXPB:
                    offspring[i], offspring[i + 1] = self.toolbox.mate(offspring[i], offspring[i + 1])  # 单点交叉
                offspring[i] = self.toolbox.mutate(offspring[i], self.parameter.MUTPB)[0]  # 二进制反转
                offspring[i + 1] = self.toolbox.mutate(offspring[i + 1], self.parameter.MUTPB)[0]  # 二进制反转
                del offspring[i].fitness.values, offspring[i + 1].fitness.values
            # 合并、去重
            pop = pop + offspring  # 种群的合并
            pop, _ = self.toolbox.remove_duplicates(pop)  # 去重
            # 保证种群大小为POPSIZE，避免因重复过多而造成的种群数量减少
            while len(pop) < self.parameter.POPSIZE:
                add_individual = []
                num_add = self.parameter.POPSIZE - len(pop)  # 需要添加的个体数量
                # 随机选择一个个体进行突变
                for i in range(0, num_add):
                    index = random.randint(0, len(offspring) - 1)
                    offspring[index] = self.toolbox.mutate(offspring[index], self.parameter.MUTPB)[0]  # 选择index对应的个体进行突变
                    del offspring[index].fitness.values
                    add_individual.append(offspring[index])
                pop = pop + add_individual  # 再次合并
                pop, _ = self.toolbox.remove_duplicates(pop)  # 再次去重
                pop = self.toolbox.individuals_constraints(pop)  # 限制每个类至少有5个实例被选择
            self.toolbox.evaluate(pop)  # 评估新种群
            feasible_pop, infeasible_pop = self.toolbox.get_feasible_infeasible(pop)  # 得到可行个体与不可行个体
            if len(feasible_pop) >= self.parameter.POPSIZE:
                pop = self.toolbox.select(feasible_pop, self.parameter.POPSIZE)
                ensembles = pop  # pop均为可行个体，则集成pop中所有个体
            elif len(feasible_pop) > 0:
                pop = feasible_pop + infeasible_pop[
                                     :self.parameter.POPSIZE - len(feasible_pop)]  # 在不可行个体中选取违约程度小的个体，保证pop数量为POPSIZE
                ensembles = feasible_pop  # 只集成可行个体
            else:
                pop = feasible_pop + infeasible_pop[
                                     :self.parameter.POPSIZE - len(feasible_pop)]  # 加入不可行个体中违约程度小的个体，保证pop数量为POPSIZE
                ensembles = [infeasible_pop[0]]  # 没有可行个体，集成不可行个体中第一个个体
            self.ensemble_classifiers = ensemble_individuals(ensembles, self.estimator, self.x_train, self.y_train)

    def predict(self, x_test):
        '''
        集成预测
        :param x_test: 测试集特征数据
        :return: 集成预测结果
        '''
        y_pred = vote_result_ensembles(self.ensemble_classifiers, x_test, result='hard')
        return y_pred

    def predict_proba(self, x_test):
        '''
        集成预测概率
        :param x_test: 测试集特征数据
        :return: 集成预测概率
        '''
        y_pred_prob = vote_result_ensembles(self.ensemble_classifiers, x_test, result='soft')
        return y_pred_prob


if __name__ == '__main__':
    mile_parameter = MILEConfig()
    IMBALANCED_DATASET_PATH = '../datasets/mat/'
    DATASET = BalanceScale  # 数据集名称（包含对应的参数配置）
    file_path = IMBALANCED_DATASET_PATH + DATASET.DATASET_NAME
    name = BalanceScale.DATASET_NAME.split('.')[0]
    mlp = MLPClassifier(hidden_layer_sizes=(DATASET.HIDDEN_SIZE,), max_iter=DATASET.MAX_ITER,
                        random_state=42, learning_rate_init=DATASET.LEARNING_RATE)

    mile = MILE(file_path=file_path, estimator=mlp, random_state=42, n_splits=5, display_distribution=True,
                parameter=mile_parameter)

    num_run = 40
    ensembles_results = []
    for i in range(0, num_run):
        mile.fit()
        y_pred = mile.predict(mile.x_test)
        y_pred_prob = mile.predict_proba(mile.x_test)
        gmean, mauc = calculate_gmean_mauc(y_pred_prob, mile.y_test)
        ensembles_results.append([gmean, mauc])
        print(f"第{i+1}次运行：Gmean：{gmean}，mAUC：{mauc}")
    ensembles_result_mean = np.mean(ensembles_results, axis=0)
    ensembles_result_std = np.std(ensembles_results, axis=0)
    print(f'集成分类结果（平均值）：{ensembles_result_mean}')
    print(f'集成分类结果（标准差）：{ensembles_result_std}')
