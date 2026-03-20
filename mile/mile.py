import random

from .ensemble_of_instance_subsets import ensemble_individuals, vote_result_ensembles
from .evolutionary_operator import init_toolbox
from utils import pre_processing
import warnings
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")  # Ignore warnings


class MILE():
    """
    Multi-Expert Ensemble with Instance Selection for Imbalanced Learning.
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
        Search for instance subsets and obtain classifier ensemble.
        :return: void
        '''
        pop = self.toolbox.population(n=self.parameter.POPSIZE)
        pop = self.toolbox.init_pop(pop)
        with ProcessPoolExecutor(max_workers=8) as pool:  # Evaluate initial population with multi-process
            result = list(pool.map(self.toolbox.evaluate, pop))
        for i, ind in enumerate(pop):
            if result[i]["valid"]:
                ind.y_sub_and_pred_proba = result[i]["y_sub_and_pred_proba"]  # 保存个体的软标签和预测概率
                ind.fitness.values = result[i]["fitness"]

        # Evolutionary search
        for gen in range(0, self.parameter.NGEN):
            # Selection, crossover, mutation
            offspring = self.toolbox.selTournamentNDCD(pop, self.parameter.POPSIZE, tournsize=3)  # 1. Non-dominated sorting rank 2. Then PFC
            offspring = [self.toolbox.clone(ind) for ind in offspring]
            for i in range(0, len(offspring) - 1, 2):
                if random.random() <= self.parameter.CXPB:
                    offspring[i], offspring[i + 1] = self.toolbox.mate(offspring[i], offspring[i + 1])
                offspring[i] = self.toolbox.mutate(offspring[i], self.parameter.MUTPB)[0]
                offspring[i + 1] = self.toolbox.mutate(offspring[i + 1], self.parameter.MUTPB)[0]
                del offspring[i].fitness.values, offspring[i + 1].fitness.values

            pop = pop + offspring
            pop, _ = self.toolbox.remove_duplicates(pop)  # Remove duplicates

            # Ensure population size is POPSIZE...
            while len(pop) < self.parameter.POPSIZE:
                add_individual = []
                num_add = self.parameter.POPSIZE - len(pop)  # 需要添加的个体数量
                # 随机选择一个个体进行突变
                for i in range(0, num_add):
                    index = random.randint(0, len(offspring) - 1)
                    offspring[index] = self.toolbox.mutate(offspring[index], self.parameter.MUTPB)[0]  # 选择index对应的个体进行突变
                    del offspring[index].fitness.values
                    add_individual.append(offspring[index])
                pop = pop + add_individual
                pop, _ = self.toolbox.remove_duplicates(pop)

            with ProcessPoolExecutor(max_workers=8) as pool:
                result = list(pool.map(self.toolbox.evaluate, pop))
            for i, ind in enumerate(pop):
                if result[i]["valid"]:
                    ind.y_sub_and_pred_proba = result[i]["y_sub_and_pred_proba"]
                    ind.fitness.values = result[i]["fitness"]

            # Constraint handling
            feasible_pop, infeasible_pop = self.toolbox.get_feasible_infeasible(pop)
            if len(feasible_pop) >= self.parameter.POPSIZE:
                pop = self.toolbox.select(feasible_pop, self.parameter.POPSIZE)
                ensembles = pop
            elif len(feasible_pop) > 0:
                pop = feasible_pop + infeasible_pop[
                                     :self.parameter.POPSIZE - len(feasible_pop)]
                ensembles = feasible_pop
            else:
                pop = feasible_pop + infeasible_pop[
                                     :self.parameter.POPSIZE - len(feasible_pop)]
                ensembles = [infeasible_pop[0]]
            self.ensemble_classifiers = ensemble_individuals(ensembles, self.estimator, self.x_train, self.y_train)

    def predict(self, x_test):
        '''
        :param x_test: Test set
        :return: Prediction results
        '''
        y_pred = vote_result_ensembles(self.ensemble_classifiers, x_test, result='hard')
        return y_pred
    def predict_proba(self, x_test):
        '''
        :param x_test: Test set
        :return: Prediction results
        '''
        y_pred_prob = vote_result_ensembles(self.ensemble_classifiers, x_test, result='soft')
        return y_pred_prob
