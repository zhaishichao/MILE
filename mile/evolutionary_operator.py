import math
import random
from itertools import chain
import array
from deap import tools, creator, base

from .multi_expert_evaluate import *
from mile.constraint_handing import *
from .guided_random_initialization import init_by_zero, random_init


# See deap.tools.selTournamentNDCD/selNSGA2
# These two algorithms have been rewritten here
# Mainly replacing crowding_dist with PFC
# Note: only the value is replaced, variable name remains crowding_dist
# This is done for easier invocation of the deap library

def binary_inversion(individual, mutation_rate=0.2):
    num_genes = len(individual)
    num_mutation = math.ceil(random.uniform(0.05, mutation_rate) * num_genes)
    indices = random.sample(range(num_genes), num_mutation)
    for index in indices:
        individual[index] ^= 1
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
    chosen = []
    for i in range(k):
        aspirants = tools.selRandom(individuals, tournsize)
        pareto_fronts = tools.sortNondominated(aspirants, len(aspirants))
        tools.emo.assignCrowdingDist(pareto_fronts[0])
        pareto_first_front = sorted(pareto_fronts[0], key=attrgetter("fitness.crowding_dist"),
                                    reverse=True)
        chosen.append(pareto_first_front[0])
    return chosen


def selNSGA2(individuals, k, nd='standard'):
    """
    NSGA-II algorithm based on PFC.
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

    assignCrowdingDist_PFC(individuals)  # Replace crowding_dist with PFC, but still name crowding_dist
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
        y_sub, y_pred_proba = ind.y_sub_and_pred_proba
        y_pred = np.argmax(y_pred_proba, axis=1)
        binary_sequence = [1 if y1 == y2 else 0 for y1, y2 in zip(y_sub, y_pred)]
        failpat.append(binary_sequence)

    accfailcred = [[0 for _ in range(len(individuals))] for _ in range(len(individuals))]

    for i in range(len(individuals) - 1):
        num_error_i = sum(1 for x in failpat[i] if x == 0)
        for j in range(i + 1, len(individuals)):
            hamming_distance = sum(x != y for x, y in zip(failpat[i], failpat[j]))  # Calculate Hamming distance
            num_error_j = sum(1 for x in failpat[j] if x == 0)
            if (num_error_i + num_error_j) > 0:
                accfailcred[i][j] = 1.0 * hamming_distance / (num_error_i + num_error_j)
            else:
                accfailcred[i][j] = 1.0 * hamming_distance
            accfailcred[j][i] = accfailcred[i][j]

    row_sum_accfailcred = [sum(row) for row in accfailcred]
    for i in range(len(individuals)):
        individuals[i].fitness.crowding_dist = 1.0 * row_sum_accfailcred[i] / (
                len(individuals) - 1)  # Use PFC instead of crowding distance, but still use crowding_dist naming to maintain consistency with crowding distance used in deap's built-in selNSGA2 algorithm


def init_toolbox(estimator, x_train, y_train, weights_train, constraints, random_state, n_splits=5):
    len_ind = len(y_train)
    creator.create("FitnessMaxAndMax", base.Fitness, weights=(1.0, 1.0, 1.0))
    creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMaxAndMax, pfc=None, estimator=None,
                   y_sub_and_pred_proba=None)
    toolbox = base.Toolbox()
    toolbox.register("gene", init_by_zero)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.gene, n=len_ind)
    toolbox.register("init_pop", random_init, y_train=y_train, ratio=0.9)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("objective_function", objective_function, weights_train=weights_train)
    toolbox.register("evaluate", evaluate_individual, estimator=estimator, x_train=x_train, y_train=y_train,
                     n_splits=n_splits, random_state=random_state, weights_train=weights_train)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", binary_inversion)
    toolbox.register("select", selNSGA2)
    toolbox.register("get_feasible_infeasible", get_feasible_infeasible, constraints=constraints)
    toolbox.register("remove_duplicates", remove_duplicates)
    toolbox.register("selTournamentNDCD", selTournamentNDCD)
    return toolbox
