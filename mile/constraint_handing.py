from operator import attrgetter


def cv(individual, constraints):
    ind_fitness = individual.fitness.values
    if len(ind_fitness) != len(constraints):
        raise ValueError("Constraints and individual fitness cannot be matched！")
    difference = [min(0, x - y) for x, y in zip(ind_fitness, constraints)]
    cv = sum(difference)
    individual.fitness.cv = cv
    return cv


def get_feasible_infeasible(pop, constraints):
    '''
    :param pop: 种群
    :param constraints: 约束阈值
    :return: 可行解和不可行解
    '''
    index = []
    for i in range(len(pop)):
        if cv(pop[i], constraints) != 0:  # Determine whether individual fitness satisfies the constraints
            index.append(i)
    feasible_pop = [ind for j, ind in enumerate(pop) if j not in index]  # feasible individual
    infeasible_pop = [ind for j, ind in enumerate(pop) if j in index]  # unfeasible individuals
    infeasible_pop = sorted(infeasible_pop, key=attrgetter("fitness.cv"),
                            reverse=True)  # Sort unfeasible individuals in descending order by cv value
    return feasible_pop, infeasible_pop


def remove_duplicates(pop, penalty_factor=0.0):
    n = len(pop)
    len_ind = len(pop[0])
    duplicates = []  # Index used to record duplicate pairs

    for i in range(n):
        duplicate = ()
        for j in range(i + 1, n):
            # Calculate the Hamming distance between pop [i] and pop [j]
            # (the number of unequal elements corresponding to two binary sequences)
            hamming_distance = sum(x != y for x, y in zip(pop[i], pop[j]))
            if 1.0 * hamming_distance / len_ind <= penalty_factor:
                duplicate = duplicate + (j,)
        duplicates.append(duplicate)

    to_remove = set()  # Only the indexes that appear afterwards are kept
    for duplicate in duplicates:
        to_remove.update(duplicate)
    return [pop[i] for i in range(len(pop)) if i not in to_remove], len(to_remove)
