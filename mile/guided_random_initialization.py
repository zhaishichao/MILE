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
    Guided random initialization
    '''
    # Use numpy.unique to get categories, counts, and indexes for each category
    unique_elements, counts = np.unique(y_train, return_counts=True)
    num_instances = int(np.ceil(counts.min() * ratio))
    # Construct a list of indexes for each category
    class_indices = {element: np.where(y_train == element)[0] for element in unique_elements}
    for i in range(len(population)):
        # For each class, randomly select num_instances of different indexes to generate a new dict
        select_class_indices = {}
        for index, item in enumerate(class_indices.items()):
            random_number = random.randint(num_instances, counts[index])  # Generates a number randomly between the corresponding number of instances in num_instances and counts
            selected_indices = np.random.choice(item[1], random_number, replace=False)  # Selects unique indices without replacement
            select_class_indices[item[0]] = selected_indices
        for element in unique_elements:
            for indexs in select_class_indices[element]:
                population[i][indexs] = 1  # Set the binary encoding of the corresponding position to 1
    return population
