class MILEConfig:
    '''
    Parameters of MILE.
    '''
    def __init__(self, ngen=30, popsize=30, cxpb=1.0, mutpb=0.2):
        self.NGEN = ngen # number of generations
        self.POPSIZE = popsize # population size
        self.CXPB = cxpb # probability of crossover
        self.MUTPB = mutpb # Probability of mutation


class DataSetConfig:
    '''
    Parameters of different datasets.
    '''
    def __init__(self, dataset_name, hidden_size, max_iter, learning_rate):
        self.DATASET_NAME = dataset_name
        self.HIDDEN_SIZE = hidden_size
        self.MAX_ITER = max_iter #  maximum number of iterations
        self.LEARNING_RATE = learning_rate #  learning rate of MLP

# benchmark datasets (from UCL and KEEL )