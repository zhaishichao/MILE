'''
Configuration information for the dataset.
'''
class DataSetConfig: # DataSetConfig
    def __init__(self, dataset_name, hidden_size, max_iter, learning_rate):
        self.DATASETNAME = dataset_name #  DataSetName
        self.HIDDEN_SIZE = hidden_size #  HiddenSize
        self.MAX_ITER = max_iter #  MaxIter
        self.LEARNING_RATE = learning_rate #  LearningRate


BalanceScale = DataSetConfig('BalanceScale.mat', 15, 500, 0.1) #  BalanceScale