
class DataSetConfig: # DataSetConfig
    '''
    Configuration information for the dataset.
    '''
    def __init__(self, dataset_name, hidden_size, max_iter, learning_rate):
        self.DATASETNAME = dataset_name #  数据集名称
        self.HIDDEN_SIZE = hidden_size #  隐藏层大小
        self.MAX_ITER = max_iter #  最大迭代次数
        self.LEARNING_RATE = learning_rate #  学习率

# 数据集配置
BalanceScale = DataSetConfig('BalanceScale.mat', 15, 500, 0.1) #  BalanceScale