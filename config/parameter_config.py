class MILEConfig:
    '''
    MILE算法参数
    '''
    def __init__(self, ngen=30, popsize=30, cxpb=1.0, mr=0.2):
        self.NGEN = ngen #  迭代次数
        self.POPSIZE = popsize # 种群大小
        self.CXPB = cxpb # 交叉概率
        self.MR = mr # 变异概率


class DataSetConfig:
    '''
    数据集参数
    '''
    def __init__(self, dataset_name, hidden_size, max_iter, learning_rate):
        self.DATASETNAME = dataset_name #  数据集名称
        self.HIDDEN_SIZE = hidden_size #  隐藏层大小
        self.MAX_ITER = max_iter #  最大迭代次数
        self.LEARNING_RATE = learning_rate #  学习率

# 数据集配置
BalanceScale = DataSetConfig('BalanceScale.mat', 15, 500, 0.1) #  BalanceScale