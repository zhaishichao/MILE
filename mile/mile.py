from sklearn.base import clone
from sklearn.neural_network import MLPClassifier

from config import BalanceScale
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
    """

    def __init__(self, file_path=None, estimator=None, random_state=42,n_splits=5):
        self.file_path = file_path
        self.estimator = clone(estimator)
        self.random_state = random_state
        self.n_splits = n_splits
        self.x_train, self.x_test, self.y_train, self.y_test, self.constraints, self.weights_train = pre_processing(file_path=self.file_path,
            estimator=estimator, display_distribution=True, random_state=random_state,n_splits=self.n_splits)

    def fit(self):
        a=1



if __name__ == '__main__':
    IMBALANCED_DATASET_PATH = '../datasets/mat/'
    DATASET = BalanceScale  # 数据集名称（包含对应的参数配置）
    file_path= IMBALANCED_DATASET_PATH + DATASET.DATASETNAME
    name = BalanceScale.DATASETNAME.split('.')[0]
    model = MLPClassifier(hidden_layer_sizes=(DATASET.HIDDEN_SIZE,), max_iter=DATASET.MAX_ITER,
                          random_state=42, learning_rate_init=DATASET.LEARNING_RATE)
    mile = MILE(file_path=file_path, estimator=model, random_state=42)
