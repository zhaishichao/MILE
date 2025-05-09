import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.base import clone
import scipy.io as sio  # 从.mat文件中读取数据集
from metrics import calculate_expert_accuracy


def get_distribution(y):
    # Use numpy.unique to get categories, counts, and indexes corresponding to each category
    unique_elements, counts = np.unique(y, return_counts=True)
    # Create an indexed list for each category
    class_indices = []
    for element in unique_elements:
        class_indices.append(np.where(y == element)[0])
    return unique_elements, class_indices, counts


def k_fold_cross_validation(estimator, x, y, n_splits=5, method='soft', random_state=42):
    """
    Perform 5-fold cross-validation and generate soft labels (probability predictions).

    Parameters:
    - model: A sklearn-compatible model with a `predict_proba` method.
    - x: Feature matrix (numpy array or pandas DataFrame).
    - y: Target vector (numpy array or pandas Series).
    - n_splits:k-fold cross validation
    - method: 'soft' or 'hard'

    Returns:
    - soft_labels: A numpy array containing the soft labels for each sample.
    - scores: A list of accuracy scores for each fold.
    """
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)  # 5-fold cross-validation
    soft_labels = np.zeros((len(y), len(np.unique(y))))  # Initialize array for soft labels
    for train_index, test_index in kf.split(x, y):
        # Split datasets into train and test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Clone and fit the model on the training set
        estimator_clone = clone(estimator)
        estimator_clone.fit(x_train, y_train)
        # Generate soft labels (probability predictions)
        y_proba = estimator_clone.predict_proba(x_test)
        soft_labels[test_index] = y_proba
    if method == 'soft':
        return soft_labels
    elif method == 'hard':
        hard_labels = np.argmax(y_proba, axis=1)
        return hard_labels
    else:
        raise ValueError("Invalid method. Choose 'soft' or 'hard'.")


def pre_processing(n_splits, file_path=None, estimator=None, display_distribution=False, random_state=42):
    mat_data = sio.loadmat(file_path)  # 加载、划分数据集
    x = mat_data['X']
    y = mat_data['Y'][:, 0]  # mat_data['Y']得到的形状为[n,1]，通过[:,0]，得到形状[n,]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y,
                                                        random_state=42)  # 划分数据集
    unique_elements_all, classes_all, counts_all = get_distribution(y)  # 获取原始数据集分布
    unique_elements_train, classes_train, counts_train = get_distribution(y_train)  # 获取训练集分布
    unique_elements_test, classes_test, counts_test = get_distribution(y_test)  # 获取测试集分布
    weights_train = (1 / counts_train.astype(float)) / np.sum(
        1 / counts_train.astype(float))  # 计算每个类的权重，用于计算每个类别的权重
    if display_distribution:
        print(f'distribution: {counts_all}')
        print(f'trainset distribution: {counts_train}')
        print(f'testset distribution: {counts_test}')
    y_train_pred_proba = k_fold_cross_validation(estimator=estimator, x=x_train, y=y_train, n_splits=n_splits,
                                                 method='soft',
                                                 random_state=random_state)  # 交叉验证得到软标签
    y_train_pred = np.argmax(y_train_pred_proba, axis=1)  # 将概率转化为预测结果
    Acc1, Acc2, Acc3 = calculate_expert_accuracy(y_train_pred, y_train, weights_train)
    constraints = [Acc1, Acc2, Acc3]
    return x_train, x_test, y_train, y_test, constraints, weights_train


def get_indices(individual):
    '''
    :param individual: individual（用二进制或0-1范围内的实值进行编码）
    :return: 被选择实例的索引
    '''
    indices = np.where(individual == 1)  # 1代表选择该实例，返回值是tuple，tuple[0]取元组中的第一个元素
    return indices[0]


def get_subset(individual, x, y):
    '''
    :param individual:
    :return: 实例子集
    '''
    indices = get_indices(individual)
    x_sub = x[indices, :]
    y_sub = y[indices]
    return x_sub, y_sub