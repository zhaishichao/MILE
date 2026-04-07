import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.base import clone
import scipy.io as sio
from sklearn.preprocessing import StandardScaler

from metrics import calculate_expert_accuracy


def get_distribution(y):
    # Use numpy.unique to get categories, counts, and indexes corresponding to each category
    unique_elements, counts = np.unique(y, return_counts=True)
    # Create an indexed list for each category
    class_indices = []
    for element in unique_elements:
        class_indices.append(np.where(y == element)[0])
    return unique_elements, class_indices, counts


def k_fold_cross_validation(estimator, x, y, random_state, n_splits=5, method='soft'):
    """
    Perform 5-fold cross-validation and generate soft labels (probability predictions).

    Parameters:
    - estimator: A sklearn-compatible estimator with a [predict_proba]
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
        # Clone and fit the estimator on the training set
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


def pre_processing(n_splits, display_distribution, random_state, file_path=None, estimator=None):
    mat_data = sio.loadmat(file_path)  # Load and split dataset
    x = mat_data['X']
    y = mat_data['Y'][:, 0]  # mat_data['Y'] has shape [n,1], use [:,0] to get shape [n,]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y,
                                                        random_state=random_state)  # Split dataset
    # Data standardization
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    unique_elements_all, classes_all, counts_all = get_distribution(y)  # Get original dataset distribution
    unique_elements_train, classes_train, counts_train = get_distribution(y_train)  # Get training set distribution
    unique_elements_test, classes_test, counts_test = get_distribution(y_test)  # Get test set distribution
    weights_train = (1 / counts_train.astype(float)) / np.sum(
        1 / counts_train.astype(float))  # Calculate weight for each class
    if display_distribution:
        print(f'distribution: {counts_all}')
        print(f'trainset distribution: {counts_train}')
        print(f'testset distribution: {counts_test}')
    y_train_pred_proba = k_fold_cross_validation(estimator=estimator, x=x_train, y=y_train, random_state=random_state,
                                                 n_splits=n_splits,
                                                 method='soft')  # Get soft labels through cross-validation
    y_train_pred = np.argmax(y_train_pred_proba, axis=1)  # Convert probabilities to predictions
    Acc1, Acc2, Acc3 = calculate_expert_accuracy(y_train_pred, y_train, weights_train)
    constraints = [Acc1, Acc2, Acc3]
    return x_train, x_test, y_train, y_test, constraints, weights_train


def get_indices(individual):
    '''
    :param individual: individual (encoded with binary or real values in 0-1 range)
    :return: Indices of selected instances
    '''
    # Convert individual to ndarray
    individual = np.array(individual)
    indices = np.where(
        individual == 1)  # 1 means the instance is selected, return value is tuple, tuple[0] gets the first element
    return indices[0]


def get_subset(individual, x, y):
    '''
    :param Individual:
    :return: Instance subset
    '''
    indices = get_indices(individual)
    x_sub = x[indices, :]
    y_sub = y[indices]
    return x_sub, y_sub
