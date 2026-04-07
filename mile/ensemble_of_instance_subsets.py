import numpy as np
from scipy import stats
from sklearn.base import clone

from utils import get_subset


def ensemble_individuals(individuals, estimator, x_train, y_train):
    ensembles = []
    # Traverse each individual, obtain subsets and labels, and train the classifier
    for ind in individuals:
        x_sub, y_sub = get_subset(ind, x_train, y_train)
        estimator_clone = clone(estimator)
        estimator_clone.fit(x_sub, y_sub)
        ind.estimator = estimator_clone
        ensembles.append(ind.estimator)
    return ensembles


def vote_result_ensembles(ensembles, x_test, result='soft'):
    y_pred_labels_ensembles = []
    y_pred_prob_labels_ensembles = []
    #  Traverse each individual to obtain the prediction result
    for estimator in ensembles:
        ind_pred = estimator.predict(x_test)
        ind_proba = estimator.predict_proba(x_test)
        y_pred_labels_ensembles.append(ind_pred)
        y_pred_prob_labels_ensembles.append(ind_proba)
    vote_pred = stats.mode(y_pred_labels_ensembles, axis=0, keepdims=False).mode.flatten()  # Vote by column, take the mode in each column as the final classification result
    vote_pred_prob = np.stack(y_pred_prob_labels_ensembles, axis=0)
    vote_pred_prob = np.mean(vote_pred_prob, axis=0)
    if result == 'soft':
        return vote_pred_prob
    elif result == 'hard':
        return vote_pred
    else:
        raise ValueError('result must be "soft" or "hard"')