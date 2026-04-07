import numpy as np
from sklearn.model_selection import cross_val_predict

from metrics import calculate_expert_accuracy
from utils import get_subset


def objective_function(individual, weights_train):
    y_sub, pred_proba = individual.y_sub_and_pred_proba  # Get the instance selection label and prediction probability of the individual
    pred = np.argmax(pred_proba, axis=1)  # Get the predicted label of an individual
    Acc1, Acc2, Acc3 = calculate_expert_accuracy(pred, y_sub, weights_train)  # Acc1、Acc2、Acc3
    return Acc1, Acc2, Acc3


def evaluate_individual(individual, estimator, x_train, y_train, n_splits, random_state, weights_train):
    if not individual.fitness.valid:
        x_sub, y_sub = get_subset(individual, x_train, y_train)
        y_pred_proba = cross_val_predict(estimator, x_sub, y_sub, cv=n_splits, method='predict_proba')
        individual.y_sub_and_pred_proba = (y_sub, y_pred_proba)
        individual.fitness.values = objective_function(individual, weights_train)
        return {"valid": True, "fitness": individual.fitness.values,
                "y_sub_and_pred_proba": individual.y_sub_and_pred_proba}
    return {"valid": False}
