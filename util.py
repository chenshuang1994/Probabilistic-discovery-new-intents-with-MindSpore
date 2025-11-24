import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score
)


##############################################
# Hungarian alignment
##############################################
def hungray_aligment(y_true, y_pred):
    """
    y_true: numpy array
    y_pred: numpy array
    """
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    # Hungarian matching
    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w


##############################################
# Clustering Accuracy (ACC)
##############################################
def clustering_accuracy_score(y_true, y_pred):
    ind, w = hungray_aligment(y_true, y_pred)
    acc = sum([w[i, j] for (i, j) in ind]) / y_pred.size
    return acc


##############################################
# Total clustering evaluation results
##############################################
def clustering_score(y_true, y_pred):
    """
    Return ACC / ARI / NMI (percentage)
    """
    return {
        "ACC": round(clustering_accuracy_score(y_true, y_pred) * 100, 2),
        "ARI": round(adjusted_rand_score(y_true, y_pred) * 100, 2),
        "NMI": round(normalized_mutual_info_score(y_true, y_pred) * 100, 2),
    }
