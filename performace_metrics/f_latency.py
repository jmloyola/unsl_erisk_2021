import numpy as np
import sklearn.metrics as metrics


def value_p(k):
    return -(np.log(1/3) / (k - 1))


def f_penalty(k, _p):
    return -1 + (2 / (1 + np.exp((-_p)*(k-1))))


def speed(y_pred, y_true, d, p):
    penalty_list = [f_penalty(k=d[i], _p=p) for i in range(len(y_pred)) if y_pred[i] == 1 and y_true[i] == 1]

    if len(penalty_list) != 0:
        return 1 - np.median(penalty_list)
    else:
        return 0.


def f_latency(labels, true_labels, delays, penalty):
    f1_score = metrics.f1_score(y_pred=labels, y_true=true_labels, average='binary')
    speed_value = speed(y_pred=labels, y_true=true_labels, d=delays, p=penalty)

    return f1_score * speed_value
