import numpy as np


def delay_cost(k, break_point):
    return 1 - (1 / (1 + np.exp(k-break_point)))


def erde_user(label, true_label, delay, _c_tp, _c_fn, _c_fp, _o):
    if label == 1 and true_label == 1:
        return delay_cost(k=delay, break_point=_o) * _c_tp
    elif label == 1 and true_label == 0:
        return _c_fp
    elif label == 0 and true_label == 1:
        return _c_fn
    elif label == 0 and true_label == 0:
        return 0


def erde(labels_list, true_labels_list, delay_list, c_fp, c_tp=1, c_fn=1, o=50):
    """Early Risk Detection Error performance measure.

    Parameters
    ----------
    labels_list : numpy.ndarray
        The numpy array of the predicted labels.
    true_labels_list : list of int
        The list of the true labels.
    delay_list : numpy.ndarray
        The delays to give a response for every user.
    c_fp : float
        The cost of the false positives.
    c_tp : float, default=1
        The cost of the true positives.
    c_fn : float, default=1
        The cost of the false negatives.
    o : int
        The threshold where the delay cost is half.

    Returns
    -------
    float
        The Early Risk Detection Error.
    """
    erde_list = [erde_user(label=l, true_label=true_labels_list[i], delay=delay_list[i], _c_tp=c_tp,
                           _c_fn=c_fn, _c_fp=c_fp, _o=o) for i, l in enumerate(labels_list)]
    return np.mean(erde_list)
