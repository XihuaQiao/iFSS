import numpy as np

def try_index(scalar_or_list, i):
    try:
        return scalar_or_list[i]
    except TypeError:
        return scalar_or_list
    

def get_lambda(alpha=1.0):
    '''Return lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam


def mixup_data(x, y, alpha=1.0, lam=None):
    if not lam:
        lam = get_lambda(alpha)
    indices = np.random.permutation(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[indices, :]

    y_a, y_b = y, y[indices]

    return mixed_x, y_a, y_b, lam
