import numpy as np

def update_stats(s, mu, covmat_first, count):
    mu = (mu * count + s) / (count + 1.)
    covmat_first = (count * covmat_first + np.outer(s, s)) / (count + 1.)
    covmat = covmat_first - np.outer(mu, mu)
    return mu, covmat_first, covmat