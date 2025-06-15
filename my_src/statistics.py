import numpy as np
from scipy.stats import ttest_ind
from itertools import combinations
from scipy.special import comb
from iteration_utilities import  random_combination


def compute_max_t_stat(a, b):
    if len(a.shape) == 2:
        t_stats, p_vals = ttest_ind(a, b, axis=0) 
        t_stats = max(np.abs(t_stats))
    else:
        t_stats, p_vals = ttest_ind(a, b) 
    return t_stats

def permutation_testing(a, b, n_perm=10000):
    n_samples_a = a.shape[0]
    n_samples_b = b.shape[0]
    n_samples = n_samples_a + n_samples_b
    
    data = np.concatenate([a, b])
    indices = range(n_samples)

    t_statistics = []
    t_statistics.append(compute_max_t_stat(a,b))
    for i in range(n_perm - 1):
        a_indices = random_combination(indices, n_samples_a)
        a_indices = list(a_indices)
        b_indices = list(set(indices) - set(a_indices))
        a_new = data[a_indices]
        b_new = data[b_indices]

        t_statistic = compute_max_t_stat(a_new, b_new)
        t_statistics.append(t_statistic)

    t_statistic_actual = t_statistics[0]
    p_value = (np.abs(t_statistics) >= np.abs(t_statistic_actual)).mean()
    return p_value, t_statistics