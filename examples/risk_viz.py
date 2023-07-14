import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import binom

n = 100
risks = np.arange(0, 1, 0.01)
eps = 0.3
delta = 0.05


def bonferroni_correction(p_value, n_tests):
    return min(1, n_tests * p_value)

p_vals = [binom.cdf(n * r_i, n, eps) for r_i in risks]


plt.plot(risks, p_vals)

plt.axhline(delta, color="red", linestyle="--")

plt.savefig("pvals.png")
