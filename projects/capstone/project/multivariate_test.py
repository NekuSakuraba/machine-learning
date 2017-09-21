import numpy as np
from multivariate_util import multivariate_t_rvs
from multivariate_t_mixture import MultivariateTMixture

mu  = [0, 0]
cov = np.eye(2)
X1 = multivariate_t_rvs(mu, cov, 4, 350)

mu  = [5, 5]
cov = np.eye(2) * .5
X2 = multivariate_t_rvs(mu, cov, 19, 350)

mu  = [-5, -5]
cov = np.eye(2) * 2
X3 = multivariate_t_rvs(mu, cov, 7.5, 350)

# X = np.concatenate([X1, X2, X3])
# t = MultivariateTMixture(3, max_iter=100)

X = np.concatenate([X1, X2])
t = MultivariateTMixture(2, max_iter=100)

t.fit(X)

print t.means
print t.sigmas
print t.df

import matplotlib.pyplot as plt
arr = range(len(t.likelihood))
plt.scatter(arr, t.likelihood)
plt.show()
