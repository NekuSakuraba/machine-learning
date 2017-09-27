import numpy as np
from multivariate_util import multivariate_t_rvs
from multivariate_t_mixture import MultivariateTMixture

def two_components():
    np.random.seed(0)
    X1 = np.random.randn(350, 2) * .5 + 7
    X2 = np.random.randn(350, 2) * 1.1

    X = np.r_[X1, X2]

    t = MultivariateTMixture(n_components=2, random_state=0)
    t.fit(X)

    print 'means'
    print t.means

    print 'covariances'
    print t.covariances

    print 'df'
    print t.df

def two_components_2():
    X1 = multivariate_t_rvs([7, 7], [[.5,0], [0,.5]], 7.5, 350)
    X2 = multivariate_t_rvs([0, 0], [[1.1,0], [0,1.1]], 18.33, 350)
    X = np.r_[X1, X2]

    t = MultivariateTMixture(n_components=2, random_state=0)
    t.fit(X)

    print 'means'
    print t.means

    print 'covariances'
    print t.covariances

    print 'df'
    print t.df

def three_components():
    np.random.seed(0)
    X1 = np.random.randn(350, 2) * .5 + 7
    X2 = np.random.randn(350, 2) * 1.1
    X3 = np.random.randn(350, 2) * .333 -2

    X = np.r_[X1, X2, X3]

    t = MultivariateTMixture(n_components=3, random_state=0)
    t.fit(X)

    # import matplotlib.pyplot as plt
    # plt.scatter(X1[:,0], X1[:,1])
    # plt.scatter(X2[:,0], X2[:,1])
    # plt.scatter(X3[:,0], X3[:,1])
    # plt.scatter(t.means[:, 0], t.means[:, 1], c='r', marker='x')
    #
    # plt.show()

    print 'means'
    print t.means

    print 'covariances'
    print t.covariances

    print 'df'
    print t.df

if __name__ == '__main__':
    two_components_2()