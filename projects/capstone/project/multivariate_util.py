# Source
## https://github.com/statsmodels/statsmodels/blob/master/statsmodels/sandbox/distributions/multivariate.py#L90
import numpy as np
from numpy import log

from scipy.special import gamma, digamma
from scipy.linalg import inv, det

from sklearn.cluster import KMeans
import scipy.linalg.blas as FB

def get_random(X):
    """Get a random sample from X.
    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
    
    Returns
    -------
    array-like, shape (1, n_features)
    """
    size = len(X)
    idx = np.random.choice(range(size))
    return X[idx]

#written by Enzo Michelangeli, style changes by josef-pktd
# Student's T random variable
def multivariate_t_rvs(m, S, df=np.inf, n=1):
    '''generate random variables of multivariate t distribution
    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))
    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable
    '''
    m = np.asarray(m)
    d = len(m)
    if df == np.inf:
        x = 1.
    else:
        x = np.random.chisquare(df, n)/df
    z = np.random.multivariate_normal(np.zeros(d),S,(n,))
    return m + z/np.sqrt(x)[:,None]   # same output format as random.multivariate_normal