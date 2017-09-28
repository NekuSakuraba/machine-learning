import numpy as np
from numpy import log as ln

from sklearn.cluster import KMeans

from scipy.linalg import cholesky, solve_triangular, LinAlgError
from scipy.special import gammaln, digamma, logsumexp

def _compute_precision_cholesky(covariances):
    n_components, n_features, _ = covariances.shape

    precisions_chol = np.empty((n_components, n_features, n_features))
    for k, covariance in enumerate(covariances):
        try:
            cov_chol = cholesky(covariance, lower=True)
        except LinAlgError:
            raise ValueError('covariance: {0}'.format(covariance))
        precisions_chol[k] = solve_triangular(cov_chol, np.eye(n_features), lower=True).T
    return precisions_chol

class MultivariateTMixture:
    def __init__(self, n_components=1, max_iter=100, random_state=None, reg_covar=1e-6, tol=1e-3):
        self.n_components = n_components
        self.random_state = random_state

        self.max_iter  = max_iter
        self.reg_covar = reg_covar

        self.tol = tol

    def _initialize_parameters(self, X):
        n_components = self.n_components

        # initializng u
        u = np.ones((self.n_samples, self.n_components))

        # initializing tau
        tau = np.zeros((self.n_samples, n_components))
        label = KMeans(n_clusters=n_components, random_state=self.random_state).fit(X).labels_
        tau[np.arange(self.n_samples), label] = 1

        return tau, u

    def _estimate_covariances(self, X, means, n, tau, u):
        n_components, n_features = means.shape
        covariances = np.empty((n_components, n_features, n_features))
        for k in range(n_components):
            diff = X - means[k]
            covariances[k] = np.dot(tau[:, k] * u[:, k] * diff.T, diff) / n[k]
            covariances[:: n_features + 1] += self.reg_covar

        return covariances

    def _estimate_parameters(self, X, tau, u):
        """

        Returns
        -------
        pi :

        means :

        covariances :
        """
        n = tau.sum(0)  # pi without normalizing
        mu = np.dot(tau.T * u.T, X) / np.sum(tau * u, 0)[:, np.newaxis]
        covariances = self._estimate_covariances(X, mu, n, tau, u)

        return n, mu, covariances

    def _compute_q1(self, tau):
        return tau * ln(self.pi/self.n_samples) # normalized pi

    def _compute_q2(self, tau, u):
        q2 = np.empty(tau.shape)
        for k in range(self.n_components):
            q2[:, k] = tau[:, k] * self._compute_q2_helper(u[:, k], self.df[k])
        return q2

    def _compute_q2_helper(self, u, df):
        half_df   = df * .5
        half_feat = self.n_features * .5

        return -gammaln(half_df) + half_df*ln(half_df) \
               +half_df*(digamma(half_df+half_feat) - ln(half_df+half_feat) \
                            + np.sum(ln(u) - u))

    def _compute_q3(self, tau, u, mahalanobis):
        # log_det_chol = np.sum(ln(self.precisions_chol.reshape(self.n_components, -1)[:, ::self.n_features + 1]), axis=1)
        log_det = self._compute_log_det()

        log_prob = mahalanobis * u
        q3 = .5 * (self.n_features*(ln(u) - ln(2 * np.pi)) - log_prob) + log_det
        return q3 * tau

    def _compute_log_likelihood(self, X, tau, u):
        q1 = self._compute_q1(tau)
        q2 = self._compute_q2(tau, u)

        self.precisions_chol = _compute_precision_cholesky(self.covariances)

        mahalanobis = self._compute_mahalanobis(X)
        q3 = self._compute_q3(tau, u, mahalanobis)

        weighted_log_prob = q1 + q2 + q3
        log_norm_prob = logsumexp(weighted_log_prob, axis=1)
        return np.mean(log_norm_prob)

    def _compute_mahalanobis(self, X):
        n_samples, _ = X.shape

        mahalanobis = np.empty((n_samples, self.n_components))
        for k, (mu, prec_chol) in enumerate(zip(self.means, self.precisions_chol)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            mahalanobis[:, k] = np.sum(np.square(y), axis=1)
        return mahalanobis

    def _compute_u(self, mahalanobis):
        return (self.df + self.n_features) / (mahalanobis + self.df)

    def fit(self, X):
        self.n_samples, self.n_features = X.shape

        tau, u = self._initialize_parameters(X)

        self.pi, means, covariances = self._estimate_parameters(X, tau, u)

        # precision cholesky
        self.precisions_chol = _compute_precision_cholesky(covariances)

        df = np.ones(self.n_components) * 4

        self.means = means
        self.covariances = covariances
        self.df = df

        self.lower_bound = -np.inf

        for _ in range(self.max_iter):
            prev_lower_bound = self.lower_bound

            log_resp, u = self._e_step_1(X)
            tau = np.exp(log_resp)

            self._cm_step_1(X, tau, u)

            u = self._e_step_2(X)
            self._cm_step_2(tau, u)

            self.lower_bound = self._compute_log_likelihood(X, tau, u)

            change = self.lower_bound - prev_lower_bound

            if abs(change) < self.tol:
                print 'done with #{} iterations'.format(_)
                break

    def _e_step_1(self, X):
        """ E-step 01 - It calculates the loglikelihood under the current parameters
            and estimates the parameters tau and u.

        Parameters
        ----------
        X : array-like, shape(n_components, n_features)

        Returns
        -------
        tau : array-like, shape(n_samples, n_components)

        u   : array-like, shape(n_samples, n_components)
        """
        # E-step 01
        n_components, n_features = self.n_components, self.n_features

        log_det = self._compute_log_det()

        log_prob = self._compute_mahalanobis(X)

        half_df = self.df * .5
        half_p  = n_features * .5
        weighted_log = gammaln(half_df+half_p) - gammaln(half_df) \
                       -(half_df + half_p)*(ln(self.df + log_prob) - ln(self.df)) \
                       -half_p*ln(np.pi * self.df) + log_det + ln(self.pi)
        # TODO Verificar se o tau precisa ser normalizado

        log_prob_norm = logsumexp(weighted_log, axis=1)[:, np.newaxis]

        log_resp = weighted_log - log_prob_norm

        u = (self.df + n_features) / (self.df + log_prob)

        return log_resp, u

    def _cm_step_1(self, X, tau, u):
        n_samples, _ = X.shape

        self.pi, self.means, self.covariances = self._estimate_parameters(X, tau, u)

        self.precisions_chol = _compute_precision_cholesky(self.covariances)

    def _e_step_2(self, X):
        # computing the mahalanobis distance
        # with the parameters from cm_step_1
        mahalanobis = self._compute_mahalanobis(X)
        return self._compute_u(mahalanobis)

    def _cm_step_2(self, tau, u):
        n = tau.sum(axis=0)
        tau_u = np.sum(tau * (np.log(u) - u), axis=0)

        df = np.empty(self.df.shape)
        for k in range(self.n_components):
            best = self.df[k]
            step = 10.
            for __ in range(5):
                for v in np.arange(best, best + step, step / 10.):
                    half_v = v * .5
                    half_p = self.n_features * .5

                    result = -digamma(half_v) + np.log(half_v) + 1 \
                             +digamma(half_v+half_p) - np.log(half_v+half_p) \
                             +tau_u[k]/n[k]

                    if result > 0:
                        best = v
                    else:
                        break
                step /= 10.
            df[k] = best
        self.df = df

    def _compute_log_det(self):
        return np.sum(ln(self.precisions_chol.reshape(self.n_components, -1)[:, ::self.n_features + 1]), axis=1)

    def score_samples(self, X):
        n_components, n_features = self.n_components, self.n_features

        log_det  = self._compute_log_det()
        log_prob = self._compute_mahalanobis(X)

        half_df = self.df * .5
        half_p  = n_features * .5
        weighted_log = gammaln(half_df+half_p) - gammaln(half_df) \
                       -(half_df + half_p)*(ln(self.df + log_prob) - ln(self.df)) \
                       -half_p*ln(np.pi * self.df) + log_det + ln(self.pi)
        return logsumexp(weighted_log, axis=1)
