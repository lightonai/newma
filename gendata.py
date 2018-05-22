# -*- coding: utf-8 -*-
"""
Functions to generate streams of data with gaussian distribution changing mean and variance. Means drawn according to
N(0, std_mean^2), variances drawn according to an inverse Wishart distribution (high concentration_wishart means strong
concentration around 1).
"""
import numpy as np
from sklearn import mixture


def gm_draw(weights, mu, Sigma, n):
    k = weights.shape[0]
    d = mu.shape[1]
    p = np.cumsum(weights)
    # label
    label = np.random.rand(n)
    for i in range(n):
        label[i] = np.sum(label[i]>p)
    # cholesky decomposition
    cSigma = np.zeros((k, d, d))
    for l in range(k):
        cSigma[l, :, :] = np.linalg.cholesky(Sigma[l, :, :])
    # data
    X = np.zeros((n, d))
    for i in range(n):
        j = int(label[i])
        X[i, :] = mu[j, :] + np.dot(np.random.randn(1, d), cSigma[j, :, :])
    return X, label


def generate_GMM(d=10, k=10, n=1000, std_mean=1., concentration_wishart=30, concentration_dirichlet=5):
    concentration_wishart = np.max((concentration_wishart, 3))
    
    # weights
    weights = np.random.dirichlet(concentration_dirichlet * np.ones(k))
    
    # means
    mu = std_mean*k**(1/d)*np.random.randn(k, d)
    
    # sigma
    Sigma = np.zeros((k, d))
    for l in range(k):
        Sigma[l, :] = (concentration_wishart - 2)/np.sum(np.random.randn(int(concentration_wishart), d)**2, axis=0)
        
    # sklearn object
    clf = mixture.GaussianMixture(n_components=k, covariance_type='diag')
    clf.means_ = mu
    clf. covariances_ = Sigma
    clf.precisions_cholesky_ = mixture.gaussian_mixture._compute_precision_cholesky(Sigma, clf.covariance_type)
    clf.weights_ = weights
    X, label = clf.sample(n_samples=n)
    p = np.random.permutation(n)
    X = X[p, :]
    label = label[p]
    return {'data': X,
            'weights': weights,
            'means': mu,
            'cov': Sigma,
            'label': label,
            'gmm': clf}
    

# generate streams: (n * nb_change, d) array of samples, with binary ground truth. A change every n samples.
def stream_gaussian(d=10, n=1000, nb_change=50, std_mean=0.5, concentration_wishart=30):
    concentration_wishart = np.max((concentration_wishart, 3))
    X = np.zeros((n * nb_change, d))
    ground_truth = np.zeros(n * nb_change)
    for i in range(nb_change):
        scaling = np.sqrt((concentration_wishart - 2) / np.sum(np.random.randn(int(concentration_wishart)) ** 2))
        wishart_contrib = scaling * np.random.randn(n, d)
        X[i*n:(i+1)*n, :] = wishart_contrib + std_mean*np.broadcast_to(np.random.randn(d), [n, d])
        if i != 0:
            ground_truth[i*n] = 1
    return X, ground_truth


def stream_GMM(d=10, k=10, n=1000, nb_change=50, std_mean=0.2, concentration_wishart=30, concentration_dirichlet=5):
    X = np.zeros((n*nb_change, d))
    ground_truth = np.zeros(n*nb_change)
    for i in range(nb_change):
        GM = generate_GMM(d=d, k=k, n=n, std_mean=std_mean, concentration_wishart=concentration_wishart,
                          concentration_dirichlet=concentration_dirichlet)
        X[i*n:(i+1)*n, :] = GM['data']
        if i != 0:
            ground_truth[i*n] = 1
    return X, ground_truth
