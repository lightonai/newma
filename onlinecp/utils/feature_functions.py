# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 17:30:04 2018

@author: nkeriven
"""
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist


def generate_frequencies(m, d, kernel='Gaussian', choice_sigma='median', data=None, sigmasq=1):
    """
    Generate random frequencies.

    Parameters
    ----------
    m: int,
        number of frequencies.
    d: int,
        dimension of frequencies.
    kernel: str, default 'Gaussian',
        type of kernel.
    choice_sigma : string,
        method to choose sigma
        'median' : median trick on some data.
    data: np.ndarray,
        some data to choose sigma.
    sigmasq: float,
        fixed variance of kernel if no method to learn it is provided.

    Returns
    -------
    W : np.ndarray (m,d),
        random frequencies.

    """
    # Median trick
    if choice_sigma == 'median':
        if data is None:
            raise Exception("You chose median for sigma in generate_frequencies, but you didn't pass any data")
        distances = euclidean_distances(data, data)
        squared_distances = distances.flatten() ** 2
        sigmasq = np.median(squared_distances)

    # generate frequencies
    if kernel == 'Gaussian':
        W = np.random.randn(m, d)/np.sqrt(sigmasq)
    return W, sigmasq


def fourier_feat(x, W):
    """Fourier Features.

    Parameters
    ----------
    x: np.ndarray (d, ),
        sample.
    W: np.ndarray (m, d),
        frequencies.

    Returns
    -------
    y: np.ndarray (2*m,),
        Fourier features at each frequency, cos and sin.
    """
    temp = x.flatten().dot(W.T).T
    return np.concatenate((np.cos(temp), np.sin(temp)))/np.sqrt(W.shape[0])


def gauss_kernel(X, y, sigma):
    """
    Gaussian kernel.

    Parameters
    ----------
    X: np.ndarray  (n,d),
    y: np.ndarray  (d,),
    sigma: float,
        variance of the kernel.

    Returns
    -------
        k (n,): kernel between each row of X and y
    """
    return np.squeeze(np.exp(-cdist(X, y[np.newaxis, :], metric='sqeuclidean') / (2 * sigma ** 2)))
