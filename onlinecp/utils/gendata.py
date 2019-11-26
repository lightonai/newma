# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 15:09:35 2018

@author: nkeriven
"""
import glob

import numpy as np
from scipy.io import wavfile, loadmat
import scipy.signal as sig
from sklearn import mixture
import sphfile


def gmdraw(weights, mu, Sigma, n):
    """Draw samples from GMM.

    Parameters
    ----------
    weights: np.ndarray (k, ),
        weights of the GMM.
    mu: np.ndarray (k, d),
        means of the GMM.
    Sigma: np.ndarray (k, d, d),
        covariance of the GMM.
    n: int,
        number of samples.

    Returns
    -------
    X: np.ndarray (n, d),
        samples.
    label: np.ndarray of int (n, ),
        labels of samples.
    """

    k = weights.shape[0]
    d = mu.shape[1]
    p = np.cumsum(weights)
    # label
    label = np.random.rand(n)
    for i in range(n):
        label[i] = np.sum(label[i] > p)
    # cholesky
    cSigma = np.zeros((k, d, d))
    for l in range(k):
        cSigma[l, :, :] = np.linalg.cholesky(Sigma[l, :, :])
    # data
    X = np.zeros((n, d))
    for i in range(n):
        j = int(label[i])
        X[i, :] = mu[j, :] + np.dot(np.random.randn(1, d), cSigma[j, :, :])
    return X, label


def generateGMM(d=10, k=10, n=1000, std_mean=1, concentration_wishart=30, concentration_dirichlet=5):
    """Generate random parameters of GMM with diag covariance, draw samples.

    Parameters
    ----------
    d: int,
        dimension.
    k: int,
        number of components.
    n: int,
        number of samples.
    std_mean: float,
        the means will be drawn from a centered Gaussian with covariance (std_mean**2)*Id.
    concentration_wishart: float,
        the bigger, the more concentrated the diagonal covariances are around Id.
    concentration_dirichlet: float,
        the bigger, the more concentrated the weights are around uniform 1/k.

    Returns
    -------
    generated_data: dictionary with fields
        'data' (n,d): samples
        'weights' (k,): weights
        'means' (k,d): means
        'cov' (k,d): diagonal of covariances
        'label' (n,): labels of samples
        'gmm': scikit_learn mixture object
    """

    concentration_wishart = np.max((concentration_wishart, 3))

    # weights
    weights = np.random.dirichlet(concentration_dirichlet*np.ones(k))

    # means
    mu = std_mean*k**(1/d)*np.random.randn(k, d)

    # sigma
    Sigma = np.zeros((k, d))
    for l in range(k):
        Sigma[l, :] = (concentration_wishart - 2)/np.sum(np.random.randn(int(concentration_wishart), d)**2, axis=0)

    # sklearn object
    # , weights_init = GM['weights'], means_init = GM['means'], precisions_init = GM['cov'], max_iter = 1)
    clf = mixture.GaussianMixture(n_components=k, covariance_type='diag')
    clf.means_ = mu
    clf. covariances_ = Sigma
    clf.precisions_cholesky_ = mixture.gaussian_mixture._compute_precision_cholesky(
        Sigma, clf.covariance_type)
    clf.weights_ = weights
    X, label = clf.sample(n_samples=n)
    p = np.random.permutation(n)
    X = X[p, :]
    label = label[p]
    generated_data = {'data': X,
                      'weights': weights,
                      'means': mu,
                      'cov': Sigma,
                      'label': label,
                      'gmm': clf}
    return generated_data


def stream_GMM(d=10, k=10, n=1000, nb_change=50, std_mean=0.2, concentration_wishart=30, concentration_dirichlet=5):
    X = np.zeros((n*(nb_change), d))
    ground_truth = np.zeros(n*(nb_change))
    for i in range(nb_change):
        GM = generateGMM(d=d, k=k, n=n, std_mean=std_mean, concentration_wishart=concentration_wishart,
                         concentration_dirichlet=concentration_dirichlet)
        X[i*n:(i+1)*n, :] = GM['data']
        if i != 0:
            ground_truth[i*n] = 1
    return X, ground_truth


def import_vad_data(root_path='/', nb_change=300, interval_speech=10, fs=16000, SNR_convex_coeff=1):
    noise_paths = glob.glob(root_path + 'QUT-NOISE/*_convert.wav', recursive=True)#[:2]
    speech_paths = glob.glob(root_path + 'TIMIT/**/*.WAV', recursive=True)
    perm_speech = np.random.permutation(len(speech_paths))

    nb_change_per_noise_file = int(nb_change/len(noise_paths))
    length_noise = fs*nb_change_per_noise_file*interval_speech

    X_tot = np.empty(0)
    gt_tot = np.empty(0)
    for noise_file in noise_paths:

        print(noise_file)
        # noise
        data = wavfile.read(noise_file)[1]
        rand_start = int(np.random.rand(1)*(len(data)-length_noise))
        data = data[rand_start:rand_start+length_noise].astype('float64')
        data *= (1-SNR_convex_coeff)/np.max(np.abs(data))

        # fill speech
        start_ind = np.zeros(nb_change_per_noise_file, dtype=np.int)
        ind = 0
        for speech_ind in range(nb_change_per_noise_file):
            start_ind[ind] = fs*interval_speech*speech_ind
            sph = sphfile.SPHFile(speech_paths[perm_speech[speech_ind]])
            sph.open()
            speech_data = sph.content.astype('float64')
            speech_data *= SNR_convex_coeff/np.max(np.abs(speech_data))
            data[start_ind[ind]:start_ind[ind]+len(speech_data)] += speech_data
            ind += 1
        data /= np.max(np.abs(data))

        # stft
        X = np.abs(sig.stft(data)[2].T)
        print(X.shape)
        X_tot = np.vstack((X_tot, X)) if X_tot.size else X

        # ground_truth
        n = X.shape[0]
        gt = np.zeros(n, dtype=np.bool)
        gt[0] = True
        gt[np.ceil(start_ind.astype(np.float64)*n/length_noise).astype(np.int)] = True
        gt_tot = np.concatenate((gt_tot, gt))

    return X_tot, gt_tot
