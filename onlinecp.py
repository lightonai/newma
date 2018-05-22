# -*- coding: utf-8 -*-
"""
Online Change Point
Code for the paper "NEWMA: a new method for scalable model-free online change-point detection"

Implements NEWMA, Scan-B, and some utils functions.

"""

import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist


class Newma:
    """ Implementation of Newma algorithm.

    Parameters
    ----------
    init_vec: float,
        initial value for the sketches.
    updt_coeff, updt_coeff2: floats,
        update coefficients for EWMAs (small and big lambda)
    updt_func: function,
        function to compute the mapping Psi (default is for detection of changes in the mean).
    dist_func: function.
        function to compute the distance.
    updt_coeff_thres: float,
        eta.
    thres_mult, thres_offset: floats,
        multiplicative constant and offset for adaptive threshold.
    store_result: boolean,
        if True, store intermediary dist and threshold for later display.


    Attributes
    ----------
    init_vec: float,
        initial value for the sketches.
    updt_coeff, updt_coeff2: floats,
        update coefficients for EWMAs (small and big lambda)
    updt_func: function,
        function to compute the mapping Psi (default is for detection of changes in the mean).
    dist_func: function.
        function to compute the distance.
    updt_coeff_thres: float,
        eta.
    thres_mult, thres_offset: floats,
        multiplicative constant and offset for adaptive threshold.
    store_result: boolean,
        if True, store intermediary dist and threshold for later display.
    thres: float,
        current value of the adaptive threshold.
    dist: list,
        history of NEWMAs.
    result: list,
        history of flagged points.
    thres_debug: list,
        history of adaptive threshold values.
    """
    def __init__(self, init_vec=0., updt_coeff=0.05, updt_coeff2=0.1, updt_func=lambda x: x,
                 dist_func=lambda x, y: linalg.norm(x-y), updt_coeff_thres=0.01, thres_mult=1.5, thres_offset=0.1,
                 store_result=False):
        self.sketch = init_vec
        self.sketch2 = init_vec
        self.thres = 0.
        self.updt_coeff = updt_coeff
        self.updt_coeff2 = updt_coeff2
        self.updt_coeff_thres = updt_coeff_thres
        self.updt_func = updt_func
        self.dist_func = dist_func
        self.thres_mult = thres_mult
        self.thres_offset = thres_offset
        self.store_result = store_result
        self.dist = []
        self.result = []
        self.thres_debug = []

    def update(self, sample):
        """Apply update online.

        Parameters
        ----------
        sample: 1D np.array,
            data point to use for the update.

        Return
        ------
        online_results: dict,
            flagged point or not, value of distance and value of threshold.
        """
        temp = self.updt_func(sample)
        # sketches
        self.sketch = (1-self.updt_coeff) * self.sketch + self.updt_coeff * temp
        self.sketch2 = (1-self.updt_coeff2) * self.sketch2 + self.updt_coeff2 * temp
        # distance
        d = self.dist_func(self.sketch, self.sketch2)
        # adaptive threshold
        self.thres = (1-self.updt_coeff_thres) * self.thres + self.updt_coeff_thres * d
        res = d > self.thres_mult * self.thres + self.thres_offset
        # record result
        if self.store_result:
            self.dist.append(d)
            self.result.append(res)
            self.thres_debug.append(self.thres)
        online_results = {'result': res, 'dist': d, 'thres': self.thres}
        return online_results

    def apply_to_data(self, data):
        """Apply "offline" the update function to a stream of data, for experiments.

        Parameters
        ----------
        data: 2D np.array,
            stream of data to run NEWMA on.

        Return
        ------
        self: Newma.
        """
        n = data.shape[0]
        for i in range(n):
            self.update(data[i, :])
        return self


class ScanB:
    """ Implementation of Scan-B kernel algorithm.
    Slight modification: we use the BIASED estimator of the MMD (i.e. with diagonal terms), because the unbiased
    estimator may be negative.
    """
    def __init__(self, d, kernel_func=lambda X, Y: np.dot(Y, X.T), B=100, N=5, updt_coeff_thres=0.01, thres_mult=1.5,
                 thres_offset=0.1, store_result=False):
        self.kernel_func = kernel_func
        self.B = B
        self.N = N
        # what do we store
        self.kernel_sum_XX = np.zeros(N)
        self.kernel_sum_XY = np.zeros(N)
        self.kernel_sum_YY = 0
        self.X = np.zeros((B, d, N))
        self.Y = np.zeros((B, d))
        
        # adaptive threshold
        self.thres = 0
        self.updt_coeff_thres = updt_coeff_thres
        self.thres_mult = thres_mult
        self.thres_offset = thres_offset
        # do we store results for debug ?
        self.store_result = store_result
        self.dist = []
        self.result = []
        self.thres_debug = []
        
    def update_kernel_sum(self, val, datax, datay, newx, newy):
        n = datax.shape[0]
        val = val + (1 / (n ** 2)) * (self.kernel_func(datax[:-1, :], newy).sum() +
                                      self.kernel_func(datay[:-1, :], newx).sum() -
                                      self.kernel_func(datax[:-1, :], datay[-1, :]).sum() -
                                      self.kernel_func(datay[:-1, :], datax[-1, :]).sum())
        return val
    
    def update(self, sample):
        # add sample to Y
        self.kernel_sum_YY = self.update_kernel_sum(self.kernel_sum_YY, self.Y, self.Y, sample, sample)
        for i in range(self.N-1):
            self.kernel_sum_XX[i] = self.update_kernel_sum(self.kernel_sum_XX[i],
                                                           self.X[:, :, i], self.X[:, :, i],
                                                           self.X[-1, :, i+1], self.X[-1, :, i+1])
            self.kernel_sum_XY[i] = self.update_kernel_sum(self.kernel_sum_XY[i],
                                                           self.X[:, :, i], self.Y,
                                                           self.X[-1, :, i+1], sample)
        self.kernel_sum_XX[-1] = self.update_kernel_sum(self.kernel_sum_XX[-1],
                                                        self.X[:, :, -1], self.X[:, :, -1],
                                                        self.Y[-1, :], self.Y[-1, :])
        self.kernel_sum_XY[-1] = self.update_kernel_sum(self.kernel_sum_XY[-1],
                                                        self.X[:, :, -1], self.Y,
                                                        self.Y[-1, :], sample)
        
        # roll out old data
        for i in range(self.N-1):
            self.X[:, :, i] = np.roll(self.X[:, :, i], 1, axis=0)
            self.X[0, :, i] = self.X[-1, :, i + 1]
        self.X[:, :, -1] = np.roll(self.X[:, :, -1], 1, axis=0)
        self.X[0, :, -1] = self.Y[-1, :]
        self.Y = np.roll(self.Y, 1, axis=0)
        self.Y[0, :] = sample
        
        # compute d
        d = self.kernel_sum_XX.sum() + self.N * self.kernel_sum_YY - 2 * self.kernel_sum_XY.sum()
        self.thres = (1 - self.updt_coeff_thres) * self.thres + self.updt_coeff_thres * d
        res = d > self.thres_mult * self.thres + self.thres_offset
        if self.store_result:
            self.dist.append(d)
            self.result.append(res)
            self.thres_debug.append(self.thres)
        return {'result': res, 'dist': d, 'thres': self.thres}
    
    def apply_to_data(self, data):
        n = data.shape[0]
        if n < (self.N + 1) * self.B:
            raise IndexError('Not enough data to apply Scan-B')
        # init with first data
        self.Y = data[self.N * self.B:(self.N + 1) * self.B, :]
        self.kernel_sum_YY = kernel_sum_u(self.Y, self.Y, self.kernel_func)
        for i in range(self.N):
            self.X[:, :, i] = data[i * self.B:(i + 1) * self.B, :]
            self.kernel_sum_XX[i] = kernel_sum_u(self.X[:, :, i], self.X[:, :, i], self.kernel_func)
            self.kernel_sum_XY[i] = kernel_sum_u(self.X[:, :, i], self.Y, self.kernel_func)
        self.dist = np.zeros((self.N + 1) * self.B).tolist()  # first points not taken into account
        self.result = np.zeros((self.N+1) * self.B).tolist()
        self.thres_debug = np.zeros((self.N + 1) * self.B).tolist()
        for i in np.arange((self.N + 1) * self.B, n):
            self.update(data[i, :])


# utilities functions
def fourier_feature(x, W):
    temp = x.flatten().dot(W.T).T
    return np.concatenate((np.cos(temp), np.sin(temp)))


def MMDu(datax, datay, kernel_func):
    n = datax.shape[0]
    val = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                x_contrib = kernel_func(datax[i, :], datax[j, :])
                y_contrib = kernel_func(datay[i, :], datay[j, :])
                cross_contrib = kernel_func(datax[i, :], datay[j, :]) + kernel_func(datay[i, :], datax[j, :])
                val = val + x_contrib + y_contrib - cross_contrib
    return val / (n * (n - 1))


def kernel_sum_u(datax, datay, kernel_func):
    n = datax.shape[0]
    if n == 1:
        return 0
    else:
        val = 0
        for i in range(n):
            for j in range(n):
                    val = val + kernel_func(datax[i, :], datay[j, :])
        return val / (n ** 2)


def gauss_kernel(X, Y, sigma):
    if np.size(X.shape) == 1:
        X = X[np.newaxis, :]
    if np.size(Y.shape) == 1:
        Y = Y[np.newaxis, :]
    return np.squeeze(np.exp(-cdist(Y, X, metric='sqeuclidean') / (2 * sigma)))
    
    
def evaluate_detection(ground_truth, flagged):
    """Evaluate detection given ground_truth and flagged points (boolean array indicating change indices)"""
    n = ground_truth.shape[0]
    if n != flagged.shape[0]:
        print('error')
    # change flagged into change point, going from 0 to 1
    cp = np.zeros(n, dtype=bool)
    for i in range(n-1):
        if not flagged[i] and flagged[i + 1]:
            cp[i] = 1

    EDD = 0
    not_detected = 0
    FA = 0
    num_change = int(ground_truth.sum())
    where_change = np.concatenate((np.argwhere(ground_truth).flatten(), np.array([n])))

    for i in range(num_change):
        begin_ind = where_change[i]
        end_ind = where_change[i + 1]
        middle_ind = int((begin_ind + end_ind) / 2)
        # EDD
        i = begin_ind
        while i <= middle_ind and not cp[i]:
            i = i+1
        if cp[i]:
            EDD += i - begin_ind
        else:
            not_detected += 1
        # FA
        FA += cp[middle_ind:end_ind].sum()
    return {'EDD': EDD / np.max((num_change - not_detected, 1)), 'not_detected': 100 * not_detected / num_change,
            'false_alarm': FA, 'cp': cp}


def compute_curves(ground_truth, dist, num_points=50, start_coeff=1.3, end_coeff=2, thres_values=np.array([np.nan]),
                   thres_offset=0):
    """ 
    Evaluate performance for several level of thresholds, thres_values can be an array of adaptive threshold at each
    time point.

    Parameters
    ----------
    ground_truth: (N,) binary array,
        ground truth change.
    dist: (N,) array,
        online statistic.
    num_points: int,
        number of points in the scatter plot.
    start_coeff, end_coeff: floats,
        range of threshold (multiplicative).
    thres_values: array,
        values of adaptive threshold (without multiplicative constant). If nan, baseline fixed threshold = mean(dist)
    thres_offset: float,
        value of offset for the adaptive threshold

    Return
    ------
    EDDs: list,
        detection delay time.
    FAs: list,
        false alarms.
    NDs: list,
        missed detections.
    """
    if np.isnan(thres_values)[0]:
        thres_values = np.mean(dist)
    thres_levels = np.linspace(start_coeff, end_coeff, num_points)
    EDDs = np.zeros(num_points)
    FAs = np.zeros(num_points)
    NDs = np.zeros(num_points)
    for i in range(num_points):
        print('Evaluate performance', i, '/', num_points)
        flagged_points = dist > thres_levels[i] * thres_values + thres_offset
        res = evaluate_detection(ground_truth, flagged_points)
        EDDs[i] = res['EDD']
        FAs[i] = res['false_alarm']
        NDs[i] = res['not_detected']
    return EDDs, FAs, NDs
