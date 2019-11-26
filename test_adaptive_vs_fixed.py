# -*- coding: utf-8 -*-
import argparse
import os
import time

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

import onlinecp.algos as algos
import onlinecp.utils.evaluation as ev
import onlinecp.utils.feature_functions as feat
import onlinecp.utils.fastfood as ff
import onlinecp.utils.gendata as gd

try:
    """
    Please get in touch to use LightOn OPU.
    """
    from lightonml.encoding.base import BinaryThresholdEncoder
    from lightonml.random_projections.opu import OPURandomMapping
    from lightonopu.opu import OPU
except ImportError:
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=2000, help='number of samples for each distribution')
    parser.add_argument('-nb', type=int, default=500, help='number of changes of distribution in the series')
    parser.add_argument('-d', type=int, default=100, help='dimensionality of the samples in the time series')
    parser.add_argument('-B', type=int, default=250, help='window size')
    parser.add_argument('-seed', type=int, default=0, help='seed for PRNG')
    parser.add_argument('-show', action='store_true', help='show performance metrics plots')
    args = parser.parse_args()

    np.random.seed(args.seed)
    algo = 'newmaRF'
    # Data generation
    n = args.n
    nb_change = args.nb
    d = args.d

    # parameters of gmm
    k = 10
    std_mean = 0.11  # the bigger, the more change in means
    wishart = 5  # the bigger, the less change in diagonal variances

    X, ground_truth = gd.stream_GMM(d=d, n=n, nb_change=nb_change, std_mean=std_mean, concentration_wishart=wishart,
                                    k=k)

    # common config
    choice_sigma = 'median'
    numel = 100
    data_sigma_estimate = X[:numel]  # data for median trick to estimate sigma
    B = args.B  # window size

    # Newma config
    big_Lambda, small_lambda = algos.select_optimal_parameters(B)  # forget factors chosen with heuristic in the paper
    print('Chose Lambda = {:.4f} and lambda = {:.4f}'.format(big_Lambda, small_lambda))
    thres_ff = small_lambda
    # number of random features is set automatically with this criterion
    m = int((1 / 4) / (small_lambda + big_Lambda) ** 2)
    print('Number of RFs: {}'.format(m))
    W, sigmasq = feat.generate_frequencies(m, d, data=data_sigma_estimate, choice_sigma=choice_sigma)

    print('Start algo ', algo, ' with fixed threshold')

    def feat_func(x):
        return feat.fourier_feat(x, W)

    detector95 = algos.NEWMA(X[0], forget_factor=big_Lambda, forget_factor2=small_lambda, feat_func=feat_func,
                             adapt_forget_factor=thres_ff)
    detector95.apply_to_data(X)

    # compute performance metrics
    detection_stat95 = np.array([i[0] for i in detector95.stat_stored])[int(10 * n):]  # padding
    online_th95 = np.array([i[1] for i in detector95.stat_stored])[int(10 * n):]
    ground_truth = ground_truth[int(10 * n):]

    if args.show:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(detection_stat95[:4 * n])
        plt.plot(online_th95[:4 * n])
        plt.plot(1.2 * np.max(detection_stat95[:4 * n]) * ground_truth[:4 * n], '--', color='k')
        plt.legend([algo, 'Threshold', 'True Changes'], framealpha=1, ncol=1, handletextpad=0.1)
        plt.show()

    # display perf
    EDD, FA, ND = ev.compute_curves(ground_truth, detection_stat95, num_points=30, start_coeff=1.05, end_coeff=1.2)
    EDDth95, FAth95, NDth95 = ev.compute_curves(ground_truth, detection_stat95, num_points=1,
                                                thres_values=online_th95, start_coeff=1, end_coeff=1)

    detector99 = algos.NEWMA(X[0], forget_factor=big_Lambda, forget_factor2=small_lambda, feat_func=feat_func,
                             adapt_forget_factor=thres_ff, thresholding_quantile=0.99)
    detector99.apply_to_data(X)

    # compute performance metrics
    detection_stat99 = np.array([i[0] for i in detector99.stat_stored])[int(10 * n):]  # padding
    online_th99 = np.array([i[1] for i in detector99.stat_stored])[int(10 * n):]
    ground_truth = ground_truth[int(10 * n):]

    if args.show:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(detection_stat99[:4 * n])
        plt.plot(online_th99[:4 * n])
        plt.plot(1.2 * np.max(detection_stat99[:4 * n]) * ground_truth[:4 * n], '--', color='k')
        plt.legend([algo, 'Threshold', 'True Changes'], framealpha=1, ncol=1, handletextpad=0.1)
        plt.show()

    # display perf
    EDD, FA, ND = ev.compute_curves(ground_truth, detection_stat99, num_points=30, start_coeff=1.05, end_coeff=1.2)
    EDDth99, FAth99, NDth99 = ev.compute_curves(ground_truth, detection_stat99, num_points=1,
                                                thres_values=online_th99, start_coeff=1, end_coeff=1)

    npz_filename = 'threshold_comparison.npz'
    np.savez(npz_filename, algo=algo, d=d, m=m, n=n, nb_change=nb_change, B=B, std_mean=std_mean,
             EDD_fixed=EDD, FA_fixed=FA, ND_fixed=ND,
             EDD_adapt95=EDDth95, FA_adapt95=FAth95, ND_adapt95=NDth95,
             EDD_adapt99=EDDth99, FA_adapt99=FAth99, ND_adapt99=NDth99)
