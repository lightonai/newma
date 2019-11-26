# -*- coding: utf-8 -*-
import argparse
import os
import time

import numpy as np

import onlinecp.algos as algos
import onlinecp.utils.evaluation as ev
import onlinecp.utils.feature_functions as feat
import onlinecp.utils.gendata as gd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('algo', choices=['newmaRFF', 'newmaFF', 'newmaOPU', 'MA', 'ScanB'])
    parser.add_argument('outfile', type=str, help='name of file to save results')
    parser.add_argument('-n', type=int, default=2000, help='number of samples for each distribution')
    parser.add_argument('-nb', type=int, default=500, help='number of changes of distribution in the series')
    parser.add_argument('-d', type=int, default=100, help='dimensionality of the samples in the time series')
    parser.add_argument('-B', type=int, default=250, help='window size')
    parser.add_argument('-seed', type=int, default=0, help='seed for PRNG')
    parser.add_argument('-show', action='store_true', help='show performance metrics plots')
    parser.add_argument('-dat', type=str, choices=['synth', 'VAD'], default='synth')
    args = parser.parse_args()

    np.random.seed(args.seed)

    algo = args.algo

    # Data generation
    n = args.n
    nb_change = args.nb
    d = args.d
    mult = 1.

    if args.dat == 'synth':
        # parameters of gmm
        k = 10
        std_mean = 0.11  # the bigger, the more change in means
        wishart = 5  # the bigger, the less change in diagonal variances

        X, ground_truth = gd.stream_GMM(d=d, n=n, nb_change=nb_change, std_mean=std_mean, concentration_wishart=wishart,
                                        k=k)
    elif args.dat == 'VAD':
        root_path = os.path.expanduser('~' + '/VAD_local_data/')
        SNR = 0.35  # 0: only noise, 1: only speech
        X, ground_truth = gd.import_vad_data(root_path=root_path, SNR_convex_coeff=SNR, nb_change=nb_change)
        (N, d) = X.shape
        n = int(N / nb_change)

    # common config
    choice_sigma = 'median'
    numel = 100
    data_sigma_estimate = X[:numel]  # data for median trick to estimate sigma
    B = args.B  # window size

    # Scan-B config
    N = 3  # number of windows in scan-B

    # Newma and MA config
    big_Lambda, small_lambda = algos.select_optimal_parameters(B)  # forget factors chosen with heuristic in the paper
    thres_ff = small_lambda
    # number of random features is set automatically with this criterion
    m = int((1 / 4) / (small_lambda + big_Lambda) ** 2)
    m_OPU = 10 * m
    W, sigmasq = feat.generate_frequencies(m, d, data=data_sigma_estimate, choice_sigma=choice_sigma)

    if algo == 'ScanB':
        print('Start algo ', algo, '... (can be long !)')
        detector = algos.ScanB(X[0], kernel_func=lambda x, y: feat.gauss_kernel(x, y, np.sqrt(sigmasq)), window_size=B,
                               nbr_windows=N, adapt_forget_factor=thres_ff)
        detector.apply_to_data(X)
    elif algo == 'MA':
        print('Start algo ', algo, '...')
        print('# RF: ', m)

        def feat_func(x):
            return feat.fourier_feat(x, W)

        detector = algos.MA(X[0], window_size=B, feat_func=feat_func, adapt_forget_factor=thres_ff)
        detector.apply_to_data(X)
    elif algo == 'newmaFF':
        print('Start algo ', algo, '...')
        print('# RF: ', m)
        import onlinecp.utils.fastfood as ff
        FF = ff.Fastfood(sigma=np.sqrt(sigmasq), n_components=m)
        FF.fit(X)
        X = FF.transform(X)

        detector = algos.NEWMA(X[0], forget_factor=big_Lambda, forget_factor2=small_lambda,
                               adapt_forget_factor=thres_ff)
        detector.apply_to_data(X)
    elif algo == 'newmaRFF':  # newma RF
        print('Start algo ', algo, '...')
        print('# RF: ', m)

        def feat_func(x):
            return feat.fourier_feat(x, W)

        detector = algos.NEWMA(X[0], forget_factor=big_Lambda, forget_factor2=small_lambda, feat_func=feat_func,
                               adapt_forget_factor=thres_ff)
        detector.apply_to_data(X)
    else:  # newmaOPU
        print('Start algo ', algo, '...')
        m_OPU = 34570
        m = m_OPU
        print('# RF: ', m)
        try:
            from lightonml.encoding.base import BinaryThresholdEncoder
            from lightonopu.opu import OPU
        except ImportError:
            raise Exception("Please get in touch to use LightOn OPU.")

        opu = OPU(n_components=m)
        opu.open()
        n_levels = 38
        Xencode = np.empty((X.shape[0], n_levels * X.shape[1]), dtype='uint8')
        t = time.time()
        mi, Ma = np.min(X), np.max(X)  # rescale to 0 255
        X = 255 * ((X - mi) / (Ma - mi))
        X = X.astype('uint8')

        for i in range(n_levels):
            Xencode[:, i * X.shape[1]:(i + 1) * X.shape[1]] = X > 65 + i * 5
        del X

        start = time.time()
        X = opu.transform(Xencode)
        print('Projections took:', time.time()-start)
        del Xencode
        opu.device.close()

        # convert to float online to avoid memory error
        mult = 1.5
        detector = algos.NEWMA(X[0], forget_factor=big_Lambda,
                               feat_func=lambda x: x.astype('float32'),
                               forget_factor2=small_lambda, adapt_forget_factor=thres_ff*mult,
                               thresholding_quantile=0.95, dist_func=lambda z1, z2: np.linalg.norm(z1 - z2))
        detector.apply_to_data(X)

    # compute performance metrics
    detection_stat = np.array([i[0] for i in detector.stat_stored])[int(10 * n):]  # padding
    online_th = np.array([i[1] for i in detector.stat_stored])[int(10 * n):]
    ground_truth = ground_truth[int(10 * n):]

    # display perf
    EDD, FA, ND = ev.compute_curves(ground_truth, detection_stat, num_points=30, start_coeff=1.05, end_coeff=1.2)
    EDDth, FAth, NDth = ev.compute_curves(ground_truth, detection_stat, num_points=1,
                                          thres_values=online_th, start_coeff=1, end_coeff=1)

    npz_filename = args.outfile
    np.savez(npz_filename,
             detection_stat=detection_stat, online_th=online_th, ground_truth=ground_truth,
             EDD=EDD, FA=FA, ND=ND, EDDth=EDDth, FAth=FAth, NDth=NDth)

    if args.show:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(FA, EDD, '-o', label='')
        plt.plot(FAth, EDDth, 'o', markersize=20)
        plt.xlabel('False Alarm')
        plt.ylabel('Expected Detection Delay')
        plt.show()

        plt.figure()
        plt.plot(FA, ND, '-o')
        plt.plot(FAth, NDth, 'o', markersize=20)
        plt.xlabel('False Alarm')
        plt.ylabel('Missed Detection')
        plt.show()
