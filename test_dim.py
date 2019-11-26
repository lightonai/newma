# -*- coding: utf-8 -*-
import argparse
import time

import numpy as np

import onlinecp.algos as algos
import onlinecp.utils.gendata as gd
import onlinecp.utils.feature_functions as feat


parser = argparse.ArgumentParser()
parser.add_argument('algo', choices=['newmaRFF', 'newmaFF', 'newmaOPU', 'MA', 'ScanB'],
                    help='algorithm to use for evaluation')

args = parser.parse_args()
algo = args.algo

# parameters for data
num_tests = 10
dims = np.ceil(10 ** np.linspace(1, 5, num_tests))  # dims to test
n = 12000
nb_change = 1
k = 20
std_mean = 0.2

# parameters for algorithms
save_res = True

window_size = 250
N = 3
ff, ff2 = algos.select_optimal_parameters(window_size)
if algo == 'newmaOPU':
    m = int(10 * (ff + ff2)**(-2))
else:
    m = int(0.25 * (ff + ff2)**(-2))
print(algo, m)
thresh_ff = ff2

np.random.seed(12345)
running_times = []
for d_ind in range(num_tests):
    d = int(dims[d_ind])
    sigmasq = d
    print('Performing test ', d_ind + 1, ' over ', num_tests, ' d = ', d)
    # generate data
    X, ground_truth = gd.stream_GMM(d=d, n=n, nb_change=nb_change, std_mean=std_mean, concentration_wishart=10, k=k)

    if algo == 'ScanB':
        print('Scan-B')
        ocpobj = algos.ScanB(X[0], kernel_func=lambda x, y: feat.gauss_kernel(x, y, np.sqrt(sigmasq)),
                             window_size=window_size, nbr_windows=N, adapt_forget_factor=thresh_ff, store_values=False)
        t = time.time()
        ocpobj.apply_to_data(X)  # actual computations
        time_method = time.time() - t
        print('time:', time_method)
    elif algo == 'MA':
        print('MA')
        W, _ = feat.generate_frequencies(m, d, data=X[:100])
        updt_func = lambda x: feat.fourier_feat(x, W)
        ocpobj = algos.MA(X[0, :], window_size=window_size, feat_func=updt_func, adapt_forget_factor=thresh_ff,
                          store_values=False)
        t = time.time()
        ocpobj.apply_to_data(X)  # actual computations
        time_method = time.time() - t
        print('time:', time_method)
    else:
        if algo == 'newmaOPU':
            print('OPU')
            from lightonml.encoding.base import BinaryThresholdEncoder
            from lightonopu.opu import OPU
            from lightonopu import types

            opu = OPU(n_components=m, disable_pbar=True, verbose_level=0,
                      features_fmt=types.FeaturesFormat.lined,
                      dmd_strategy=types.DmdRoiStrategy.full)
            opu.open()
            encoder = BinaryThresholdEncoder(threshold_enc=125)
            mi = np.min(X.flatten())  # rescale to 0 255
            Ma = np.max(X.flatten())
            X = 255 * ((X - mi) / (Ma - mi))
            t = time.time()
            X = opu.transform(encoder.transform(X))

            def updt_func(x): return x
        elif algo == 'newmaFF':
            print('FF')
            from onlinecp.utils.fastfood import Fastfood
            fastfood = Fastfood(n_components=m)
            fastfood.fit(X)
            t = time.time()
            X = fastfood.transform(X)

            def updt_func(x): return x
        else:
            print('RFF')
            W, _ = feat.generate_frequencies(m, d, data=X[:100])
            updt_func = lambda x: feat.fourier_feat(x, W)
            t = time.time()

        ocpobj = algos.NEWMA(X[0], forget_factor=ff, forget_factor2=ff2, feat_func=updt_func,
                             adapt_forget_factor=thresh_ff, store_values=False)

        ocpobj.apply_to_data(X)  # actual computations
        time_method = time.time() - t
        print('Execution time: ', time_method)
    if algo == 'newmaOPU':
        opu.device.close()
    running_times.append(time_method)

# save results for later plot
np.savez('test_dim_algo_{}_12k.npz'.format(algo), dims=dims, running_times=np.array(running_times))

"""
if save_res:
    np.save('../npz4plots/test_dim__{}_algo_{}.npz'.format(d, algo),
            {'dist': np.array([stat[0] for stat in ocpobj.stat_stored]),
             'thres': np.array([stat[1] for stat in ocpobj.stat_stored]),
             'result': np.array([stat[2] for stat in ocpobj.stat_stored]),
             'gt': ground_truth,
             'd': d,
             'n': n,
             'm': m,
             'time': time_method})
"""
