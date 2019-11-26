# -*- coding: utf-8 -*-
import argparse
import time

import numpy as np

import onlinecp.algos as algos
import onlinecp.utils.gendata as gd
import onlinecp.utils.feature_functions as feat


parser = argparse.ArgumentParser()
parser.add_argument('algo', choices=['newmaRFF', 'ScanB'],
                    help='algorithm to use for evaluation')

args = parser.parse_args()
algo = args.algo

# parameters for data
np.random.seed(12345)
num_tests = 15
d = 100
n = 3000
nb_change = 1
k = 20
std_mean = 0.2

# generate data
sigmasq = d
X, ground_truth = gd.stream_GMM(d=d, n=n, nb_change=nb_change, std_mean=std_mean, concentration_wishart=10, k=k)

# parameters for algorithms
m = 2000

save_res = True

window_size_list = np.linspace(100, 500, num_tests)
N = 3

running_times = []
for i, window_size in enumerate(window_size_list):
    window_size = int(window_size)
    print('Performing test ', i + 1, ' over ', num_tests, ' window size = ', window_size)
    ff, ff2 = algos.select_optimal_parameters(window_size)
    thresh_ff = ff2

    if algo == 'ScanB':
        print('Scan-B')
        ocpobj = algos.ScanB(X[0], kernel_func=lambda x, y: feat.gauss_kernel(x, y, np.sqrt(sigmasq)),
                             window_size=window_size, nbr_windows=N, adapt_forget_factor=thresh_ff, store_values=False)
        t = time.time()
        ocpobj.apply_to_data(X)  # actual computations
        time_method = time.time() - t
        print('time:', time_method)
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
    running_times.append(time_method)

# save results for later plot
np.savez('test_B_runningtime_algo_{}.npz'.format(algo), windows=window_size_list,
         running_times=np.array(running_times))
