# -*- coding: utf-8 -*-
"""
Test script
"""
import time

import numpy as np
import matplotlib.pyplot as plt

import fastfood as ff
import gendata as gd
import onlinecp as ocp


# parameters
d = 100  # dimension of each data point
n = 2000  # interval between changes
nb_change = 100  # number of changes: must be high enough if you want to display

# parameters of gmm
k = 10
std_mean = 0.11  # the bigger, the more changes in means
wishart = 5  # the bigger, the less changes in diagonal variances

# parameters of the algorithm
algo = 'newmaRF'  # 'scanB', 'newmaFF'

sigma_sq = d  # squared variance of gaussian kernel
B = 250  # window size
N = 3  # number of windows in scan-B

size_batch = 10000  # for the FastFood transform, we are not truly "online" but do the transform by batch
m = 2000  # number of random features
c = 2
small_lambda = (c**(1/B)-1)/(c**((B+1)/B)-1)  # small lambda
big_Lambda = c*small_lambda  # Big Lambda
updt_thres = small_lambda/2  # eta

np.random.seed(0)

# generate data
X, ground_truth = gd.stream_GMM(d=d, n=n, nb_change=nb_change, std_mean=std_mean, concentration_wishart=wishart, k=k)

if algo == 'scanB':
    print('Start algorithm Scan-B on ', nb_change * n, ' samples in dimension ', d, ' (can be long !).')
    ocp_object = ocp.ScanB(d, store_result=True, updt_coeff_thres=updt_thres, B=B, N=N,
                           kernel_func=lambda x, y: ocp.gauss_kernel(x, y, sigma_sq))
    t = time.time()
    ocp_object.apply_to_data(X)  # actual computations
    time_method = time.time() - t
    print('time:', time_method)
elif algo == 'newmaFF':
    FF = ff.Fastfood(sigma=np.sqrt(sigma_sq), n_components=m)
    FF.fit(X)  # initialize fastfood
    print('Start algorithm NEWMA-FF on ', nb_change * n, ' samples in dimension ', d, ' with ', m, ' random features.')
    updt_func = lambda x: x  # we perform the random feature outside of Newma, the code we use is not performant "online"
    ocp_object = ocp.Newma(store_result=True, updt_func=updt_func, updt_coeff=small_lambda, updt_coeff2=big_Lambda,
                           updt_coeff_thres=updt_thres)
    
    nb_batch = int(np.ceil(n * nb_change / size_batch))
    t = time.time()
    # transform and process data by batch
    for b in range(nb_batch):
        print('batch ', b)
        Xb = X[b * size_batch:np.min((n * nb_change, (b + 1) * size_batch)), :]
        Xb = FF.transform(Xb)
        ocp_object.apply_to_data(Xb)
    time_method = time.time() - t
    print('time: ', time_method)
        
else:  # newma RF
    W = np.random.randn(m, d)/np.sqrt(sigma_sq)
    updt_func = lambda x: ocp.fourier_feature(x, W)  # /np.sqrt(m)
    t = time.time()
    print('Start algorithm NEWMA-RF on ', nb_change*n, ' samples in dimension ', d, ' with ', m, ' random features.')
    ocp_object = ocp.Newma(store_result=True, updt_func=updt_func, updt_coeff=small_lambda, updt_coeff2=big_Lambda,
                           updt_coeff_thres=updt_thres)
    ocp_object.apply_to_data(X)  # actual computations
    time_method = time.time() - t
    print('time: ', time_method)

# plot result
padding = int(5 * n / 2)  # we remove the first point because of initialization
num_points = 20
dist = np.array(ocp_object.dist)[padding:]
thres = np.array(ocp_object.thres_debug)[padding:] # adaptive threshold
gt = ground_truth[padding:]
EDD, FA, MD = ocp.compute_curves(gt, dist, start_coeff=1.1, end_coeff=1.2, thres_values=thres, num_points=num_points)

plt.figure()
plt.plot(dist[:10000])
plt.plot(1.12 * thres[:10000])
plt.plot((np.max(dist[:10000]) - np.min(dist[:10000])) * gt[:10000] + np.min(dist[:10000]))
plt.legend(('Stat.', 'Adapt Th.', 'Changes'))
plt.show()

plt.figure()
plt.plot(FA, EDD, 'o:')
plt.xlabel('False alarms')
plt.ylabel('Detection Delay')
plt.show()

plt.figure()
plt.plot(FA, MD, 'o:')
plt.xlabel('False alarms')
plt.ylabel('Missed Detections (%)')
plt.show()
