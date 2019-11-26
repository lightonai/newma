import os

import matplotlib.pyplot as plt
import numpy as np


home = os.path.expanduser('~')
results_dir = home + '/newma-paper/npz4plots/test_VAD_data/'
algorithm_list = ['newmaRF', 'newmaOPU', 'ScanB']
labels = ['NEWMA-RFF', 'NEWMA-OPU', 'ScanB']
styles = ['o-', 'h--', 'd--']
colors = ['blue', 'violet', 'green']
start_coeffs = [1.7, 1.7, 2.2]
end_coeffs = [1.9, 1.9, 2.35]

plt.style.use('mystyle.mplstyle')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
for i, algorithm in enumerate(algorithm_list[:4]):
    if algorithm == 'newmaOPU':
        m = 12200
        filename = (results_dir + 'results_algo{}_datVAD_d129_'.format(algorithm) +
                    'm{}_n1250_nb300_B150_stdmean_ppf1.64.npz'.format(m))
        data = np.load(filename)

    else:
        m = 1220
        filename = (results_dir + 'results_algo{}_datVAD_d129_'.format(algorithm) +
                    'm{}_n1250_nb300_B150_stdmean_ppf1.64.npz'.format(m))
        data = np.load(filename)

    detection_stat = data['detection_stat']
    ground_truth = data['ground_truth']
    import onlinecp.utils.evaluation as ev

    EDD, FA, MD = ev.compute_curves(ground_truth, detection_stat, num_points=30, start_coeff=start_coeffs[i],
                                    end_coeff=end_coeffs[i])

    FAth = data['FAth']
    EDDth = data['EDDth']
    MDth = data['NDth']

    axes[0].plot(FA, EDD, styles[i], label=labels[i], color=colors[i])
    axes[0].plot(FAth, EDDth, styles[i][0], label=labels[i]+'0.95', color=colors[i])
    axes[0].set_xlabel('False Alarms')
    axes[0].set_ylabel('Detection Delay')
    axes[0].legend(loc='best', framealpha=1)

    axes[1].plot(FA, MD, styles[i], label=labels[i], color=colors[i])
    axes[1].plot(FAth, MDth, styles[i][0], label=labels[i]+'0.95', color=colors[i])
    axes[1].set_xlabel('False Alarms')
    axes[1].set_ylabel('Missed Detections (%)')
    axes[1].legend(loc='best', framealpha=1)
plt.savefig('VAD_data.pdf')
