import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("filenames", nargs="+")
home = os.path.expanduser('~')
results_dir = home + '/newma-paper/npz4plots/test_synth_data/'
algorithm_list = ['newmaRFF', 'newmaFF', 'newmaOPU', 'ScanB', 'MA']
labels = ['NEWMA-RFF', 'NEWMA-FF', 'NEWMA-OPU', 'ScanB', 'MA']
styles = ['o-', 'v--', 'h--', 'd--', 's--']
colors = ['blue', 'orange', 'violet', 'green', 'red']
start_coeffs = [1., 1., 1., 1.05, 1.025]
end_coeffs = [1.15, 1.15, 1.15, 1.2, 1.125]

plt.style.use('mystyle.mplstyle')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
for i, algorithm in enumerate(algorithm_list):
    if algorithm == 'newmaOPU':
        m = 34570
        filename = (results_dir +
                    'newma_1.50_algonewmaOPU_datsynth_d100_m34570_n2000_nb500_B250_stdmean0.11_ppf1.64.npz')
        data = np.load(filename)
    else:
        m = 3457
        filename = (results_dir +
                    'newma_1.00_algo{}_datsynth_d100_m3457_n2000_nb500_B250_stdmean0.11_ppf1.64.npz'.format(algorithm))
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

    axes[1].plot(FA, MD, styles[i], label=labels[i], color=colors[i])
    axes[1].plot(FAth, MDth, styles[i][0], label=labels[i]+'0.95', color=colors[i])
plt.legend()
plt.tight_layout()
plt.savefig('synth_data.pdf')
"""
data = np.load(results_dir + 'results_algonewmaOPU_datsynth_d100_m12200_n2000_nb500_B150_stdmean0.11_ppf1.64.npz')
FAth = data['FAth']
EDDth = data['EDDth']
MDth = data['NDth']
axes[0].plot(FAth, EDDth, 's', label='B=150')
axes[1].plot(FAth, MDth, 's', label='B=150')

data = np.load(results_dir + 'res_2.50_algonewmaOPU_datsynth_d100_m12200_n2000_nb500_B150_stdmean0.11_ppf1.64.npz')
FAth = data['FAth']
EDDth = data['EDDth']
MDth = data['NDth']
axes[0].plot(FAth, EDDth, 's', label='B=150 aff2.5')
axes[1].plot(FAth, MDth, 's', label='B=150 aff2.5')

data = np.load(results_dir + 'res1_2.50_algonewmaOPU_datsynth_d100_m12200_n2000_nb500_B150_stdmean0.11_ppf1.64.npz')
FAth = data['FAth']
EDDth = data['EDDth']
MDth = data['NDth']
axes[0].plot(FAth, EDDth, 's', label='B=150 aff2.5 rescale')
axes[1].plot(FAth, MDth, 's', label='B=150 aff2.5 rescale')

data = np.load(results_dir + 'res1_1.50_algonewmaOPU_datsynth_d100_m12200_n2000_nb500_B150_stdmean0.11_ppf1.64.npz')
FAth = data['FAth']
EDDth = data['EDDth']
MDth = data['NDth']
axes[0].plot(FAth, EDDth, 's', label='B=150 aff1.5 rescale 34570')
axes[1].plot(FAth, MDth, 's', label='B=150 aff1.5 rescale 34570')

data = np.load(results_dir + 'res1_1.50_algonewmaOPU_datsynth_d100_m34570_n2000_nb500_B150_stdmean0.11_ppf1.64.npz')
FAth = data['FAth']
EDDth = data['EDDth']
MDth = data['NDth']
axes[0].plot(FAth, EDDth, 's', label='B=150 aff1.5 rescale')
axes[1].plot(FAth, MDth, 's', label='B=150 aff1.5 rescale')

data = np.load(results_dir + 'res1_1.50_algonewmaOPU_datsynth_d100_m34570_n2000_nb500_B250_stdmean0.11_ppf1.64.npz')
FAth = data['FAth']
EDDth = data['EDDth']
MDth = data['NDth']
axes[0].plot(FAth, EDDth, 's', label='aff1.5 rescale 256*m')
axes[1].plot(FAth, MDth, 's', label='aff1.5 rescale 256*m')


data = np.load(results_dir + 'newma_1.50_algonewmaOPU_datsynth_d100_m34570_n2000_nb500_B250_stdmean0.11_ppf1.64.npz')
FAth = data['FAth']
EDDth = data['EDDth']
MDth = data['NDth']
axes[0].plot(FAth, EDDth, 's', label='bs2000')
axes[1].plot(FAth, MDth, 's', label='bs2000')

axes[0].set_xlabel('False Alarms')
axes[0].set_ylabel('Detection Delay')
axes[0].legend(loc='best', framealpha=1)
axes[1].set_xlabel('False Alarms')
axes[1].set_ylabel('Missed Detections (%)')
axes[1].legend(loc='best', framealpha=1)
plt.show()
"""