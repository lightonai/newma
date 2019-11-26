import os

import matplotlib.pyplot as plt
import numpy as np


home = os.path.expanduser('~')
results_dir = home + '/newma-paper/npz4plots/test_B_running_time/'
algorithm_list = ['newmaRFF', 'ScanB']
labels = ['NEWMA-RFF', 'ScanB']
styles = ['o-', 'd--']
colors = ['dodgerblue', 'green']

plt.style.use('mystyle.mplstyle')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5.5))
for i, algorithm in enumerate(algorithm_list):
    filename = results_dir + 'test_B_runningtime_algo_{}.npz'.format(algorithm)
    data = np.load(filename)
    window_sizes = data['windows']
    running_times = data['running_times']
    ax.plot(window_sizes, running_times, styles[i], label=labels[i], color=colors[i])
ax.set_xlabel('Window size')
ax.set_ylabel('Time (s)')
ax.legend(loc='best', framealpha=0.75)
plt.yticks([0, 5, 10, 15, 20])
plt.savefig('test_B_runningtime.pdf', bbox_inches='tight')
