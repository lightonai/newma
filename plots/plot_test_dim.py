import os

import matplotlib.pyplot as plt
import numpy as np


home = os.path.expanduser('~')
results_dir = home + '/newma-paper/npz4plots/test_dim_running_time/'
algorithm_list = ['newmaRFF', 'newmaFF', 'newmaOPU', 'ScanB', 'MA']
labels = ['NEWMA-RFF', 'NEWMA-FF', 'NEWMA-OPU', 'ScanB', 'MA']
styles = ['o-', 'v--', 'h--', 'd--', 's--']
colors = ['dodgerblue', 'orange', 'mediumpurple', 'green', 'red']

plt.style.use('mystyle.mplstyle')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5.5))
for i, algorithm in enumerate(algorithm_list):
    filename = results_dir + 'test_dim_algo_{}.npz'.format(algorithm)
    data = np.load(filename)
    dims = data['dims']
    running_times = data['running_times']
    ax.loglog(dims, running_times, styles[i], label=labels[i], color=colors[i])
ax.set_xlabel('dim.')
ax.set_ylabel('Time (s)')
ax.legend(loc='best', framealpha=0.75)
plt.savefig('test_dim.pdf', bbox_inches='tight')
