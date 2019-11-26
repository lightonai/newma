import os

import matplotlib.pyplot as plt
import numpy as np


plt.style.use('mystyle.mplstyle')
home = os.path.expanduser('~')
results_dir = home + '/newma-paper/npz4plots/'
filename = results_dir + 'fixed_vs_adaptive.npz'
data = np.load(filename)

EDD_fixed = data['EDD_fixed']
FA_fixed = data['FA_fixed']
ND_fixed = data['ND_fixed']
EDD_adapt95 = data['EDD_adapt95']
FA_adapt95 = data['FA_adapt95']
ND_adapt95 = data['ND_adapt95']
EDD_adapt99 = data['EDD_adapt99']
FA_adapt99 = data['FA_adapt99']
ND_adapt99 = data['ND_adapt99']

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5.5))
ax.plot(FA_fixed, EDD_fixed, 'o-', color='dodgerblue', label='Fixed th.')
ax.plot(FA_adapt95, EDD_adapt95, 'o', color='orange', label='Adaptive th. 0.95', markersize=20)
ax.plot(FA_adapt99, EDD_adapt99, 'o', color='chocolate', label='Adaptive th. 0.99', markersize=20)
ax.set_xlabel('False Alarms')
ax.set_ylabel('Detection Delay')
ax.legend(loc='best', framealpha=0.75)
plt.savefig('EDD_FA_fixed_vs_adapt.pdf', bbox_inches='tight')

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5.5))
ax.plot(FA_fixed, ND_fixed, 'o-', color='dodgerblue', label='Fixed th.')
ax.plot(FA_adapt95, ND_adapt95, 'o', color='orange', label='Adaptive th. 0.95', markersize=20)
ax.plot(FA_adapt99, ND_adapt99, 'o', color='chocolate', label='Adaptive th. 0.99', markersize=20)
ax.set_xlabel('False Alarms')
ax.set_ylabel('Missed Detections (%)')
ax.legend(loc='best', framealpha=0.75)
plt.savefig('MD_FA_fixed_vs_adapt.pdf', bbox_inches='tight')
