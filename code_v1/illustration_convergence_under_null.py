'''
Illustration of the convergence in law of NEWMA when the forgetting factor goes to 0
'''

import numpy as np
import onlinecp as ocp
from matplotlib import pyplot as plt

def features_Fourier(x, eigenvalues):
    # dim 1
    n = eigenvalues.shape[0]
    return np.sqrt(2)*eigenvalues*np.cos(2*np.pi*np.arange(1,n+1)*x)

if __name__ == '__main__':
    n = 20 # number of eigenvalues
    eigenvalues = 3*np.random.rand(n)
    updt = 0.01 # forgetting factor
    c = 2
    t = int(2*np.log(1/updt)/updt)
    num_trial = 1000
    
    # stat
    print('Theory...')
    stat_theory = np.zeros(num_trial)
    const = (c-1)**2/(2*c+2)
    for i in range(num_trial):
        stat_theory[i] = const*np.dot(eigenvalues**2, np.random.randn(n)**2)
    
    # test newma
    print('Simulations of NEWMA...')
    stat_newma = np.zeros(num_trial)
    for i in range(num_trial):
        print('Trial ', i+1, ' over ', num_trial)
        data = np.random.rand(t,1)
        newma = ocp.Newma(updt_coeff =updt, updt_coeff2 = c*updt, updt_func = lambda x:features_Fourier(x,eigenvalues))
        newma.apply_to_data(data)
        stat_newma[i] = np.linalg.norm(newma.sketch - newma.sketch2)**2/updt
        
    #%% plot
    plt.figure()
    plt.hist(stat_theory,bins=np.arange(1, 35, 0.5))
    plt.hist(stat_newma,bins=np.arange(1, 35, 0.5),alpha=0.7)
    plt.legend(['Theory', 'Simu.'])
    plt.ylabel('Count')
    plt.xlabel('Value')