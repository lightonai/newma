# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import numpy as np
import onlinecp as ocp
from matplotlib import pyplot as plt
import scipy.stats as scp

def generate_states_newma(num_el, tau, maxbound = 1):
    grid = np.concatenate((np.linspace(0,maxbound,num_el),np.linspace(-maxbound,0,num_el)))[:-1]
    l = grid.shape[0]
    res = []
    for i in range(l):
        for j in range(l):
            ai = grid[i]
            aj = grid[j]
            if np.abs(ai-aj)<tau:
                res.append([ai,aj])
    return {'states': np.array(res), 'grid': grid}

def generate_states_ewma(num_el, tau):
    return np.concatenate((np.linspace(0,tau,num_el),np.linspace(-tau,0,num_el)))[:-1]

# transition probability
def proba_trans_newma(a,b,eps,eta,eta2, cdf = lambda x: scp.norm.cdf(x)):
    low1 = (b[0]-(1-eta)*a[0]-eps/2)/eta
    up1 = (b[0]-(1-eta)*a[0]+eps/2)/eta
    low2 = (b[1]-(1-eta2)*a[1]-eps/2)/eta2
    up2 = (b[1]-(1-eta2)*a[1]+eps/2)/eta2
    low = np.max((low1,low2))
    up = np.min((up1,up2))
    if low>up:
        return 0
    else:
        return cdf(up)-cdf(low)
    
def proba_trans_ewma(a,b,eps,eta, cdf = lambda x: scp.norm.cdf(x)):
    low = (b-(1-eta)*a-eps/2)/eta
    up = (b-(1-eta)*a+eps/2)/eta
    return cdf(up)-cdf(low)

    
if __name__ == '__main__':
    save_res = True
    taus = np.linspace(0.2,0.45,3) # fixed thresholds to test
    num_el = 35 # number of elements in the grid (will be squared for newma)
    max_bound = 1.5 # maximum bound of sketch for newma, should not cross
    updt = 0.1 # forgetting factor
    c = 2
    updt2 = c*updt
    num_trial = 1000
    
    arl_theory_newma = np.zeros(len(taus))
    arl_theory_ewma = np.zeros(len(taus))
    arl_expe_newma = np.zeros(len(taus))
    arl_expe_ewma = np.zeros(len(taus))
    for ind in range(len(taus)):
        print('Trial ',ind+1,'/',len(taus))
        tau = taus[ind]
        # newma theory
        print('Theory...')
        statesg = generate_states_newma(num_el,tau, max_bound)
        eps = statesg['grid'][1] - statesg['grid'][0]
        states = statesg['states']
        num_states = states.shape[0]
        A = np.repeat(states[:,np.newaxis,:],num_states,axis=1)
        B = np.repeat(states[np.newaxis,:,:],num_states,axis=0)
        nonzerosproba = np.nonzero(np.maximum((B[:,:,0]-(1-updt)*A[:,:,0]-eps/2)/updt,(B[:,:,1]-(1-updt2)*A[:,:,1]-eps/2)/updt2)<np.minimum((B[:,:,0]-(1-updt)*A[:,:,0]+eps/2)/updt,(B[:,:,1]-(1-updt2)*A[:,:,1]+eps/2)/updt2))
        num_nonzeros = nonzerosproba[0].shape[0]
        print(num_states, ' states, ', num_nonzeros, ' non-zeros.')
        transMat = np.zeros((num_states,num_states))
        for pind in range(num_nonzeros):
            i = nonzerosproba[0][pind]
            j = nonzerosproba[1][pind]
            transMat[i,j] = proba_trans_newma(states[i,:],states[j,:],eps,updt,updt2)
        arl_theory_newma[ind] = np.linalg.solve(np.eye(num_states)-transMat,np.ones(num_states))[0] #np.sum(v, axis=1)[0]
        
        # newma practice
        print('Simulations...')
        arl_expe = np.zeros(num_trial)
        for i in range(num_trial):
            print('trial ', i+1,'/',num_trial)
            newma = ocp.Newma(updt_coeff = updt, updt_coeff2 = updt2)
            detect = False
            runtime = 1
            while not detect:
                x = np.random.randn(1)
                res = newma.update(x)
                if newma.sketch>max_bound:
                    print('Warning! Sketch out of bound! Val:',newma.sketch)
                d = res['dist']
                if d>tau:
                    detect = True
                else:
                    runtime = runtime + 1
            arl_expe[i] = runtime
        arl_expe_newma[ind] = np.mean(arl_expe)
        
        # ewma theory
        print('EWMA...')
        states_e = generate_states_ewma(num_el,tau)
        eps_e = states_e[1] - states_e[0]
        num_states_e = states_e.shape[0]
        print(num_states_e, ' states generated.')
        transMat_e = np.zeros((num_states_e,num_states_e))
        for i in range(num_states_e):
            for j in range(num_states_e):
                transMat_e[i,j] = proba_trans_ewma(states_e[i],states_e[j],eps_e,updt)
        arl_theory_ewma[ind] = np.linalg.solve(np.eye(num_states_e)-transMat_e,np.ones(num_states_e))[0]
        
        # newma practice
        arl_expe = np.zeros(num_trial)
        for i in range(num_trial):
            ewma = ocp.Newma(updt_coeff = updt, updt_coeff2 = 0)
            detect = False
            runtime = 1
            while not detect:
                x = np.random.randn(1)
                res = ewma.update(x)
                d = res['dist']
                if d>tau:
                    detect = True
                else:
                    runtime = runtime + 1
            arl_expe[i] = runtime
        arl_expe_ewma[ind] = np.mean(arl_expe)
        
    plt.figure()
    plt.semilogy(taus, arl_theory_newma, taus, arl_expe_newma, taus, arl_theory_ewma, taus, arl_expe_ewma)
    plt.legend(('Theo NEWMA', 'Simu NEWMA', 'Theo NEWMA', 'Simu EWMA'))
    plt.ylabel('T')
    plt.xlabel('Threshold')