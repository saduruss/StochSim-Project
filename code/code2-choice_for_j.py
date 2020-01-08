import numpy as np 
from numpy import matlib
import scipy.stats as st 
import matplotlib.pyplot as plt 
import sobol_new as sn # file to generate Sobol low discrepancy sequance (admits dimensions up to 21201)
import time

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'],
     'size' : '12'})
rc('text', usetex=True)
rc('lines', linewidth=2)
plt.rcParams['axes.facecolor']='w'
import matplotlib
matplotlib.rcParams['text.latex.preamble'] = [
    r'\usepackage{amsmath}',
    r'\usepackage{amssymb}']

def genPsi(type, xi, t, r, sigma, S0, K):
    dt = np.diff(t)
    W = np.append([0], np.cumsum(np.sqrt(dt)*xi))
    S = S0*np.exp((r - sigma**2/2)*t + sigma*W)
    if type == 1:
        val = (np.abs(np.mean(S) - K) + (np.mean(S) - K))/2
    elif type == 2:
        val = (np.mean(S) - K) > 0
    return val

def evaluate(type, x, r=0.1, sigma=0.1, T=1, S0=100, K=100):
    d = x.shape[0]
    M = x.shape[1]
    val = np.zeros(M)
    t = np.linspace(0, T, d+1)
    if type == 1:
        for j in range(M):
            xi = st.norm.ppf(x[:,j])
            val[j] = genPsi(type, xi, t, r, sigma, S0, K)
    elif type == 2:
        for j in range(M):
            xi = st.norm.ppf(x[:,j])
            val[j] = genPsi(type, xi, t, r, sigma, S0, K)
    return val

def CMC(type, d, M):
    x = np.random.random((d, M))
    data = evaluate(type, x)
    est = np.mean(data)
    err_est = np.std(data)/np.sqrt(M)
    #err = np.abs(est - exact)
    return est, err_est

def QMC(type, d, N, K):
    x = sn.generate_points(N, d, 0)
    x = x.T
    data = np.zeros(K)
    for i in range(K):
        dat = evaluate(type, np.mod(x + matlib.repmat(np.random.random(size=(int(d),1)), 1, int(N)),1))
        data[i] = np.mean(dat)
    est = np.mean(data)
    err_est = 3*np.std(data)/np.sqrt(K)
    #err = np.abs(est - exact)
    return est, err_est


def p_Simpson(type, xi, t, r, sigma, S0, K, j_pos, U_star=0):
    M_Simpson = 2**6
    h = 1/(M_Simpson+1)
    x_interior = st.norm.ppf(np.linspace(U_star, 1-h, M_Simpson+1)) # M point inside
    dt = np.diff(t)
    
    f = np.zeros(M_Simpson+1)
    for l in range(M_Simpson+1):
        x = np.insert(xi,j_pos,x_interior[l])
        W = np.append([0], np.cumsum(np.sqrt(dt)*x))
        S = S0*np.exp((r - sigma**2/2)*t + sigma*W)
        
        if type == 1:
            f[l] = np.maximum(np.mean(S)-K, 0)
        elif type == 2:
            f[l] = (np.mean(S) - K) > 0
    
    
    val = f[0] + 4*f[1] + f[-1]
    for k in np.arange(1,M_Simpson/2):
        val = val + 2*f[int(2*k)] + 4*f[int(2*k+1)]
    return h/3 * val

def pre_int_evaluate(type, x, j_pos, r=0.1, sigma=0.1, T=1, S0=100, K=100):
    d = x.shape[0]
    M = x.shape[1]
    val = np.zeros(M)
    t = np.linspace(0, T, d+2)
    for j in range(M):
        xi = st.norm.ppf(x[:,j])
        val[j] = p_Simpson(type, xi, t, r, sigma, S0, K, j_pos)
    return val

def pre_int_CMC(type, d, M, j_pos=0):
    x = np.random.random((d-1, M))
    data = pre_int_evaluate(type, x, j_pos)
    est = np.mean(data)
    err_est = np.std(data)/np.sqrt(M)
    #err = np.abs(est - exact)
    return est, err_est

def pre_int_QMC(type, d, N, K, j_pos=0):
    x = sn.generate_points(N, d-1, 0)
    x = x.T
    data = np.zeros(K)
    for i in range(K):
        dat = pre_int_evaluate(type, np.mod(x + matlib.repmat(np.random.random(size=(int(d-1),1)), 1, int(N)),1), j_pos)
        data[i] = np.mean(dat)
    est = np.mean(data)
    err_est = 3*np.std(data)/np.sqrt(K)
    #err = np.abs(est - exact)
    return est, err_est


#simulation parameters
m = 32
r = 0.1
sigma = 0.1
T = 1
S0 = 100
K = 100

# compare different j for CMC or QMC with pre-int.
Mlist = 2**np.arange(5,10)
Nlist = 2**np.arange(7,13)

nM = np.size(Mlist)
nN = np.size(Nlist)

cmc_est_pre0, cmc_err_est_pre0 = np.zeros(nN), np.zeros(nN)
cmc_est_pre1, cmc_err_est_pre1 = np.zeros(nN), np.zeros(nN)
cmc_est_pre2, cmc_err_est_pre2 = np.zeros(nN), np.zeros(nN)
qmc_est_pre0, qmc_err_est_pre0 = np.zeros(nN), np.zeros(nN)
qmc_est_pre1, qmc_err_est_pre1 = np.zeros(nN), np.zeros(nN)
qmc_est_pre2, qmc_err_est_pre2 = np.zeros(nN), np.zeros(nN)

fig, axes = plt.subplots(nrows = 5, ncols = 2, figsize = (16,30))
axes = axes.flatten()
types = 1 # change to 1 or 2

K = 20
for j in range(nM):
    print('Computation for m = '+str(Mlist[j])+' ...')
    for i in range(nN):
        cmc_est_pre0[i], cmc_err_est_pre0[i] = pre_int_CMC(types, Mlist[j], Nlist[i], j_pos=0)
        qmc_est_pre0[i], qmc_err_est_pre0[i] = pre_int_QMC(types, Mlist[j], Nlist[i]/K, K, j_pos=0)
        
        cmc_est_pre1[i], cmc_err_est_pre1[i] = pre_int_CMC(types, Mlist[j], Nlist[i], j_pos=Mlist[j]-1)
        qmc_est_pre1[i], qmc_err_est_pre1[i] = pre_int_QMC(types, Mlist[j], Nlist[i]/K, K, j_pos=Mlist[j]-1)
        
        cmc_est_pre2[i], cmc_err_est_pre2[i] = pre_int_CMC(types, Mlist[j], Nlist[i], j_pos=int(Mlist[j]/2))
        qmc_est_pre2[i], qmc_err_est_pre2[i] = pre_int_QMC(types, Mlist[j], Nlist[i]/K, K, j_pos=int(Mlist[j]/2))


    # plot:
    ax = axes[int(2*j)]
    ax.loglog(Nlist, cmc_err_est_pre0, '-', label = 'CMC error estimate j = 0')
    ax.loglog(Nlist, cmc_err_est_pre1, '-', label = 'CMC error estimate j = M/2')
    ax.loglog(Nlist, cmc_err_est_pre2, '-', label = 'CMC error estimate j = M')
    ax.loglog(Nlist, Nlist**-0.5, '--', label = r'$N^{-1/2}$',color='gray')
    if types == 2:
        ax.loglog(Nlist, Nlist**-1.0, ':',  label = r'$N^{-1}$',color='gray')
    ax.set_title('CMC error estimate for '+r'$\Psi_'+str(types)+'$, $m='+str(Mlist[j])+'$')
    ax.grid(True,which='both') 
    ax.legend()

    ax = axes[int(2*j+1)]
    ax.loglog(Nlist, qmc_err_est_pre0, '-', label = 'QMC error estimate j = 0')
    ax.loglog(Nlist, qmc_err_est_pre1, '-', label = 'QMC error estimate j = M/2')
    ax.loglog(Nlist, qmc_err_est_pre2, '-', label = 'QMC error estimate j = M')
    ax.loglog(Nlist, Nlist**-0.5, '--', label = r'$N^{-1/2}$',color='gray')
    if types == 2:
        ax.loglog(Nlist, Nlist**-1.0, ':',  label = r'$N^{-1}$',color='gray')
    ax.set_title('QMC error estimate for '+r'$\Psi_'+str(types)+'$, $m='+str(Mlist[j])+'$')
    ax.grid(True,which='both') 
    ax.legend()

plt.savefig('./figures/ex2_j_test_error_Psi_'+str(types)+'.pdf', format='pdf', bbox_inches='tight')
plt.show()
