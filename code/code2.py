import numpy as np 
from numpy import matlib
import scipy.stats as st 
import matplotlib.pyplot as plt 
import sobol_new as sn # generate Sobol' low discrepancy sequance (admits dimensions up to 21201)

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


def Simpson(type, M, U_star, S, dt, r, sigma, K):
    if U_star == 1:
        #print('U_star is 1')
        return 0
    else:
        h = (1-U_star) / M
        xi = st.norm.ppf(np.linspace(U_star, 1-h, M+1))
        S_end = S[-1] * np.exp((r - sigma**2/2)*dt + sigma*np.sqrt(dt)*xi)
        
        
        mean_S = np.mean(np.concatenate((matlib.repmat(S,M+1,1),S_end.reshape((M+1,1))), axis=1), 1)
        
        
        f = np.zeros(M+1)
        if type == 1:
            f = (np.abs(mean_S - K) + (mean_S - K))/2
        elif type == 2:
            f = (mean_S - K) > 0
        
        val = f[0] + 4*f[-2] + f[-1]
        for k in np.arange(1,M/2-1):
            val = val + 2*f[int(2*k)] + 4*f[int(2*k-1)]
        return h/3 * val


def p(type, xi, t, r, sigma, S0, K, M):
    d = xi.shape[0]
    dt = np.diff(t)
    W = np.append([0], np.cumsum(np.sqrt(dt)*xi))
    S = S0*np.exp((r - sigma**2/2)*t + sigma*W)  # stock price form 0 to T-dt (without last value)
    
    # integration bound U_star for the integrale
    Stm_star = (d+1)*K - np.sum(S)
    if Stm_star > 0:
        value = np.log(Stm_star/S0)/(dt[0]*sigma*t[-1]) - (r-sigma**2/2)/(dt[0]*sigma) - W[-1]/dt[0]
        U_star = st.norm.cdf(value)
    else:
        U_star = 0
    
    if type == 1:
        val = Simpson(type, M, U_star, S, dt[0], r, sigma, K)
    elif type == 2:
        val = Simpson(type, M, U_star, S, dt[0], r, sigma, K)
    return val

def pre_int_evaluate(type, x, r=0.1, sigma=0.1, T=1, S0=100, K=100):
    d = x.shape[0]
    M = x.shape[1]
    val = np.zeros(M)
    t = np.linspace(0, T, d+1)
    for j in range(M):
        xi = st.norm.ppf(x[:,j])
        val[j] = p(type, xi, t, r, sigma, S0, K, M)
    return val

def pre_int_CMC(type, d, M):
    x = np.random.random((d-1, M)) # we choosej = m, the last point
    data = pre_int_evaluate(type, x)
    est = np.mean(data)
    err_est = np.std(data)/np.sqrt(M)
    #err = np.abs(est - exact)
    return est, err_est

def pre_int_QMC(type, d, N, K):
    x = sn.generate_points(N, d, 0)
    x = x.T
    data = np.zeros(K)
    for i in range(K):
        dat = pre_int_evaluate(type, np.mod(x + matlib.repmat(np.random.random(size=(int(d),1)), 1, int(N)),1))
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

Mlist = 2**np.arange(5,10)
Nlist = 2**np.arange(7,14)
nM = np.size(Mlist)
nN = np.size(Nlist)

cmc_est, cmc_err_est = np.zeros(nM), np.zeros(nM)
qmc_est, qmc_err_est = np.zeros(nM), np.zeros(nM)

cmc_est_pre, cmc_err_est_pre = np.zeros(nM), np.zeros(nM)
qmc_est_pre, qmc_err_est_pre = np.zeros(nM), np.zeros(nM)

fig, axes = plt.subplots(nrows = 5, ncols = 1, figsize = (8,30))
types = 2 # change to 1 or 2

K = 20
for j in range(nM):
    for i in range(nM):
        cmc_est[i], cmc_err_est[i] = CMC(types, Mlist[j], Nlist[i])
        qmc_est[i], qmc_err_est[i] = QMC(types, Mlist[j], Nlist[i]/K, K)
        cmc_est_pre[i], cmc_err_est_pre[i] = pre_int_CMC(types, Mlist[j], Nlist[i])
        qmc_est_pre[i], qmc_err_est_pre[i] = pre_int_QMC(types, Mlist[j], Nlist[i]/K, K)
    
    # save results:
    cmc = np.append('cmc_err_est_pre',cmc_err_est_pre)
    qmc = np.append('qmc_err_est_pre',qmc_err_est_pre)
    fileName = 'results/ex2_error'+str(types)+'_' + str(Mlist[j]) + '.csv'
    np.savetxt(fileName, [p for p in zip(cmc, qmc)], delimiter=';', fmt='%s')

    # plot:
    ax = axes[j]
    ax.loglog(Mlist, cmc_err_est, '-', label = 'CMC error estimate')
    ax.loglog(Mlist, qmc_err_est, '-', label = 'QMC error estimate')
    ax.loglog(Mlist, cmc_err_est_pre, '-', label = 'CMC with pre-int error estimate')
    ax.loglog(Mlist, qmc_err_est_pre, '-', label = 'QMC with pre-int error estimate')
    ax.loglog(Mlist, Mlist**-0.5, '--', label = r'$M^{-1/2}$',color='gray')
    
    if types == 2:
        ax.loglog(Mlist, Mlist**-1.0, ':',  label = r'$M^{-1}$',color='gray')
    
    ax.set_title(r'$\Psi_'+str(types)+'$, $m='+str(Mlist[j])+'$')
    ax.grid(True,which='both') 
    ax.legend()

plt.savefig('./figures/ex2_error_Psi_'+str(types)+'.pdf', format='pdf', bbox_inches='tight')
plt.show()