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
	return val, np.mean(S)

def evaluate(type, x, r=0.1, sigma=0.1, T=1, S0=100, K=100):
    d = x.shape[0]
    M = x.shape[1]
    val = np.zeros(M)
    S = np.zeros(M)
    t = np.linspace(0, T, d+1)
    if type == 1:
        for j in range(M):
            xi = st.norm.ppf(x[:,j])
            val[j], S[j] = genPsi(type, xi, t, r, sigma, S0, K)
    elif type == 2:
        for j in range(M):
            xi = st.norm.ppf(x[:,j])
            val[j], S[j] = genPsi(type, xi, t, r, sigma, S0, K)
    return val, S

def CMC(type, d, M):
    x = np.random.random((d, M))
    data, _ = evaluate(type, x)
    est = np.mean(data)
    err_est = np.std(data)/np.sqrt(M)
    #err = np.abs(est - exact)
    return est, err_est

def QMC(type, d, N, K):
    x = sn.generate_points(N, d, 0)
    x = x.T
    data = np.zeros(K)
    for i in range(K):
        dat, _ = evaluate(type, np.mod(x + matlib.repmat(np.random.random(size=(int(d),1)), 1, int(N)),1))
        data[i] = np.mean(dat)
    est = np.mean(data)
    err_est = 3*np.std(data)/np.sqrt(K)
    #err = np.abs(est - exact)
    return est, err_est

def CV(type, d, N, N_bar):
    # pilot run
    x1 = np.random.random((d, N_bar))
    data1, S2 = evaluate(type, x1)
    t = np.linspace(0,T,d+1)
    t = t[1:]
    mean = S0/d*np.sum(np.exp(r*t))
    #var = (S0/d)**2*np.sum((np.exp(t*sigma**2)-1)*np.exp(2*r*t))
    #mu_z = np.mean(data1)
    #sigma2_zy = 1/(N_bar-1)*np.sum((data1-mu_z)*(S2-mean))
    #a_opt = -sigma2_zy/var
    C = np.cov(data1,S2)
    a_opt = -C[0,1]/C[1,1]
    # Monte Carlo
    x1 = np.random.random((d, N))
    data1, S2 = evaluate(type, x1)
    Z_tilde = data1 + a_opt*(S2-mean)
    est = np.mean(Z_tilde)
    err_est = np.sqrt(np.var(Z_tilde)/N)
    return est, err_est

def Simpson(type, M, U_star, S, dt, r, sigma, K):
    if U_star == 1:
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
    S = S0*np.exp((r - sigma**2/2)*t + sigma*W)  # stock price 0,...,T-dt (without last value)
    
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
        M = 2**7
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
Nlist = 2**np.arange(7,13)

nM = np.size(Mlist)
nN = np.size(Nlist)

cmc_mean, qmc_mean          = np.zeros(nM), np.zeros(nM)
pre_cmc_mean, pre_qmc_mean  = np.zeros(nM), np.zeros(nM)
cv_mean = np.zeros(nM)

times_cmc, times_qmc        = np.zeros(nN), np.zeros(nN)
times_pre_cmc,times_pre_qmc = np.zeros(nN), np.zeros(nN)
times_cv = np.zeros(nN)

cmc_est, cmc_err_est = np.zeros(nN), np.zeros(nN)
qmc_est, qmc_err_est = np.zeros(nN), np.zeros(nN)
cmc_est_pre, cmc_err_est_pre = np.zeros(nN), np.zeros(nN)
qmc_est_pre, qmc_err_est_pre = np.zeros(nN), np.zeros(nN)
cv_est, cv_err_est = np.zeros(nN), np.zeros(nN)

fig, axes = plt.subplots(nrows = 5, ncols = 2, figsize = (16,30))
axes = axes.flatten()
types = 1 # change to 1 or 2

K = 20
for j in range(nM):
    print('Computation for m = '+str(Mlist[j])+' ...')
    for i in range(nN):
        start = time.time()
        cmc_est[i], cmc_err_est[i] = CMC(types, Mlist[j], Nlist[i])
        time1 = time.time()
        times_cmc[i] = time1-start
        
        qmc_est[i], qmc_err_est[i] = QMC(types, Mlist[j], Nlist[i]/K, K)
        time2 = time.time()
        times_qmc[i] = time2-time1
        
        cmc_est_pre[i], cmc_err_est_pre[i] = pre_int_CMC(types, Mlist[j], Nlist[i])
        time3 = time.time()
        times_pre_cmc[i] = time3-time2
        
        qmc_est_pre[i], qmc_err_est_pre[i] = pre_int_QMC(types, Mlist[j], Nlist[i]/K, K)
        time4 = time.time()
        times_pre_qmc[i] = time4-time3
        
        cv_est[i], cv_err_est[i] = CV(types, Mlist[j], Nlist[i], int(Nlist[i]/2**4))
        time5 = time.time()
        times_cv[i] = time5 - time4

    
    # save price results
    cmc_mean[j], qmc_mean[j] = np.mean(cmc_est), np.mean(qmc_est)
    pre_cmc_mean[j], pre_qmc_mean[j] = np.mean(cmc_est_pre), np.mean(qmc_est_pre)
    cv_mean[j] = np.mean(cv_est)
    
    # save error results:
    cmc = np.append('cmc_err_est_pre',cmc_err_est_pre)
    qmc = np.append('qmc_err_est_pre',qmc_err_est_pre)
    cv = np.append('cv_err_est', cv_err_est)
    fileName = 'results/final/error_Psi'+str(types)+'_' + str(Mlist[j]) + '.csv'
    np.savetxt(fileName, [p for p in zip(cmc, qmc, cv)], delimiter=';', fmt='%s')

    # plot:
    ax = axes[int(2*j)]
    ax.loglog(Nlist, cmc_err_est, '-', label = 'CMC error estimate')
    ax.loglog(Nlist, qmc_err_est, '-', label = 'QMC error estimate')
    ax.loglog(Nlist, cmc_err_est_pre, '-', label = 'CMC with pre-int error estimate')
    ax.loglog(Nlist, qmc_err_est_pre, '-', label = 'QMC with pre-int error estimate')
    ax.loglog(Nlist, cv_err_est, '-', label = 'CV error estimate')
    ax.loglog(Nlist, Nlist**-0.5, '--', label = r'$N^{-1/2}$',color='gray')
    if types == 2:
        ax.loglog(Nlist, Nlist**-1.0, ':',  label = r'$N^{-1}$',color='gray')
    ax.set_title('Error estimate for '+r'$\Psi_'+str(types)+'$, $m='+str(Mlist[j])+'$')
    ax.grid(True,which='both') 
    ax.legend()

    ax = axes[int(2*j+1)]
    ax.loglog(Nlist, times_cmc, '-', label = 'CMC')
    ax.loglog(Nlist, times_qmc, '-', label = 'QMC')
    ax.loglog(Nlist, times_pre_cmc, '-', label = 'CMC with pre-int.')
    ax.loglog(Nlist, times_pre_qmc, '-', label = 'QMC with pre-int.')
    ax.loglog(Nlist, times_cv, '-', label = 'CV')
    ax.loglog(Nlist, Nlist/2**12, '--', label = r'$O(N)$',color='gray')
    ax.set_title('Time for '+r'$\Psi_'+str(types)+'$, $m='+str(Mlist[j])+'$')
    ax.grid(True,which='both') 
    ax.legend()

plt.savefig('./figures/final_error_Psi_'+str(types)+'.pdf', format='pdf', bbox_inches='tight')
plt.show()


# price for m:
r, T = 0.1, 1
price_cmc = np.exp(-r*T) * cmc_mean
price_qmc = np.exp(-r*T) * qmc_mean
price_pre_cmc = np.exp(-r*T) * pre_cmc_mean
price_pre_qmc = np.exp(-r*T) * pre_qmc_mean
price_cv = np.exp(-r*T) * cv_mean

print('Price for m = 32, 64, 128, 256, 512')
print(price_cmc)
print(price_qmc)
print(price_pre_cmc)
print(price_pre_qmc)
print(price_cv)