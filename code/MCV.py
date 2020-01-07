import numpy as np 
from numpy import matlib
import scipy.stats as st 
import matplotlib.pyplot as plt 
import sobol_new as sn

######
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
######

def genPsi(type, xi, t, r, sigma, S0, K):
	dt = np.diff(t)
	W = np.cumsum(np.sqrt(dt)*xi)
	S = S0*np.exp((r - sigma**2/2)*t[1:] + sigma*W)
	if type == 1:
		val = (np.abs(np.mean(S) - K) + (np.mean(S) - K))/2
	elif type == 2:
		val = (np.mean(S) - K) > 0
	return val, S

def evaluate(type, x, r=0.1, sigma=0.1, T=1, S0=100, K=100):
    d = x.shape[0]
    M = x.shape[1]
    val = np.zeros(M)
    S = np.zeros((M,d))
    t = np.linspace(0, T, d+1)
    if type == 1:
        for j in range(M):
            xi = st.norm.ppf(x[:,j])
            val[j], S[j,:] = genPsi(type, xi, t, r, sigma, S0, K)
    elif type == 2:
        for j in range(M):
            xi = st.norm.ppf(x[:,j])
            val[j], S[j,:] = genPsi(type, xi, t, r, sigma, S0, K)
    return val, S

def CMC(type, d, M):
	x = np.random.random((d, M))
	data, S = evaluate(type, x)
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
    x = np.random.random((d, N))
    data1, S2 = evaluate(type, x)
    Z_tilde = data1 + a_opt*(S2-mean)
    est = np.mean(Z_tilde)
    err_est = np.sqrt(np.var(Z_tilde)/N)
    return est, err_est

def MCV(type, d, N):
    x = np.random.random((d,N))
    data, S = evaluate(type,x)
    t = np.linspace(0,T,d+1)
    t = t[1:]
    mu_y = S0*np.exp(r*t)
    mu_z = np.mean(data)
    s2_zy = np.zeros(d)
    s2_yy = np.zeros((d,d))
    for j in range(d):
        s2_zy[j] = 1/(N-1)*np.sum((data-mu_z)*(S[:,j] - mu_y[j]))
    for j in range(d):
        for k in range(d):
            s2_yy[j,k] = np.mean((S[:,j] - mu_y[j])*(S[:,k] - mu_y[k]))
    a_opt = -np.linalg.inv(s2_yy)@s2_zy
    Z_tilde = np.zeros(N)
    for i in range(N):
        Z_tilde[i] = data[i] + a_opt.dot(S[i,:] - mu_y) 
    est = np.mean(Z_tilde)
    err_est = np.sqrt(np.var(Z_tilde)/N)
    return est, err_est

def AV(type, d, N):
    N2 = int(N/2)
    x1 = np.random.random((d, N2))
    x2 = 1 - x1
    data1, _ = evaluate(type, x1)
    data2, _ = evaluate(type, x2)
    est = np.mean(0.5*(data1 + data2))
    err_est = np.std(0.5*(data1 + data2))/np.sqrt(N2)
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

cmc_mean = np.zeros(nM)
cv_mean = np.zeros(nM)
av_mean = np.zeros(nM)

cmc_err_est = np.zeros(nN)
cv_err_est = np.zeros(nN)
av_err_est = np.zeros(nN)

cmc_est = np.zeros(nN)
cv_est = np.zeros(nN)
av_est = np.zeros(nN)

fig, axes = plt.subplots(nrows = 5, ncols = 1, figsize = (8,30))
types = 1 # change to 1 or 2

K = 20
for j in range(nM):
    for i in range(nN):
        cmc_est[i], cmc_err_est[i] = CMC(types, Mlist[j], Nlist[i])
        cv_est[i], cv_err_est[i] = MCV(types, Mlist[j], Nlist[i])
        av_est[i], av_err_est[i] = AV(types, Mlist[j], Nlist[i])
    
    cmc_mean[j] = np.mean(cmc_est)
    cv_mean[j] = np.mean(cv_est)
    av_mean[j] = np.mean(av_est)
    # save results:
    cmc = np.append('cmc_err_est',cmc_err_est)
    cv = np.append('cv_err_est',cv_err_est)
    av = np.append('av_err_est',av_err_est)
    fileName = 'results/ex3/error_Psi'+str(types)+'_' + str(Mlist[j]) + '.csv'
    #np.savetxt(fileName, [p for p in zip(cmc, cv, av)], delimiter=';', fmt='%s')
    
    # plot:
    ax = axes[j]
    ax.loglog(Nlist, cmc_err_est, '-',  label = 'CMC error estimate')
    ax.loglog(Nlist, cv_err_est, '-',  label = 'CV error estimate')
    ax.loglog(Nlist, av_err_est, '-',  label = 'AV error estimate')
    ax.loglog(Nlist, Nlist**-0.5, '--', label = r'$M^{-1/2}$',color='gray')
    
    if types == 2:
        ax.loglog(Nlist, Nlist**-1.0, ':',  label = r'$M^{-1}$',color='gray')
    
    ax.set_title(r'$\Psi_'+str(types)+'$, $m='+str(Mlist[j])+'$')
    ax.grid(True,which='both') 
    ax.legend()

#plt.savefig('./figures/ex3_error_Psi_'+str(types)+'.pdf', format='pdf', bbox_inches='tight')
plt.show()

# price for m:
r, T = 0.1, 1
price_cmc = np.exp(-r*T) * cmc_mean
price_cv = np.exp(-r*T) * cv_mean
price_av = np.exp(-r*T) * av_mean

print('Price for m = 32, 64, 128, 256, 512')
print(price_cmc)
print(price_cv)
print(price_av)