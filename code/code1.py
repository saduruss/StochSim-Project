import numpy as np 
from numpy import matlib
import scipy.stats as st 
import matplotlib.pyplot as plt 
import sobol_new as sn

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

cmc_err_est = np.zeros(nM)
qmc_err_est = np.zeros(nM)

cmc_est = np.zeros(nM)
qmc_est = np.zeros(nM)

cmc_err = np.zeros(nM)
qmc_err = np.zeros(nM)

K = 20
for j in range(nM):
    for i in range(nM):
        cmc_est[i], cmc_err_est[i] = CMC(1, Mlist[j], Nlist[i])
        qmc_est[i], qmc_err_est[i] = QMC(1, Mlist[j], Nlist[i]/K, K)
    
    cmc = np.append('cmc_err_est',cmc_err_est)
    qmc = np.append('qmc_err_est',qmc_err_est)
    fileName = 'results/error1_' + str(Mlist[j]) + '.csv'
    np.savetxt(fileName, [p for p in zip(cmc, qmc)], delimiter=';', fmt='%s')
    #plt.loglog(Mlist, cmc_err_est, label="CMC err est")
    #plt.loglog(Mlist, qmc_err_est, label="QMC err est")
    #plt.legend()
    #plt.show()
