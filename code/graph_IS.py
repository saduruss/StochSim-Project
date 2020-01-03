import numpy as np 
from numpy import matlib
import scipy.stats as st 
import matplotlib.pyplot as plt 
import sobol_new as sn

x = np.linspace(80,150,1000)

m = 32
r = 0.1
sigma = 0.1
T = 1
S0 = 100
K = 100

psi1 = lambda x: (x-K)*(x-K>0)
psi2 = lambda x: (x-K>0)

xs = x/S0
pdf = st.lognorm.pdf(xs,np.sqrt(T)*sigma,scale=np.exp((r-0.5*sigma**2)*T))
ert = S0*np.exp(r*T)
print(ert)

plt.figure()
plt.plot(x,psi1(x),label=r'$\Psi_1(S_t)$')
plt.plot(x,psi2(x),label=r'$\Psi_2(S_t)$')
plt.plot(x,pdf,label=r'$pdf(S_t)$')
plt.axvline(ert,label=r'$E(S_t)$')
plt.axhline(0)
plt.legend()
plt.show()

plt.figure()
plt.plot(xs,pdf,label=r'$pdf(S_t)$')
plt.axvline(ert/S0,label=r'$E(S_t)$')
plt.axhline(0)
plt.legend()
plt.show()