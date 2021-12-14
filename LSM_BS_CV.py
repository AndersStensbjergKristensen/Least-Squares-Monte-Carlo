#%%
import numpy as np
import matplotlib.pyplot as plt
import time #Used to measure time to run code
import warnings
warnings.simplefilter('ignore')
from numpy.polynomial.laguerre import lagfit, lagval
%matplotlib inline
from numba import jit
from scipy.stats import norm
import scipy.stats as si
from sympy.stats import Normal, cdf
# %%
# parameters
class par: pass
par.S0 = 100
par.K = 100
par.r = 0.04
par.sigma = 0.2
par.T = 1.0
par.n = 100000
par.M = 252
# %%
# Random number generation
np.random.seed(1996)
z = np.random.normal(size=(par.M + 1, par.n)) # Drawing all random numbers at once
S = np.nan + np.zeros((par.M+1,par.n)) # Empty array for GBM
#%%
@jit
def gbm_LSM_CV(par):
    ''' Valuation of American option in Black-Scholes-Merton
    by Monte Carlo simulation by LS algorithm. Built with a dynamic BSM control variate.
    Parameters
    ==========
    S0 : spot price
    K : strike price
    r : riskless interest rate
    n : int, number of paths to be simulated
    T : time to maturity, in years
    M : int, number of time intervals for discretization
    sigma : vol
    convar : enable control variate
    Returns
    =======
    Est : float
    estimated present value of American call option
    '''
    dt = par.T / par.M # step size
    df = np.exp(-par.r * dt) # discount function

    # Generation of underlying asset process
    # Stock Price Paths
    S = par.S0 * np.exp(np.cumsum((par.r - 0.5 * par.sigma ** 2) * dt
    + par.sigma * np.sqrt(dt) * z, axis=0)) # by exponentiating the Brownian motion
    S[0] = par.S0 # Initiliazing underlying path
    
    # put option pay-off
    h = np.maximum(par.K - S, 0)
    V = np.copy(h)
    deg = 17
    # LS algorithm
    for t in range(par.M - 1, 0, -1): # Backwards induction
        reg = lagfit(S[t], V[t + 1] * df, deg)
        C = lagval(S[t], reg)
        V[t] = np.where(C > h[t], V[t + 1] * df, h[t])
    
    # Regular estimator
    y_i = df * V[1]
    Var = np.var(y_i)

    # BS Control variate 
    x_i = df * np.maximum(par.K - S[-1], 0) # S[-1] Last elements of Stock price paths
    # Control mean
    d1 = (1/(par.sigma*np.sqrt(par.T)))*(np.log(par.S0/par.K)+(par.r+0.5*par.sigma**2)*par.T)
    d2 = d1-par.sigma*np.sqrt(par.T)
    Ex_i = (par.K * np.exp(-par.r * par.T) * si.norm.cdf(-d2, 0.0, 1.0) - par.S0 * si.norm.cdf(-d1, 0.0, 1.0))

    # estimate of correlation coefficient
    gamma = (np.sum((x_i - Ex_i) * (y_i - np.mean(y_i)))
    / np.sum((x_i - Ex_i) ** 2))
    
    # Est w. control variate
    y_cv = (y_i - gamma*(x_i-Ex_i)) # correction
    Est = np.mean(y_cv)
    SE = np.std(y_cv, ddof=1)/np.sqrt(par.n) # ddof = 1 for sample std. dev.
    Var_cv = np.var(y_cv) 
    VRF = 1/(Var_cv/Var) # Variance reduction function
    
    return Est, VRF, SE, gamma
#%%
# Table 2 BS-CV estimates

# Estimate loop
for par.T in range(1,2+1,1):
    for par.K in range(90,110+1,5):
        for par.sigma in (.2, .3):
            print("T:",par.T,"K:",par.K,"sigma:",par.sigma,"(Price,SE):"
            ,round(gbm_LSM_CV(par)[0],3)
            ,round(gbm_LSM_CV(par)[2],3))
#%%
# Initiate empty lists for regression coefficients
gamma1 = []
gamma2 = []
gamma3 = []
gamma4 = []

K1 = np.linspace(90,110,5) # 5 evenly spaced strikes
#%%
# Control variate regression coefficient loop and plot
stepsize = 5
par.sigma = .2
par.T = 1
for par.K in range(90,110+1,stepsize):
    gamma1.append(gbm_LSM_CV(par)[3])
par.T = 2
for par.K in range(90,110+1,stepsize):
    gamma2.append(gbm_LSM_CV(par)[3])
par.sigma = .3
par.T = 1
for par.K in range(90,110+1,stepsize):
    gamma3.append(gbm_LSM_CV(par)[3])
par.T = 2
for par.K in range(90,110+1,stepsize):
    gamma4.append(gbm_LSM_CV(par)[3])

#%%
plt.plot(K1, gamma1,label='$\sigma$=.2 T=1')
plt.plot(K1, gamma2,label='$\sigma$=.2 T=2')
plt.plot(K1, gamma3,label='$\sigma$=.3 T=1')
plt.plot(K1, gamma4,label='$\sigma$=.3 T=2')
plt.legend()
plt.xlabel('K')
plt.ylabel('$\hat{\gamma}$')
plt.show()

#%%
# Initiate empty lists for variance reduction functions
vrf1 = []
vrf2 = []
vrf3 = []
vrf4 = []

K1 = np.linspace(90,110,5) # 5 evenly spaced strikes
#%%
# Variance reduction function loop and plot
stepsize = 5
par.sigma = .2
par.T = 1
for par.K in range(90,110+1,stepsize):
    vrf1.append(gbm_LSM_CV(par)[1])
par.T = 2
for par.K in range(90,110+1,stepsize):
    vrf2.append(gbm_LSM_CV(par)[1])
par.sigma = .3
par.T = 1
for par.K in range(90,110+1,stepsize):
    vrf3.append(gbm_LSM_CV(par)[1])
par.T = 2
for par.K in range(90,110+1,stepsize):
    vrf4.append(gbm_LSM_CV(par)[1])

#%%
plt.plot(K1, vrf1,label='$\sigma$=.2 T=1')
plt.plot(K1, vrf2,label='$\sigma$=.2 T=2')
plt.plot(K1, vrf3,label='$\sigma$=.3 T=1')
plt.plot(K1, vrf4,label='$\sigma$=.3 T=2')
plt.legend()
plt.xlabel('K')
plt.ylabel('VRF')
plt.show()
