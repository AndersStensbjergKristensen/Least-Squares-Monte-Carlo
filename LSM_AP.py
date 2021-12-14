# %%
import numpy as np
import warnings
warnings.simplefilter('ignore')
from numpy.polynomial.laguerre import lagfit, lagval
%matplotlib inline
from numba import jit
# %%
# parameters
class par: pass
par.S0 = 100
par.K = 100
par.r = 0.04
par.sigma = 0.2
par.T = 1.0
par.n = 50000 # 100 000 / 2
par.M = 252
# %%
# Random number generator
np.random.seed(1996)
z = np.random.normal(size=(par.M + 1, par.n)) # Drawing all random numbers at once
S = np.nan + np.zeros((par.M+1, par.n)) # Empty array for GBM
#%%
@jit
def gbm_LSM_AP(par):
    dt = par.T / par.M # step size
    df = np.exp(-par.r * dt) # discount function

    # Generation of underlying asset process
    # Stock Price Paths
    S = par.S0 * np.exp(np.cumsum((par.r - 0.5 * par.sigma ** 2) * dt
    + par.sigma * np.sqrt(dt) * z, axis=0)) # by exponentiating the Brownian motion
    S[0] = par.S0
    S1 = par.S0 * np.exp(np.cumsum((par.r - 0.5 * par.sigma ** 2) * dt
    + par.sigma * np.sqrt(dt) * - z, axis=0)) # Antithetic paths
    S1[0] = par.S0
    
    # put option pay-off
    h = np.maximum(par.K - S, 0)
    h1 = np.maximum(par.K - S1, 0) # Antithetic payoff
    # LS algorithm
    V = np.copy(h)
    V1 = np.copy(h1)
    deg = 17 
    for t in range(par.M - 1, 0, -1):
        reg = lagfit(S[t], V[t + 1] * df, deg)
        C = lagval(S[t], reg)
        V[t] = np.where(C > h[t], V[t + 1] * df, h[t])
        reg1 = lagfit(S1[t], V1[t + 1] * df, deg)
        C1 = lagval(S1[t], reg1)
        V1[t] = np.where(C1 > h1[t], V1[t + 1] * df, h1[t])
        
    # MC estimator
    y_i = df * (V[1]+V1[1])/2 # avg. pairs
    Est = np.mean(y_i)
    SE = np.std(y_i,ddof=1)/np.sqrt(par.n) # Sample std. dev. of avg. pairs
    CI_u = Est+1.645*SE # 90% CI upper bound 
    
    return Est, SE, CI_u
#%%
# AP Estimate loop for table 2
# Table 2

# Estimate loop
for par.T in range(1,2+1,1):
    for par.K in range(90,110+1,5):
        for par.sigma in (.2, .3):
            print("T:",par.T,"K:",par.K,"sigma:",par.sigma,"(Price,SE, CI upper bound):"
            ,round(gbm_LSM_AP(par)[0],3)
            ,round(gbm_LSM_AP(par)[1],3)
            ,round(gbm_LSM_AP(par)[2],3))
#%%
# Timer
par.n = 100000
start_time = tic()
np.random.normal(size=(par.M + 1, par.n))
t1 = toc()

par.n = 50000
start_time = tic()
np.random.normal(size=(par.M + 1, par.n))
t2 = toc()

print("Relative efficiency:",t1/t2)
