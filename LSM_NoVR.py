#%%
import numpy as np
import warnings
warnings.simplefilter('ignore')
from numpy.polynomial.laguerre import lagfit, lagval
from numba import jit
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
#%%
# Random number generation
np.random.seed(1996)
z = np.random.normal(size=(par.M + 1, par.n)) # Drawing all random numbers at once
S = np.nan + np.zeros((par.M + 1, par.n)) # Empty array for GBM
#%%
@jit
def gbm_LSM(par):
    ''' Valuation of American option by Least Squares Monte Carlo.
    Parameters
    ==========
    S0 : spot price
    K : strike float
    r : riskless interest rate
    I : int, number of paths to be simulated
    T : time to maturity, in years
    M : int, number of time steps for discretization
    sigma : vol
    Returns
    =======
    Est : float
    estimated present value of American put option
    '''
    dt = par.T / par.M # step length
    df = np.exp(-par.r * dt) # discount function
    
    # GBM Stock Price Paths
    # exponentiating Brownian Motion
    S = par.S0 * np.exp(np.cumsum((par.r - 0.5 * par.sigma ** 2) * dt 
    + par.sigma * np.sqrt(dt) * z, axis=0))
    S[0] = par.S0

    # put option pay-off
    h = np.maximum(par.K - S, 0)
    # LS algorithm
    V = np.copy(h)
    deg = 17 # Polynomial degree (Basis functions for regression) 
    for t in range(par.M - 1, 0, -1): # Backwards induction
        reg = lagfit(S[t], V[t + 1] * df, deg) # OLS w. Laguerre pol. as basis funct.
        C = lagval(S[t], reg) # Continuation values
        V[t] = np.where(C > h[t], V[t + 1] * df, h[t]) # ITM paths
        
    # MC estimator
    y_i = df * V[1]
    Est = np.mean(y_i)
    Var = np.var(y_i)
    SE = np.std(y_i, ddof=1) / np.sqrt(par.n) # ddof = 1 for 1/(I-1), sample std. dev. 
    
    return Est, Var, SE
#%%
# Standard errors for Table 2

# Estimate loop
for par.T in range(1,2+1,1):
    for par.K in range(90,110+1,5):
        for par.sigma in (.2, .3):
            print("T:",par.T,"K:",par.K,"sigma:",par.sigma,"(Price,SE):"
            ,round(gbm_LSM(par)[0],3)
            ,round(gbm_LSM(par)[2],3))

# %%
