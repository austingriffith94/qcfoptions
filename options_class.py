# Austin Griffith
# Derivatives HW 6
# Python 3.6.4
# 3/6/2018

import pandas as pd
import numpy as np
from scipy.stats import norm

#%%
# option class for all options
class Options:
    def __init__(self,s_,s2_,k_,r_,T_,sig_,sig2_,q_,q2_):
        self.s = s_
        self.s2 = s2_
        self.k = k_
        self.r = r_
        self.T = T_
        self.sigma = sig_
        self.sigma2 = sig2_
        self.q = q_
        self.q2 = q2_

    def euroPut(self):
        d1_top = (np.log(self.s/self.k) + (self.r - self.q + 0.5*np.square(self.sigma))*T)
        d1 = d1_top/(self.sigma*np.sqrt(self.T))
        d2 = d1 - self.sigma*np.sqrt(self.T)

        option = np.exp(-1*self.q*self.T)*self.s*norm.cdf(-d1)
        strike = np.exp(-1*self.r*self.T)*self.k*norm.cdf(-d2)

        price = strike - option
        return(price)

    def euroCall(self):
        d1_top = (np.log(self.s/self.k) + (self.r - self.q + 0.5*np.square(self.sigma))*T)
        d1 = d1_top/(self.sigma*np.sqrt(self.T))
        d2 = d1 - self.sigma*np.sqrt(self.T)

        option = np.exp(-1*self.q*self.T)*self.s*norm.cdf(d1)
        strike = np.exp(-1*self.r*self.T)*self.k*norm.cdf(d2)

        price = option - strike
        return(price)

    def powerCall(self):
        d1_top = (np.log(self.s**2) - np.log(self.k) +
              (2*self.r - 2*self.q + 3*np.square(self.sigma))*self.T)
        d1 = d1_top/(2*self.sigma*np.sqrt(self.T))
        d2 = d1 - 2*self.sigma*np.sqrt(self.T)

        option = (np.exp((self.r - 2*self.q + self.sigma**2)*self.T)*np.square(self.s)*
                  norm.cdf(d1))
        strike = self.k*np.exp(-1*self.r*self.T)*norm.cdf(d2)

        price = option - strike
        return(price)

    def powerPut(self):
        d1_top = (np.log(self.s**2) - np.log(self.k) +
              (2*self.r - 2*self.q + 3*np.square(self.sigma))*self.T)
        d1 = d1_top/(2*self.sigma*np.sqrt(self.T))
        d2 = d1 - 2*self.sigma*np.sqrt(self.T)

        option = (np.exp((self.r - 2*self.q + self.sigma**2)*self.T)*np.square(self.s)*
                  norm.cdf(-d1))
        strike = self.k*np.exp(-1*self.r*self.T)*norm.cdf(-d2)

        price = strike - option
        return(price)

    def margrabeOption(self,correlation=0):
        sigmaMix = np.sqrt(self.sigma**2 + self.sigma2**2 - self.sigma*self.sigma2*correlation)
        d1_top = (np.log(self.s/self.s2) +
                  (self.q2 - self.q + 0.5*(sigmaMix**2))*self.T)
        d1 = d1_top/(sigmaMix*np.sqrt(self.T))
        d2 = d1 - sigmaMix*np.sqrt(self.T)

        option = np.exp(-1*self.q*self.T)*self.s*norm.cdf(d1)
        option2 = np.exp(-1*self.q2*self.T)*self.s2*norm.cdf(d2)

        price = option - option2
        return(price)

#%%
# variables
S1 = 99
S2 = 99
K = 100
T = 3

sig1 = 0.25
sig2 = 0.3

div1 = 0.04
div2 = 0.03

rf = 0.02

#%%
opt = Options(S1,S2,K,rf,T,sig1,sig2,div1,div2)

print('\nEuro Put Price')
print(opt.euroPut())

print('\nEuro Call Price')
print(opt.euroCall())

print('\nPower Call Price')
print(opt.powerCall())

print('\nPower Put Price')
print(opt.powerPut())

print('\nMargrabe Option Price')
print(opt.margrabeOption())
