# Austin Griffith
# Python 3.6.5
# Problem 2 Analytical Solutions
# 4/18/2018

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

#%%
# mean value for lambda
def driftVal(rf,sig):
    m = rf - np.square(sig)*0.5
    return(m)

# lambda constant
def lambdaVal(rf,sig):
    l = 1 + driftVal(rf,sig)/np.square(sig)
    return(l)

# gammas are set values used as the analytical solution to the barrier put options
def I1(alpha,beta,rf,T,k,s,sig,Z):
    L = lambdaVal(rf,sig)
    xval = np.log(s/k)/(sig*np.sqrt(T)) + L*sig*np.sqrt(T)
    price = alpha*s*norm.cdf(alpha*xval) - alpha*k*np.exp(-rf*T)*norm.cdf(alpha*xval - alpha*sig*np.sqrt(T))
    return(price)

def I2(alpha,beta,rf,T,k,s,sig,Z):
    L = lambdaVal(rf,sig)
    xval = np.log(s/Z)/(sig*np.sqrt(T)) + L*sig*np.sqrt(T)
    price = alpha*s*norm.cdf(alpha*xval) - alpha*k*np.exp(-rf*T)*norm.cdf(alpha*xval - alpha*sig*np.sqrt(T))
    return(price)

def I3(alpha,beta,rf,T,k,s,sig,Z):
    L = lambdaVal(rf,sig)
    xval = np.log(np.square(Z)/(s*k))/(sig*np.sqrt(T)) + L*sig*np.sqrt(T)
    price = (alpha*s*np.power(Z/s,2*L)*norm.cdf(beta*xval) -
        alpha*k*np.exp(-rf*T)*np.power(Z/s,2*L-2)*norm.cdf(beta*xval-beta*sig*np.sqrt(T)))
    return(price)

def I4(alpha,beta,rf,T,k,s,sig,Z):
    L = lambdaVal(rf,sig)
    xval = np.log(Z/s)/(sig*np.sqrt(T)) + L*sig*np.sqrt(T)
    price = (alpha*s*np.power(Z/s,2*L)*norm.cdf(beta*xval) -
        alpha*k*np.exp(-rf*T)*np.power(Z/s,2*L - 2)*norm.cdf(beta*xval - beta*sig*np.sqrt(T)))
    return(price)

# plot values for a given dataframe
def plotVals(data,title,xlab,ylab,leg=True):
    plt.figure()
    for i in data.columns:
        plt.plot(data[i],label=i)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if leg == True:
        plt.legend()
    plt.grid()
    plt.show()

#%%
# parameters
r = 0.01
E = 1
T = 1
step = 0.01
N = 1024
sigma = 0.2

# barrier values
Zhigh = 1.2
Zlow = 0.6

# stock values
sMax = step*N
sVals = np.arange(step,sMax+step,step)

#%%
# down and out put calculation
doPut = []
for i in sVals:
    if i < Zlow:
        doPut.append(0)
    else:
        dop = (I1(-1,1,r,T,E,i,sigma,Zlow) - I2(-1,1,r,T,E,i,sigma,Zlow) +
            I3(-1,1,r,T,E,i,sigma,Zlow) - I4(-1,1,r,T,E,i,sigma,Zlow))
        doPut.append(dop)

#%%
# up and out put option, Z > E
uoPutHigh = []
for i in sVals:
    if i > Zhigh:
        uoPutHigh.append(0)
    else:
        uolp = I1(-1,-1,r,T,E,i,sigma,Zhigh) - I3(-1,-1,r,T,E,i,sigma,Zhigh)
        uoPutHigh.append(uolp)

#%%
# up and out put option, Z < E
uoPutLow = []
for i in sVals:
    if i > Zlow:
        uoPutLow.append(0)
    else:
        uohp = I2(-1,-1,r,T,E,i,sigma,Zlow) - I4(-1,-1,r,T,E,i,sigma,Zlow)
        uoPutLow.append(uohp)

#%%
# plot analytical solutions
plotRange = 200

optPlot = pd.DataFrame([doPut,uoPutLow,uoPutHigh,sVals.tolist()]).transpose()
optPlot.columns = ['Down-Out Put','Up-Out Low Z Put','Up-Out High Z Put','Stocks']
optPlot = optPlot.set_index('Stocks')

xlab = 'S'
ylab = 'V(S)'
titles = ['Down-Out Put','Up-Out Low Z Put','Up-Out High Z Put']
for t in titles:
    temp = pd.DataFrame(optPlot[t][sVals[:plotRange]])
    plotVals(temp,t+' Analytical',xlab,ylab,leg=False)


