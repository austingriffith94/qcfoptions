# Austin Griffith
# Simulation

import numpy as np
import scipy.stats as sctats
import time

# simulation payoffs
def _EuroSim(S,k,r,T):
    callMotion = (S[-1] - k).clip(0)
    putMotion = (k - S[-1]).clip(0)
    
    call = np.exp(-r*T)*np.average(callMotion)
    put = np.exp(-r*T)*np.average(putMotion)
    return([[call,put],[callMotion,putMotion]])
    
def _AsianGeoFixSim(S,k,r,T):
    avg = sctats.gmean(S,axis=0)
    callMotion = (avg - k).clip(0)
    putMotion = (k - avg).clip(0)
    
    call = np.exp(-r*T)*np.average(callMotion)
    put = np.exp(-r*T)*np.average(putMotion)
    return([[call,put],[callMotion,putMotion]])

def _AsianGeoFloatSim(S,m,r,T):
    avg = sctats.gmean(S,axis=0)
    callMotion = (S[-1] - m*avg).clip(0)
    putMotion = (m*avg - S[-1]).clip(0)
    
    call = np.exp(-r*T)*np.average(callMotion)
    put = np.exp(-r*T)*np.average(putMotion)
    return([[call,put],[callMotion,putMotion]])
    
def _AsianArithFixSim(S,k,r,T):
    avg = np.average(S,axis=0)
    callMotion = (avg - k).clip(0)
    putMotion = (k - avg).clip(0)
    
    call = np.exp(-r*T)*np.average(callMotion)
    put = np.exp(-r*T)*np.average(putMotion)
    return([[call,put],[callMotion,putMotion]])
    
def _AsianArithFloatSim(S,m,r,T):
    avg = np.average(S,axis=0)
    callMotion = (S[-1] - m*avg).clip(0)
    putMotion = (m*avg - S[-1]).clip(0)
    
    # class
    call = np.exp(-r*T)*np.average(callMotion)
    put = np.exp(-r*T)*np.average(putMotion)
    return([[call,put],[callMotion,putMotion]])
    
def _PowerSim(S,k,r,T,n):
    power = np.power(S[-1],n)
    callMotion = (power - k).clip(0)
    putMotion = (k - power).clip(0)
    
    call = np.exp(-r*T)*np.average(callMotion)
    put = np.exp(-r*T)*np.average(putMotion)
    return([[call,put],[callMotion,putMotion]])
    
def _PowerStrikeSim(S,k,r,T,n):
    powerS = np.power(S[-1],n)
    callMotion = (powerS - k**n).clip(0)
    putMotion = (k**n - powerS).clip(0)
    
    call = np.exp(-r*T)*np.average(callMotion)
    put = np.exp(-r*T)*np.average(putMotion)
    return([[call,put],[callMotion,putMotion]])
    
def _AvgBarrier(S,Z,r,timeMatrix):
    Z = 110
    s0 = S[0][0]
    if s0 < Z: # below
        hitBarrier = np.cumprod(S < Z,axis=0)
    if s0 > Z: # above
        hitBarrier = np.cumprod(S > Z,axis=0)
    
    paymentTime = np.array(np.max(np.multiply(timeMatrix,hitBarrier),axis=0))
    payoffMotion = np.sum(np.multiply(hitBarrier,S),axis=0) / np.sum(hitBarrier,axis=0)
    price = np.average(np.exp(-r*paymentTime)*payoffMotion)
    return([price,payoffMotion])
    
def simpleSim(s0,r,T,vol,dt,paths):
    intervals = int(T/dt)
    
    S = np.random.random([intervals+1,paths])
    S = -1 + 2*(S > 0.5)
    S = S*np.sqrt(dt)*vol + (r - 0.5*vol*vol)*dt
    S[0] = np.ones(paths)*np.log(s0)
    S = np.exp(np.matrix.cumsum(S,axis=0))
    return(S)
    
def hestonSim(s0,r,T,vol,phi,kappa,xi,dt,paths):
    intervals = int(T/dt)
    
    S = np.sqrt(dt)*(-1 + 2*(np.random.random([intervals+1,paths]) > 0.5))
    V = np.sqrt(dt)*(-1 + 2*(np.random.random([intervals+1,paths]) > 0.5))
    V = phi*S + np.sqrt(1 - phi*phi)*V
    
    volMotion = np.zeros([intervals+1,paths])
    volMotion[0] = vol*vol*np.ones(paths)
    
    for t in range(intervals):
        vt = volMotion[t]
        dvt = kappa*(vol*vol - vt)*dt + xi*np.sqrt(vt)*V[t]
        volMotion[t+1] = vt + dvt
    
    S = (r - 0.5*volMotion)*dt + np.sqrt(volMotion)*S
    S[0] = np.ones(paths)*np.log(s0)
    S = np.exp(np.matrix.cumsum(S,axis=0))
    return([S, volMotion])


class simple:
    def __init__(self,s0,r,T,vol,dt=0.001,paths=10000):
        self.s0 = s0
        self.r = r
        self.T = T
        self.vol = vol
        self.dt = dt
        self.paths = paths
        
        timeInt = np.matrix(np.arange(0,T+dt,dt)).transpose()
        self.timeMatrix = np.matmul(timeInt,np.matrix(np.ones(paths)))
        self.discount = np.exp(-r*self.timeMatrix)

        start = time.time()        
        self.S = simpleSim(s0,r,T,vol,dt,paths)
        self.simtime = time.time() - start
        
class heston:
    def __init__(self,s0,r,T,vol,phi,kappa,xi,dt=0.001,paths=10000):
        self.s0 = s0
        self.r = r
        self.T = T
        self.vol = vol
        self.dt = dt
        self.paths = paths
        self.phi = phi # correlation of price to volatility
        self.kappa = kappa # speed of adjustment
        self.xi = xi # volatility of volatility
        
        intervals = int(T/dt)
        
        timeInt = np.matrix(np.arange(0,T+dt,dt)).transpose()
        self.timeMatrix = np.matmul(timeInt,np.matrix(np.ones(paths)))
        self.discount = np.exp(-r*self.timeMatrix)
        
        start = time.time()
        [self.S, self.volMotion] = hestonSim(s0,r,vol,phi,kappa,xi,dt,intervals,paths)
        self.simtime = time.time() - start
    
    
    