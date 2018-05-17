import numpy as np
import time



class simple:
    def __init__(self,s0,r,T,vol,dt=0.001,paths=10000):
        self.s0 = s0
        self.r = r
        self.T = T
        self.vol = vol
        self.dt = dt
        self.paths = paths
        
        intervals = int(T/dt)
        
        timeInt = np.matrix(np.arange(0,T+dt,dt)).transpose()
        self.timeMatrix = np.matmul(timeInt,np.matrix(np.ones(paths)))
        self.discount = np.exp(-r*self.timeMatrix)

        start = time.time()        
        S = np.random.random([intervals+1,paths])
        S = -1 + 2*(S > 0.5)
        S = S*np.sqrt(dt)*vol + (r - 0.5*vol*vol)*dt
        S[0] = np.ones(paths)*np.log(s0)
        self.S = np.exp(np.matrix.cumsum(S,axis=0))
        self.simtime = time.time() - start
        
class heston:
    def __init__(self,s0,r,T,vol,corr,kappa,xi,dt=0.001,paths=10000):
        self.s0 = s0
        self.r = r
        self.T = T
        self.vol = vol
        self.dt = dt
        self.paths = paths
        self.corr = corr # correlation of price to volatility
        self.kappa = kappa # speed of adjustment
        self.xi = xi # volatility of volatility
        
        intervals = int(T/dt)
        
        timeInt = np.matrix(np.arange(0,T+dt,dt)).transpose()
        self.timeMatrix = np.matmul(timeInt,np.matrix(np.ones(paths)))
        self.discount = np.exp(-r*self.timeMatrix)
        
        start = time.time()
        S = np.sqrt(dt)*(-1 + 2*(np.random.random([intervals+1,paths]) > 0.5))
        V = np.sqrt(dt)*(-1 + 2*(np.random.random([intervals+1,paths]) > 0.5))
        V = corr*S + np.sqrt(1 - corr*corr)*V
        
        
        volMotion = np.zeros([intervals+1,paths])
        volMotion[0] = vol*vol*np.ones(paths)
        
        for t in range(intervals):
            vt = volMotion[t]
            dvt = kappa*(vol*vol - vt)*dt + xi*np.sqrt(vt)*V[t]
            volMotion[t+1] = vt + dvt
        
        S = (r - 0.5*volMotion)*dt + np.sqrt(volMotion)*S
        S[0] = np.ones(paths)*np.log(s0)
        self.S = np.exp(np.matrix.cumsum(S,axis=0))
        self.volMotion = volMotion
        self.simtime = time.time() - start
        
        