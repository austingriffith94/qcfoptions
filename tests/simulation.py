import numpy as np
import time
from qcfoptions import barriers, bsoptions


#%%
s0 = 100
r = 0.015
vol = 0.25
T = 1

dt = 0.001
intervals = int(T/dt)
paths = 2000

timeInt = np.matrix(np.arange(0,T+dt,dt)).transpose()
timeMatrix = np.matmul(timeInt,np.matrix(np.ones(paths)))
discount = np.exp(-r*timeMatrix)

#%%
bro = time.time()
S = np.random.random([intervals+1,paths])
S = -1 + 2*(S > 0.5)
S = S*np.sqrt(dt)*vol + (r - 0.5*vol*vol)*dt
S[0] = np.ones(paths)*np.log(s0)
S = np.exp(np.matrix.cumsum(S,axis=0))

print(time.time() - bro)

#%%
corr = -0.4 # correlation of price to volatility
kappa = 10 # speed of adjustment
xi = 0.25**2 # volatility of volatility

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
S = np.exp(np.matrix.cumsum(S,axis=0))
print(time.time() - start)