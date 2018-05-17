import numpy as np
import scipy.stats as sctats
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
start = time.time()
S = np.random.random([intervals+1,paths])
S = -1 + 2*(S > 0.5)
S = S*np.sqrt(dt)*vol + (r - 0.5*vol*vol)*dt
S[0] = np.ones(paths)*np.log(s0)
S = np.exp(np.matrix.cumsum(S,axis=0))

print(time.time() - start)

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

#%%
k = 100
 
#euro
callMotion = (S[-1] - k).clip(0)
putMotion = (k - S[-1]).clip(0)

# class
call = np.exp(-r*T)*np.average(callMotion)
put = np.exp(-r*T)*np.average(putMotion)

#%%
# asian fixed geometric
k = 100

avg = sctats.gmean(S,axis=0)
callMotion = (avg - k).clip(0)
putMotion = (k - avg).clip(0)

# class
call = np.exp(-r*T)*np.average(callMotion)
put = np.exp(-r*T)*np.average(putMotion)

#%%
# asian fixed arithmetic
k = 100

avg = np.average(S,axis=0)
callMotion = (avg - k).clip(0)
putMotion = (k - avg).clip(0)

# class
call = np.exp(-r*T)*np.average(callMotion)
put = np.exp(-r*T)*np.average(putMotion)

#%%
# asian floating geometric
k = 0.9

avg = sctats.gmean(S,axis=0)
callMotion = (S[-1] - k*avg).clip(0)
putMotion = (k*avg - S[-1]).clip(0)

# class
call = np.exp(-r*T)*np.average(callMotion)
put = np.exp(-r*T)*np.average(putMotion)

#%%
# asian floating arithmetic
k = 0.9

avg = np.average(S,axis=0)
callMotion = (S[-1] - k*avg).clip(0)
putMotion = (k*avg - S[-1]).clip(0)

# class
call = np.exp(-r*T)*np.average(callMotion)
put = np.exp(-r*T)*np.average(putMotion)

#%%
# power
k = 50
n = 2.5

power = np.power(S[-1],n)
callMotion = (power - k).clip(0)
putMotion = (k - power).clip(0)

#class
call = np.exp(-r*T)*np.average(callMotion)
put = np.exp(-r*T)*np.average(putMotion)

#%%
# pwoer strike
k = 5.5
n = 2.5

power = np.power(S[-1],n)
callMotion = (power - k**n).clip(0)
putMotion = (k**n - power).clip(0)

#class
call = np.exp(-r*T)*np.average(callMotion)
put = np.exp(-r*T)*np.average(putMotion)

#%%
Z = 110
# average barrier
if s0 < Z: # below
    hitBarrier = np.cumprod(S < Z,axis=0)
if s0 > Z: # above
    hitBarrier = np.cumprod(S > Z,axis=0)

paymentTime = np.array(np.max(np.multiply(timeMatrix,hitBarrier),axis=0))
payoff = np.sum(np.multiply(hitBarrier,S),axis=0) / np.sum(hitBarrier,axis=0)
price = np.average(np.exp(-r*paymentTime)*payoff)


#%%