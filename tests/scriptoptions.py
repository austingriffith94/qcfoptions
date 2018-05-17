# from qcfoptions import barriers, bsoptions
import barriers, bsoptions
import numpy as np
import pandas as pd
import time

#%%
T = 2
s = 1
k = 1
r = 0.015
q = 0.01
sig = 0.25
zlow = 0.6
zhigh = 1.2

print(bsoptions.Lookback(s,k,r,T,sig,q))

S = np.array([0.5,1.0,1.5,2.0])
bsoptions.Lookback(S,k,r,T,sig,q)