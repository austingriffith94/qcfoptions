from qcfoptions import barriers, bsoptions
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

print(barriers.UpInPut(s,k,r,zhigh,T,sig,q))
