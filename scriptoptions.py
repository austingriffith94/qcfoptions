import bsoptions
import barriers
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm
from pandas_datareader import data as pdr
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
