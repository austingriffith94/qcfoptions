from qcfoptions import simulation
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

#%%
s0 = 1
r = 0.015
vol = 0.35
T = 1
dts = 0.001
path = 10000

sim = simulation.Simple(s0,r,T,vol,dt=dts,paths=path)
sPaths = sim.S

avgS = np.mean(sPaths,axis=1)

plt.plot(sim.timeMatrix,sPaths)
plt.plot(sim.timeMatrix,avgS)
plt.grid()
plt.xlabel('Time')
plt.ylabel('Underlying Value ($)')
plt.show()


