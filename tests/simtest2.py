import simulation

s0 = 1
r = 0.015
T = 2
vol = 0.25
phi = -0.4
kappa = 8
xi = 0.35
dt = 0.001
paths = 1000


simulation.hestonSim(s0,r,T,vol,phi,kappa,xi,dt,paths)
