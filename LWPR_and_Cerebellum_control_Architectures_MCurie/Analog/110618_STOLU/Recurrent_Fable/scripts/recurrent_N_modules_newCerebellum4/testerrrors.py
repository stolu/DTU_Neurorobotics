import sys, time, math
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from datetime import datetime, date
#from _random import Random
import random


sys.path.append("/users/stolu/workspace/DTU_neurorobotics/lib/fable_1_api/python/api")
sys.path.append("/users/stolu//workspace/DTU_neurorobotics/lib/Robot_toolboxPy/robot")
sys.path.append("/users/stolu//workspace/DTU_neurorobotics/projects/lwpr_fable_recurrent/scripts/recurrent_N_modules_newCerebellum4")

moving_avesp0 = []

'''epr = np.random.rand(4000,1)
print(len(epr))'''

x = loadmat('TestREC_1box_1modules_fab80_II.mat')
err0 = x['errp0']
print (err0[0])
print(len(err0[0]))

N = int(1/0.005)
print(N)
        
moving_avesp0 = []
dummy = 0.0
for idx in range(len(err0[0])): # enumerate(self.module.epr[0], 1):
    dummy_N = 0.0
    if (idx>N) and (idx+N) < len(err0[0]):
        for n in range (0, N):
            dummy_N += abs(err0[0,idx-N+n])
        moving_avesp0.append(dummy_N / (1.*N))
    elif idx>len(err0[0])-N:
        it = len(err0[0])-idx-1
        for n in range(0,it):
            dummy_N += abs(err0[0,idx-it+n])
        moving_avesp0.append(dummy_N / (1.*it + 2.))
    else:
        dummy += abs(err0[0,idx])
        moving_avesp0.append(dummy/(idx+1.))
          
#print(moving_avesp0)

N = int(1/0.005)
cumsum_err0, moving_aves_err0 = [0], []
for i, x in enumerate(abs(err0[0]), 1):
    cumsum_err0.append(cumsum_err0[i-1] + x)
    if i>=N:
        moving_ave = (cumsum_err0[i] - cumsum_err0[i-N])/N
        #can do stuff with moving_ave here
        moving_aves_err0.append(moving_ave)

plt.plot(moving_aves_err0)
plt.plot(moving_avesp0)
plt.plot(err0[0])
plt.show()