"""
Created on Feb 10 2015
__author__ = 'slyto'
"""

import sys, time, math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

#sys.path.append("../../FableApi/src/api/python/api")
sys.path.append("../../Fable_api_PY_NEW/python/api")
#sys.path.append("../../FableApi_Dropbox/python/api")
sys.path.append("../../Robot_toolboxPy/robot")
sys.path.append("fforward_macFableNew")

from lwpr import *
from AFEL_LWPRandC_class import *
from Robot import *
from fableAPI import *  # FableAPI
from simfable import *
from dynamics import *

print("Go")

n_iter = 20000
njoints = 2

# ki = [0.7, 2.0]  #with desired acc
# kp = [2.9, 20.1] #with desired acc
# kv = [2.0, 17.0] # with desired acc
ki = [1.0, 1.0]  # 1.0 dt = 1 acc const = 0.22
kp = [7.5, 7.5]  # 7.9 dt = 0.01 acc const = 0.22
kv = [6.4, 6.4]  # 6.4 dt = 0.01 acc const = 0.22

ModuleID = 80

grav = [0, 0, 9.81]
api = FableAPI()
fab = moduleFable()
fab17 = fab.Fab()
api.setup(1)  # blocking=True)


# Example of use of class MLandC
# Class object for every joint link
mlcj = MLandC(4, njoints)

# Variable definitions
A1 = 10 #12
A2 = 10 #7.5
phase = math.pi / 2
dt = 0.01
t0 = 0


q1 = [0 for k in range(n_iter + 1)]
q2 = [0 for a in range(n_iter + 1)]
q1d = [0 for c in range(n_iter + 1)]
q2d = [0 for v in range(n_iter + 1)]
q1dd = [0 for o in range(n_iter + 1)]
q2dd = [0 for l in range(n_iter + 1)]
x = [0 for k in range(n_iter + 1)]
y = [0 for k in range(n_iter + 1)]
z = [0 for k in range(n_iter + 1)]
posr = np.zeros((njoints, n_iter + 1), dtype=np.double)
velr = np.zeros((njoints, n_iter + 1), dtype=np.double)
accr = np.zeros((njoints, n_iter + 1), dtype=np.double)
D = np.zeros((njoints, n_iter + 1), dtype=np.double)
erra = np.zeros((njoints, n_iter + 1), dtype=np.double)
errp = np.zeros((njoints, n_iter + 1), dtype=np.double)
errv = np.zeros((njoints, n_iter + 1), dtype=np.double)
ea = np.zeros((njoints, n_iter + 1), dtype=np.double)
ep = np.zeros((njoints, n_iter + 1), dtype=np.double)
ev = np.zeros((njoints, n_iter + 1), dtype=np.double)
torquesLF = np.zeros((njoints, n_iter + 1), dtype=np.double)
torqLWPR = np.zeros((njoints, n_iter + 1), dtype=np.double)
Ctorques = np.zeros((njoints, n_iter + 1), dtype=np.double)
torquestot = np.zeros((njoints, n_iter + 1), dtype=np.double)
posC = np.zeros((njoints, n_iter + 1), dtype=np.double)
velC = np.zeros((njoints, n_iter + 1), dtype=np.double)
pLWPR = np.zeros((njoints, n_iter + 1), dtype=np.double)
vLWPR = np.zeros((njoints, n_iter + 1), dtype=np.double)

# maxSpeed = 1
# maxTorque = 50

class Trajectory2j_eightfig(object):

    def __init__(self):
        self.A1 = A1
        self.A2 = A2
        self.phase = phase
        self.dt = dt
        self.t0 = t0
        self.p = n_iter

    def calc_trajectory(self):
        for i in range(self.p):
            # Joint coordinates
            q1dd[i] = math.sin(2*math.pi*self.t0) * self.A1 #(-math.sin(2*math.pi*self.t0)* (self.A1 * 4 * math.pow(math.pi,2)))
            q1d[i] = ((-1/2)*math.pi) * math.cos(2*math.pi*self.t0) * (self.A1) #math.cos(2*math.pi*self.t0) * (self.A1 * 2 * math.pi)
            q1[i] = (-math.pow(((1/2)*math.pi),2)) * math.sin(2*math.pi*self.t0) * self.A1 #math.sin(2*math.pi*self.t0) * self.A1
            
            
            q2dd[i] = math.cos(2*math.pi*self.t0 + math.pi/4) * self.A1 # -math.sin(2*math.pi*self.t0 + math.pi/4)* (self.A1 * 2 * math.pi)
            q2d[i] =((1/2)*math.pi) * math.sin(2*math.pi*self.t0 + math.pi/4) * self.A1  #(-math.cos(2*math.pi*self.t0 + math.pi/4)* (self.A1 * 4 * math.pow(math.pi,2)))
            q2[i] = (-math.pow(((1/2)*math.pi),2))*math.cos(2*math.pi*self.t0 + math.pi/4) * self.A1 #math.cos(2*math.pi*self.t0 + math.pi/4) * self.A1
            self.t0 += self.dt
            # Cartesian coordinates
            x[i] = fab.l1 * math.cos(q1[i]*math.pi/180) * (fab.l2 * math.cos(q2[i]*math.pi/180) + math.cos(q2[i]*math.pi/180));
            y[i] = fab.l1 * math.sin(q1[i]*math.pi/180) * (fab.l2 * math.cos(q2[i]*math.pi/180) + math.cos(q2[i]*math.pi/180));
        return q1, q2, q1d, q2d, q1dd, q2dd, x, y

T_circle = Trajectory2j_eightfig()
(q1, q2, q1d, q2d, q1dd, q2dd, x, y) = T_circle.calc_trajectory()

print("size q1: ", size(q1))
#plt.plot(q1, q2)
#plt.plot(q1, q2)
plt.plot(x, y)

# battery level of motors
#a = api.getModuleBatteryLevel(ModuleID, nBatCells=1)
#b = api.getModuleBatteryLevel(ModuleID, nBatCells=2)
#c = api.getModuleBatteryLevel(ModuleID, nBatCells=3)
#print("a, b, c: ", a, b, c)

api.setModuleMotorPosition(ModuleID, 0, q1[0])
api.setModuleMotorPosition(ModuleID, 1, q2[0])
plt.show()    
time.sleep(1)
end_time = time.time()
for j in range(n_iter):
    for i in range(njoints):     
        # Feedback error learning
        if j > 1:
            D[i, j] = D[i, j - 1] + (erra[i, j]) * ki[i] + (errp[i, j] * (kp[i])) + (errv[i, j] * (kv[i]))
            print("D: ", D[i, j])
        else:
            D[i, j] = (erra[i, j]) * ki[i] + (errp[i, j] * (kp[i])) + (errv[i, j] * (kv[i]))
            print("D: ", D[i, j])

    tau = rne(fab17, [q1[j], q2[j]], [0, 0], [1, 1], [0, 0, 0])
#    print("rne")
#    #print("tau:", tau)
#
    torquesLF[0, j] = tau[0, 0] * D[0, j]
    torquesLF[1, j] = tau[0, 1] * D[1, j]

    # predictions
    (torqLWPR[:, j], Ctorques[:, j]) = mlcj.ML_prediction(np.array([q1[j], q2[j], posr[0, j], posr[1, j]]), torquesLF[:, j])
#    if Ctorques[0, j] == "Nan":
#        Ctorques[0, j] = 0
#    if Ctorques[1, j] == "Nan":
#       Ctorques[1, j] = 0

    torquestot[0, j] = torquesLF[0, j] + torqLWPR[0, j] + Ctorques[0, j]
    torquestot[1, j] = torquesLF[1, j] + torqLWPR[1, j] + Ctorques[1, j]
  
    # Avoid torques higher than 100
    if torquestot[0, j] > 100.0:
        torquestot[0, j] = 100.0
    if torquestot[1, j] > 100.0:
        torquestot[1, j] = 100.0
    # Avoid torques smaller than -100
    if torquestot[0, j] < -100.0:
        torquestot[0, j] = -100.0
    if torquestot[1, j] < -100.0:
        torquestot[1, j] = -100.0
    print("j: ", j)
    print("torquesLF: ", torquesLF[:, j])
    print("torques LWPR: ", torqLWPR[:, j])
    print("torquestot: ", torquestot[:, j])    
    print("C torques: ", Ctorques[:, j])

    # Control in motor torques
    api.setModuleMotorTorque(ModuleID, 0, -torquestot[0, j], 1, -torquestot[1, j], ack=False)
  
    velr[0, j+1] = -api.getModuleMotorSpeed(ModuleID, 0) + 10 # (posr[0,j+1] - posr[0,j] / 2*t)  # 
    velr[1, j+1] = -api.getModuleMotorSpeed(ModuleID, 1) + 10 # (posr[1,j+1] - posr[1,j] / 2*t)  #  
  
    time.sleep(dt)
    #t = (time.time() - end_time)
    
    # Receive feedback positions from motors
    posr[0, j+1] = api.getModuleMotorPosition(ModuleID, 0)
    posr[1, j+1] = api.getModuleMotorPosition(ModuleID, 1)
    
    #print("t: ", t)
    # Compute errors
    errp[0, j+1] = (q1[j] - posr[0, j+1])
    errp[1, j+1] = (q2[j] - posr[1, j+1])
    errv[0, j+1] = (errp[0, j+1] - errp[0, j]) / dt #(q1d[j] - velr[0, j+1]) #
    errv[1, j+1] = (errp[1, j+1] - errp[1, j]) / dt #(q2d[j] - velr[1, j+1]) #
    erra[0, j+1] = 0.22  # q1dd[j] #erra[0, j] + (errp[0, j+1] * dt)
    erra[1, j+1] = 0.22  # q2dd[j]


    print("errp: ", errp[:, j+1])
    print("errv: ", errv[:, j+1])
    print("erra: ", erra[:, j+1])
    # print("veld: ", [q1d[j], q2d[j]])
    print("velr: ", velr[:, j+1])
    print("posr: ", posr[:, j+1])

    # Update models
    print("update")
    # mlcj.ML_update(np.array([q1[j], q2[j], q1dd[j], q2dd[j], posr[0, j], posr[1, j]]), (torquestot[:, j]))
    mlcj.ML_update(np.array([q1[j], q2[j], posr[0, j], posr[1, j]]), (torquestot[:, j]))
 
    mlcj.ML_rfs()
    # plt.plot(torqLWPR)
    end_time = time.time()

# Save with Matlab compatibility
scipy.io.savemat('TestForward_15_04_2016_Mac_eightFig.mat', dict(x=x, y=y, q1=q1, q2=q2, q1d=q1d, q2d=q2d, posr1=posr[0], posr2=posr[1], velr1 = velr[0], velr2 = velr[1], errp1=errp[0], errp2=errp[1], torquesLF1=torquesLF[0], torquesLF2=torquesLF[1], torqLWPR1=torqLWPR[0], torqLWPR2=torqLWPR[1], Ctorques1=Ctorques[0], Ctorques2=Ctorques[1], torquestot1=torquestot[0], torquestot2=torquestot[1]))
# scipy.io.savemat('TestForward_PDonly_04_02_2016_Mac.mat', dict(q1=q1, q2=q2, posr1=posr[0], posr2=posr[1], errp1=errp[0], errp2=errp[1], torquesLF1=torquesLF[0], torquesLF2=torquesLF[1], torqLWPR1=torqLWPR[0], torqLWPR2=torqLWPR[1], Ctorques1=Ctorques[0], Ctorques2=Ctorques[1], torquestot1=torquestot[0], torquestot2=torquestot[1]))

# Plot
m = []
for i in range(0, n_iter + 1):
    m.append(i)
plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(m, torquesLF[1], label="Torques LF joint 1")
plt.plot(m, torquesLF[0], label="Torques LF joint 2")
plt.plot(m, torqLWPR[0], label="Torques LWPR joint 1")
plt.plot(m, torqLWPR[1], label="Torques LWPR joint 2")
plt.plot(m, Ctorques[0], label="Torques Cereb joint 1")
plt.plot(m, Ctorques[1], label="Torques Cereb joint 2")
plt.plot(m, torquestot[0], label="Torques tot 1")
plt.plot(m, torquestot[1], label="Torques tot 2")
plt.axis([0, n_iter, -200.00, 200.00])
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.subplot(2, 1, 2)
plt.plot(m, errp[0], label="Error pos joint 1")
plt.plot(m, errp[1], label="Error pos joint 2")
# plt.plot(m, errv[0])
# plt.plot(m, errv[1])
plt.axis([0, n_iter, -200.00, 200.00])
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
           
plt.figure(2)
plt.plot(m, q1, label="Desired pos joint 1")
plt.plot(m, q2, label="Desired pos joint 2")
plt.plot(m, posr[0], label="Real pos joint 1")
plt.plot(m, posr[1], label="Real pos joint 2")
plt.axis([0, n_iter, -100.00, 100.00])
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()

api.terminate() 
print("Done...")
