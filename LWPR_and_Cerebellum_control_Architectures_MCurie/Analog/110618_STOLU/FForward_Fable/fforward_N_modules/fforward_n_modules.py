#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""============================================================
* File Name: recurrent_n_modules.py
* Purpose: Recurrent control loop - Control several modules of Fable at the same
            time, using one single dongle and from just
            one script
* Creation Date : 05-09-2017
* Last Modified : 05-09-2017
__author__  	= "Silvia Tolu"
__credits__ 	= ["Silvia Tolu"]
__maintainer__ 	= "Silvia Tolu"
__email__ 	= ["stolu@elektro.dtu.dk"]

============================================================"""

import sys, time, math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from datetime import datetime, date


sys.path.append("/home/silvia/workspace/DTU_neurorobotics/lib/fable_1_api/python/api")
sys.path.append("/home/silvia/workspace/DTU_neurorobotics/lib/Robot_toolboxPy/robot")
sys.path.append("/home/silvia/workspace/DTU_neurorobotics/projects/lwpr_fable_fforward/scripts/fforward_N_modules")

import dynamics
import SetUpModules

#####################################################################
###################### Variables initialization #####################
#####################################################################

platform = "linux"              # Select platform between "linux" or "mac"
all_modules_ids = [74]      # Select the list of modules ids[74, 80]
n_iter = 10500                    # Select the number of iterations
control_type = 0                # Select the control architecture:
                                              # 0: fforward
                                              # 1: recurrent
                                              # 2: hybrid
eef_trajectory_type = 2    # Select the trajectory of the end effector:
                                              # 0: circle
                                              # 1: crawl
                                              # 2: eigth
                                              
#constant c1,c2,c3= 1 for vertical down - constant c1= 2 c2,c3=2.5for vertical up position
c1 = 1 #2.0 
c2 = 1 #2.5
c3 = 1 #2.5

kp = [7.5*c2, 7.5*c2]       # Select the kp pf the Learning Feedback controller
ki = [1.0*c1, 1.0*c1]       # Select the ki pf the Learning Feedback controller
kv = [6.4*c3, 6.4*c3]       # Select the kv pf the Learning Feedback controller

modules = SetUpModules.MODULES(platform, all_modules_ids, n_iter, control_type, eef_trajectory_type)

#####################################################################
############################# Run test ##############################
#####################################################################

modules.setModuleMotorsPos(0)
time.sleep(1)
t = modules.module[0].dt        

for j in range(n_iter):
   
    end_time = time.time()
    
    for i in range(len(modules.ModuleID)):
        if j > 1:
            modules.module[i].D[:, j] = modules.module[i].D[:, j - 1] + (modules.module[i].erra[:, j] * ki[i]) + (modules.module[i].errp[:, j] * kp[i]) + (modules.module[i].errv[:, j] * kv[i])
        else:
            modules.module[i].D[:, j] = (modules.module[i].erra[:, j] * ki[i]) + (modules.module[i].errp[:, j] * kp[i]) + (modules.module[i].errv[:, j] * kv[i])

        tau = dynamics.rne(modules.fab17, [modules.module[i].q1[j], modules.module[i].q2[j]], [0, 0], [1, 1], [0, 0, 0] )

        modules.module[i].torquesLF[0, j+1] = tau[0, 0] * modules.module[i].D[0, j]
        modules.module[i].torquesLF[1, j+1] = tau[0, 1] * modules.module[i].D[1, j]

    input_data = []
    for i in range(len(modules.ModuleID)):
        input_data.append(modules.module[i].q1[j]) 
        input_data.append(modules.module[i].q2[j])
        input_data.append(modules.module[i].q1d[j]) 
        input_data.append(modules.module[i].q2d[j]) 
        input_data.append(modules.module[i].posr[0, j]) 
        input_data.append(modules.module[i].posr[1, j]) 

    modules.input_lwpr = np.array(input_data)
    # predictions
    for i in range(len(modules.ModuleID)):
        (modules.module[i].torqLWPR[:, j], modules.module[i].Ctorques[:, j]) = modules.mlcj[i].ML_prediction(modules.input_lwpr, modules.module[i].torquesLF[:, j])

        modules.module[i].torquestot[0, j] = modules.module[i].torquesLF[0, j] + modules.module[i].torqLWPR[0, j] + modules.module[i].Ctorques[0, j]
        modules.module[i].torquestot[1, j] = modules.module[i].torquesLF[1, j] + modules.module[i].torqLWPR[1, j] + modules.module[i].Ctorques[1, j]
      
        # Avoid torques higher than 100
        if modules.module[i].torquestot[0, j] > 100.0:
            modules.module[i].torquestot[0, j] = 100.0
        if modules.module[i].torquestot[1, j] > 100.0:
            modules.module[i].torquestot[1, j] = 100.0
        # Avoid torques smaller than -100
        if modules.module[i].torquestot[0, j] < -100.0:
            modules.module[i].torquestot[0, j] = -100.0
        if modules.module[i].torquestot[1, j] < -100.0:
           modules.module[i].torquestot[1, j] = -100.0
        
        print("j: ", j)
        print("Current torquesLF:    ", modules.module[i].torquesLF[:, j+1])
        print("C output: ", modules.module[i].outputC[:, j+1])
        print("lwpr output: ", modules.module[i].pLWPR[0:2, j+1])

#        print("torquestot: ", torquestot[:, j])    


        # Control in motor torques
        modules.setModuleMotorsTorque(j)
        # Get motor positions
        modules.getMotorsPos(j)

        #print("t: ", t)
        # Compute errors
        modules.module[i].errp[:, j+1] = modules.module[i].q[:, j] - modules.module[i].posr[:, j+1] 
        modules.module[i].errv[:, j+1] = (modules.module[i].errp[:, j+1] - modules.module[i].errp[:, j])/modules.module[i].dt
        modules.module[i].erra[:, j+1] = 0.22
        
        #errp[1, j+1] = (q2[j] - posr[1, j+1])
        #errv[0, j+1] = (errp[0, j+1] - errp[0, j]) / dt #(q1d[j] - velr[0, j+1]) #
        #errv[1, j+1] = (errp[1, j+1] - errp[1, j]) / dt #(q2d[j] - velr[1, j+1]) #
        #erra[0, j+1] = 0.22 #q1dd[j] #0.22  #erra[0, j] + (errp[0, j+1] * dt)
        #erra[1, j+1] = 0.22 #q2dd[j] #0.22  # 


        #print("errp: ", errp[:, j+1])
        #print("errv: ", errv[:, j+1])
        #print("erra: ", erra[:, j+1])
        ## print("veld: ", [q1d[j], q2d[j]])
        #print("veld1: ", q1d[j])
        #print("veld2: ", q2d[j])   
        #print("velr: ", velr[:, j+1])
        #print("posr: ", posr[:, j+1])

        # Update models
        print("update")
        modules.mlcj[i].ML_update(modules.input_lwpr, modules.module[i].torquestot[:, j])

        modules.mlcj[i].ML_rfs()
        # plt.plot(torqLWPR)
        end_time = time.time()

#####################################################################
######################## Termination of test ########################
#####################################################################
# Termination of api and class usage

del modules
time.sleep(1)
print("Test - Done...")
SetUpModules.api.sleep(1)
SetUpModules.api.terminate()
