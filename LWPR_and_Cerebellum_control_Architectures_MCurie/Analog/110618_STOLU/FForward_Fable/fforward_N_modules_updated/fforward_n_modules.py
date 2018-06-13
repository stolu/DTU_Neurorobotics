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


sys.path.append("/users/stolu/workspace/DTU_neurorobotics/lib/fable_1_api/python/api")
sys.path.append("/users/stolu/workspace/DTU_neurorobotics/lib/Robot_toolboxPy/robot")
sys.path.append("/users/stolu/workspace/DTU_neurorobotics/projects/lwpr_fable_forward_recurrent/scripts")

import dynamics
import SetUpModules

#####################################################################
###################### Variables initialization #####################
#####################################################################

platform = "mac"              # Select platform between "linux" or "mac"
all_modules_ids = [74]      # Select the list of modules ids[74, 80]
n_iter = 20500                    # Select the number of iterations
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
    
    for i in range(len(modules.ModuleID)):
        end_time = time.time()
        modules.module[i].errp[:, j+1] = modules.module[i].q[:, j] - modules.module[i].posr[:, j] 
        modules.module[i].errv[:, j+1] = (modules.module[i].errp[:, j+1] - modules.module[i].errp[:, j]) / modules.module[i].dt
        modules.module[i].erra[:, j+1] = 0.22
        
        if j > 1:
            modules.module[i].D[:, j] = modules.module[i].D[:, j - 1] + (modules.module[i].erra[:, j+1] * ki[i]) + (modules.module[i].errp[:, j+1] * kp[i]) + (modules.module[i].errv[:, j+1] * kv[i])
        else:
            modules.module[i].D[:, j] = (modules.module[i].erra[:, j+1] * ki[i]) + (modules.module[i].errp[:, j+1] * kp[i]) + (modules.module[i].errv[:, j+1] * kv[i])

        tau = dynamics.rne(modules.fab17, [modules.module[i].q1[j], modules.module[i].q2[j]], [0, 0], [1, 1], [0, 0, 0] )

        modules.module[i].torquesLF[0, j] = tau[0, 0] * modules.module[i].D[0, j]
        modules.module[i].torquesLF[1, j] = tau[0, 1] * modules.module[i].D[1, j]

    
    for i in range(len(modules.ModuleID)):
        input_data = []
        input_data.append(modules.module[i].q1[j]) 
        input_data.append(modules.module[i].q2[j])
        input_data.append(modules.module[i].q1d[j]) 
        input_data.append(modules.module[i].q2d[j]) 
        input_data.append(modules.module[i].posr[0, j]) 
        input_data.append(modules.module[i].posr[1, j]) 
        modules.module[i].input_lwpr = np.array(input_data)
        
    # predictions
    for i in range(len(modules.ModuleID)):
        (modules.module[i].torqLWPR[:, j], modules.module[i].Ctorques[:, j], modules.module[i].outputDCN[:, j], modules.module[i].weights_mod) = modules.mlcj[i].ML_prediction(modules.module[i].input_lwpr, modules.module[i].torquesLF[:, j])

        modules.module[i].torquestot[0, j] = modules.module[i].torquesLF[0, j] + modules.module[i].torqLWPR[0, j] + modules.module[i].outputDCN[0, j] #modules.module[i].Ctorques[0, j]
        modules.module[i].torquestot[1, j] = modules.module[i].torquesLF[1, j] + modules.module[i].torqLWPR[1, j] + modules.module[i].outputDCN[1, j] #modules.module[i].Ctorques[1, j]
      
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

#        print("torquestot: ", torquestot[:, j])    


        # Control in motor torques
        modules.setModuleMotorsTorque(j)
        
        print("j: ", j)
        #print("Current torquesLF:    ", modules.module[i].torquesLF[:, j])
        print("Torques Tot:    ", modules.module[i].torquestot[:, j])
        print("DCN output:    ", modules.module[i].outputDCN[:, j])
        print("C output: ", modules.module[i].Ctorques[:, j])
        print("LWPR output: ", modules.module[i].torqLWPR[:, j])
        
        # Get motor positions
        modules.getMotorsPos(j)

        #print("t: ", t)
        # Compute errors
        #modules.module[i].errp[:, j+1] = modules.module[i].q[:, j] - modules.module[i].posr[:, j+1] 
        #modules.module[i].errv[:, j+1] = (modules.module[i].errp[:, j+1] - modules.module[i].errp[:, j])/modules.module[i].dt
        #modules.module[i].erra[:, j+1] = 0.22
        
        
        input_data = []
        for i in range(len(modules.ModuleID)):
            input_data.append(modules.module[i].q1[j]) 
            input_data.append(modules.module[i].q2[j])
            input_data.append(modules.module[i].q1d[j]) 
            input_data.append(modules.module[i].q2d[j]) 
            input_data.append(modules.module[i].posr[0, j+1]) 
            input_data.append(modules.module[i].posr[1, j+1])
        
        # Update models
        print("update")
        modules.mlcj[i].ML_update(modules.module[i].input_lwpr, modules.module[i].torquestot[:, j])

        modules.module[i].num_rfs = modules.mlcj[i].ML_rfs()
        # plt.plot(torqLWPR)
        end_time = time.time()
        print("\n\n")
        t = (time.time() - end_time)
        print("Modules_{} ".format(modules.ModuleID[i]), "- j: ", j, "- t: ", t)

#####################################################################
######################## Termination of test ########################
#####################################################################
now = datetime.now()
for i in range(len(modules.ModuleID)): 
    scipy.io.savemat('TestFF_1box_{0}modules_fab{1}_{2}_{3}_{4}.mat'.format(len(modules.ModuleID),
                                                                        modules.ModuleID[i], 
                                                                        now.strftime('%d-%m-%Y_%H:%M'), 
                                                                        modules.module[i].num_rfs, 
                                                                        modules.mlcj[i].model[i].init_D[0]),
                 dict(q0 = modules.module[i].q1, q1 = modules.module[i].q2, q0d = modules.module[i].q1d, q1d = modules.module[i].q2d, 
                      velr0 = modules.module[i].velr[0], velr1 = modules.module[i].velr[1], posr0 = modules.module[i].posr[0], 
                      posr1 = modules.module[i].posr[1], errp0 = modules.module[i].errp[0], errp1 = modules.module[i].errp[1], 
                      errv0 = modules.module[i].errp[0], errv1 = modules.module[i].errv[1], torquesLF0 = modules.module[i].torquesLF[0], 
                      torquesLF1 = modules.module[i].torquesLF[1], DCN0 = modules.module[i].outputDCN[0], DCN1 = modules.module[i].outputDCN[1], 
                      torquesC0 = modules.module[i].outputC[0], torquesC1 = modules.module[i].outputC[1], torquesLWPR0 = modules.module[i].torqLWPR[0], 
                      torquesLWPR1 = modules.module[i].torqLWPR[1], weights_mod_LWPR = modules.module[i].weights_mod))

    print('FeedForward Thread-test for Module_{}: finishing'.format(modules.ModuleID))


# Termination of api and class usage
modules.terminate()
del modules
print("Test - Done...")

