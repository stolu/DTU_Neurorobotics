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
#import SetUpModules

#modules = SetUpModules.MODULES(platform, all_modules_ids, n_iter, control_type, eef_trajectory_type)

#####################################################################
############################# Run test ##############################
#####################################################################


def fforward(modules, kp, ki, kv):
    modules.setModuleMotorsPos(0)
    time.sleep(1)
    t = modules.module[0].dt 
    
    for j in range(modules.n_iter):
        begin_time = time.time()
        for i in range(len(modules.ModuleID)):
            if j > 0:
                modules.module[i].errp[:, j+1] = (modules.module[i].q[:, j] - modules.module[i].posr[:, j]) + modules.module[i].etp[:, j]
                modules.module[i].errv[:, j+1] = (modules.module[i].errp[:, j+1] - modules.module[i].errp[:, j]) / modules.module[i].dt
                modules.module[i].erra[:, j+1] = 0.22
            
            if j > 2:
                modules.module[i].D[:, j] = modules.module[i].D[:, j - 1] + (modules.module[i].erra[:, j+1] * ki[i]) + (modules.module[i].errp[:, j+1] * kp[i]) + (modules.module[i].errv[:, j+1] * kv[i])
            else:
                modules.module[i].D[:, j] = (modules.module[i].erra[:, j+1] * ki[i]) + (modules.module[i].errp[:, j+1] * kp[i]) + (modules.module[i].errv[:, j+1] * kv[i])
    
            tau = dynamics.rne(modules.fab17, [modules.module[i].q1[j], modules.module[i].q2[j]], [0, 0], [1, 1], [0, 0, 0] )
    
            modules.module[i].torquesLF[0, j] = np.round(tau[0, 0] * modules.module[i].D[0, j],2)
            modules.module[i].torquesLF[1, j] = np.round(tau[0, 1] * modules.module[i].D[1, j],2)
    
        #for i in range(len(modules.ModuleID)):
            input_data = []
            input_data.append(modules.module[i].q1[j]) 
            input_data.append(modules.module[i].q2[j])
            input_data.append(modules.module[i].q1d[j]) 
            input_data.append(modules.module[i].q2d[j]) 
            input_data.append(modules.module[i].posr[0, j]) 
            input_data.append(modules.module[i].posr[1, j]) 
            input_data.append(modules.module[i].velr[0, j]) 
            input_data.append(modules.module[i].velr[1, j])
            modules.module[i].input_lwpr = np.array(input_data)
            
        # predictions
        #for i in range(len(modules.ModuleID)):
            if j>2:
                modules.module[i].normtorquesLF[0, j] = (modules.module[i].torquesLF[0, j] - modules.module[i].torquesLF[0].min()) / (modules.module[i].torquesLF[0].max()-modules.module[i].torquesLF[0].min())
                modules.module[i].normtorquesLF[1, j] = (modules.module[i].torquesLF[1, j] - modules.module[i].torquesLF[1].min()) / (modules.module[i].torquesLF[1].max()-modules.module[i].torquesLF[1].min())
                #print('norm', np.round(modules.module[i].normtorquesLF[:,j],2))
                # LWPR predictions
                (modules.module[i].torqLWPR[:, j], modules.module[i].torquesDCN[:, j], modules.module[i].weights_mod) = modules.mlcj[i].ML_prediction(modules.module[i].input_lwpr, np.round(modules.module[i].normtorquesLF[:, j],2))
           
            if modules.module[i].torquesLF[0, j] < 0:
                        modules.module[i].torquestot[0, j] = modules.module[i].torquesLF[0, j] - np.round(modules.module[i].torquesDCN[0, j],2) + np.round(modules.module[i].torqLWPR[0, j],2) 
            else:
                        modules.module[i].torquestot[0, j] = modules.module[i].torquesLF[0, j] + np.round(modules.module[i].torquesDCN[0, j],2) + np.round(modules.module[i].torqLWPR[0, j],2)  
                 
            if modules.module[i].torquesLF[1, j] < 0:    
                        modules.module[i].torquestot[1, j] = modules.module[i].torquesLF[1, j] - np.round(modules.module[i].torquesDCN[1, j],2) + np.round(modules.module[i].torqLWPR[1, j],2)
            else:
                        modules.module[i].torquestot[1, j] = modules.module[i].torquesLF[1, j] + np.round(modules.module[i].torquesDCN[1, j],2) + np.round(modules.module[i].torqLWPR[1, j],2)
            
            print("Total Torques:    ", np.round(modules.module[i].torquestot[:, j],2))
            print("LF Torques:    ", np.round(modules.module[i].torquesLF[:, j],2))
            # Avoid torques higher than 100
            if modules.module[i].torquestot[0, j] > 100.0:
                modules.module[i].torquestot[0, j] = 20.0
            if modules.module[i].torquestot[1, j] > 100.0:
                modules.module[i].torquestot[1, j] = 20.0
            # Avoid torques smaller than -100
            if modules.module[i].torquestot[0, j] < -100.0:
                modules.module[i].torquestot[0, j] = -20.0
            if modules.module[i].torquestot[1, j] < -100.0:
               modules.module[i].torquestot[1, j] = -20.0 
            # Control in motor torques
            modules.setModuleMotorsTorque(j)
            
            print("DCN:    ", np.round(modules.module[i].torquesDCN[:, j], 2))
            #print("C output: ", modules.module[i].Ctorques[:, j])
            print("LWPR: ", np.round(modules.module[i].torqLWPR[:, j], 2))
            
            # Get motor positions
            modules.getMotorsPosVel(j)
    
            #print("t: ", t)
            # Compute errors
            modules.module[i].etp[:, j+1] = modules.module[i].q[:, j] - modules.module[i].posr[:, j+1] 
            modules.module[i].etv[:, j+1] = modules.module[i].qd[:, j] - modules.module[i].velr[:, j+1] #
            #modules.module[i].errp[:, j+1] = modules.module[i].etp[:, j+1]
            #modules.module[i].errv[:, j+1] =(modules.module[i].errp[:, j+1] - modules.module[i].errp[:, j]) / modules.module[i].dt
            #modules.module[i].erra[:, j+1] = 0 #0.22
            print("errp: ", modules.module[i].etp[:, j+1])
            print("errv: ", modules.module[i].etv[:, j+1])
            
            input_data = []
            #for i in range(len(modules.ModuleID)):
            input_data.append(modules.module[i].q1[j]) 
            input_data.append(modules.module[i].q2[j])
            input_data.append(modules.module[i].q1d[j]) 
            input_data.append(modules.module[i].q2d[j]) 
            input_data.append(modules.module[i].posr[0, j+1]) 
            input_data.append(modules.module[i].posr[1, j+1])
            input_data.append(modules.module[i].velr[0, j+1]) 
            input_data.append(modules.module[i].velr[1, j+1])
            
            # LWPR update
            modules.mlcj[i].ML_update(modules.module[i].input_lwpr, modules.module[i].torquestot[:, j])
    
            modules.module[i].num_rfs = modules.mlcj[i].ML_rfs()
            # plt.plot(torqLWPR)
            print("\n\n")
            t = (time.time() - begin_time)
            print("Modules_{} ".format(modules.ModuleID[i]), "- j: ", j, "- t: ", np.round(t,2))
    
    #####################################################################
    ######################## Termination of test ########################
    #####################################################################
    now = datetime.now()
    for i in range(len(modules.ModuleID)): 
        scipy.io.savemat('TestFF_1box_{0}modules_fab{1}_{2}_{3}_{4}.mat'.format(len(modules.ModuleID),
                                                                            modules.ModuleID[i], 
                                                                            now.strftime('%d-%m-%Y_%H:%M'), 
                                                                            modules.mlcj[i].model.num_rfs, 
                                                                            modules.mlcj[i].model.init_D[0][0]),
                     dict(q0 = modules.module[i].q1, q1 = modules.module[i].q2, q0d = modules.module[i].q1d, q1d = modules.module[i].q2d, 
                          velr0 = modules.module[i].velr[0], velr1 = modules.module[i].velr[1], posr0 = modules.module[i].posr[0], 
                          posr1 = modules.module[i].posr[1], errp0 = modules.module[i].etp[0], errp1 = modules.module[i].etp[1], 
                          errv0 = modules.module[i].etv[0], errv1 = modules.module[i].etv[1], torquesLF0 = modules.module[i].torquesLF[0], 
                          torquesLF1 = modules.module[i].torquesLF[1], DCN0 = modules.module[i].torquesDCN[0], DCN1 = modules.module[i].torquesDCN[1], 
                          torquesTot0 = modules.module[i].torquestot[0], torquesTot1 = modules.module[i].torquestot[1], torquesLWPR0 = modules.module[i].torqLWPR[0], 
                          torquesLWPR1 = modules.module[i].torqLWPR[1], weights_mod_LWPR = modules.module[i].weights_mod))
    
        print('FeedForward Thread-test for Module_{}: finishing'.format(modules.ModuleID))
        
    return True
