#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""============================================================
* File Name: recurrent_n_modules.py
* Purpose: Recurrent control loop - Control several modules of Fable at the same
            time, using one single dongle and from just
            one script.Smith controller
* Creation Date : 05-02-2018
* Last Modified : 05-02-2018
__author__      = "Silvia Tolu"
__credits__     = ["Silvia Tolu"]
__maintainer__     = "Silvia Tolu"
__email__     = ["stolu@elektro.dtu.dk"]

============================================================"""

import sys, time, math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from datetime import datetime, date


sys.path.append("/users/stolu/workspace/DTU_neurorobotics/lib/fable_1_api/python/api")
sys.path.append("/users/stolu/workspace/DTU_neurorobotics/lib/Robot_toolboxPy/robot")
sys.path.append("/users/stolu/workspace/DTU_neurorobotics/projects/lwpr_fable_forward_recurrent/scripts")

meanerrorp = [0.0, 0.0]
meanerrorv = [0.0, 0.0]

import dynamics

#####################################################################
############################# Run test ##############################
#####################################################################
def recurrent(modules, kp, ki, kv):
    for i in range(len(modules.ModuleID)):
        modules.setModuleMotorsPos(0, i)
    time.sleep(1)
    t = modules.module[0].dt 
        
    for j in range(modules.n_iter):
        begin_time = time.time()
        for i in range(len(modules.ModuleID)):
            # LWPR input data
            input_data = []
            input_data.append(modules.module[i].torquesLF[0, j])                
            input_data.append(modules.module[i].torquesLF[1, j])
            input_data.append(modules.module[i].q1[j]) 
            input_data.append(modules.module[i].q2[j])
            input_data.append(modules.module[i].q1d[j]) 
            input_data.append(modules.module[i].q2d[j]) 
            input_data.append(modules.module[i].posr[0, j]) 
            input_data.append(modules.module[i].posr[1, j]) 
            input_data.append(modules.module[i].velr[0, j]) 
            input_data.append(modules.module[i].velr[1, j])
            modules.input_lwpr = np.array(input_data)
            
            (modules.module[i].output_x, modules.module[i].outputC[:, j+1], modules.module[i].outputDCN[:, j+1], modules.module[i].init_D) = modules.mlcj[i].ML_prediction(
                                                                                                            modules.input_lwpr,
                                                                                                            modules.module[i].etp[:, j],
                                                                                                            modules.module[i].etv[:, j], meanerrorp, meanerrorv, modules.module[i].normetp[:, j], modules.module[i].normetv[:, j])
            modules.module[i].pLWPR[0, j+1] = modules.module[i].output_x[0]
            modules.module[i].vLWPR[0, j+1] = modules.module[i].output_x[1]
            modules.module[i].pLWPR[1, j+1] = modules.module[i].output_x[2]
            modules.module[i].vLWPR[1, j+1] = modules.module[i].output_x[3]
            
            #print("lwpr output: ", modules.module[i].pLWPR[0:2, j+1])
            
            '''if modules.module[i].etp[0, j] < 0:
                modules.module[i].outputDCN[0, j+1] = - modules.module[i].outputDCN[0, j+1]
                  
            if modules.module[i].etv[0, j] < 0:
                modules.module[i].outputDCN[2, j+1] = - modules.module[i].outputDCN[2, j+1]
                  
            if modules.module[i].etp[1, j] < 0:
                modules.module[i].outputDCN[1, j+1] = - modules.module[i].outputDCN[1, j+1]
                  
            if modules.module[i].etv[1, j] < 0:
                modules.module[i].outputDCN[3, j+1] = - modules.module[i].outputDCN[3, j+1]'''
            print("DCN output: ", modules.module[i].outputDCN[:, j+1])
            #print("C output: ", modules.module[i].output_C[:, j+1])
            
            # the best
            #modules.module[i].epd[:, j]  =  (modules.module[i].q[:,j] - modules.module[i].pLWPR[:, j+1]) - modules.module[i].epp[:, j] + (modules.module[i].q[:, j] - modules.module[i].pLWPR[:, j]) - modules.module[i].epp[:, j-1] #+ modules.module[i].etp[:,j] 
            modules.module[i].epd[:, j]  =  (modules.module[i].q[:,j] - modules.module[i].pLWPR[:, j+1]) - modules.module[i].epp[:, j] + modules.module[i].etp[:,j] - modules.module[i].epp[:, j-1] #The BEST
    
            #modules.module[i].epd[:, j]  =  (modules.module[i].q[:,j] - modules.module[i].pLWPR[:, j+1]) - modules.module[i].epp[:, j] # Smith model 2
            #modules.module[i].epd[:, j]  =  (modules.module[i].q[:,j] - modules.module[i].pLWPR[:, j+1]) + modules.module[i].pLWPR[:, j] - modules.module[i].posr[:, j] # Smith model 3
      
            #modules.module[i].epd[:, j]  = modules.module[i].epr[:, j] # No SMITH
            
            modules.module[i].ea[:, j+1] = 0.22
            modules.module[i].ep[:, j+1] = modules.module[i].epd[:, j] + modules.module[i].outputDCN[0:2, j+1]
            modules.module[i].ev[:, j+1] = ((modules.module[i].ep[:, j+1] - modules.module[i].ep[:, j]) / modules.module[0].dt) + modules.module[i].outputDCN[2:4, j+1]
            # Feedback error learning
            if j > 1:
                modules.module[i].D[:, j] = modules.module[i].D[:, j - 1] + (modules.module[i].ea[:, j+1] * ki[i]) + (modules.module[i].ep[:, j+1] * kp[i]) + (modules.module[i].ev[:, j+1] * kv[i])
            else:
                modules.module[i].D[:, j] = (modules.module[i].ea[:, j+1] * ki[i]) + (modules.module[i].ep[:, j+1] * kp[i]) + (modules.module[i].ev[:, j+1] * kv[i])
    
            tau = dynamics.rne(modules.fab17, [modules.module[i].q1[j], modules.module[i].q2[j]], [0, 0], [1, 1], [0, 0, 0] )
    
            modules.module[i].torquesLF[0, j+1] = tau[0, 0] * modules.module[i].D[0, j]
            modules.module[i].torquesLF[1, j+1] = tau[0, 1] * modules.module[i].D[1, j]
    
    
            modules.module[i].torquestot[0, j+1] = modules.module[i].torquesLF[0, j+1] # + modules.module[i].torqLWPR[0, j] + modules.module[i].Ctorques[0, j]
            modules.module[i].torquestot[1, j+1] = modules.module[i].torquesLF[1, j+1] # + modules.module[i].torqLWPR[1, j] + modules.module[i].Ctorques[1, j]
    
            modules.setModuleMotorsTorque(j,i)
            modules.getMotorsPos(j,i)
        
            
            modules.module[i].etp[:, j+1] = modules.module[i].q[:, j] - modules.module[i].pLWPR[:, j+1] 
            modules.module[i].etv[:, j+1] = modules.module[i].qd[:, j] - modules.module[i].vLWPR[:, j+1]
            
            modules.module[i].epr[:, j+1] = modules.module[i].q[:, j] - modules.module[i].posr[:, j+1] 
            modules.module[i].eprv[:, j+1] = modules.module[i].qd[:, j] - modules.module[i].velr[:, j+1] 
            print('Trajectory error:', modules.module[i].epr[:, j+1])
            # Prediction errors
            modules.module[i].epp[:, j+1] = modules.module[i].posr[:, j+1] - modules.module[i].pLWPR[:, j+1]
            modules.module[i].evv[:, j+1] = modules.module[i].velr[:, j+1] - modules.module[i].vLWPR[:, j+1]
            print('Prediction error:', modules.module[i].epp[:, j])
            if j>2:         
                meanerrorp[0] = np.mean(modules.module[i].etp[0, 2:j+1])
                meanerrorp[1] = np.mean(modules.module[i].etp[1, 2:j+1])
                meanerrorv[0] = np.mean(modules.module[i].etv[0, 2:j+1])
                meanerrorv[1] = np.mean(modules.module[i].etv[1, 2:j+1])
            
                # Normalizations
                modules.module[i].normetp[0, j+1] = (modules.module[i].etp[0, j+1] - np.min((modules.module[i].etp[0, 0:j+1])))/(np.max((modules.module[i].etp[0,0:j+1]))-np.min((modules.module[i].etp[0,0:j+1])))
                modules.module[i].normetp[1, j+1] = (modules.module[i].etp[1, j+1] - np.min((modules.module[i].etp[1, 0:j+1])))/(np.max((modules.module[i].etp[1,0:j+1]))-np.min((modules.module[i].etp[1,0:j+1])))
                modules.module[i].normetv[0, j+1] = (modules.module[i].etv[0, j+1] - np.min((modules.module[i].etv[0, 0:j+1])))/(np.max((modules.module[i].etv[0,0:j+1]))-np.min((modules.module[i].etv[0,0:j+1])))
                modules.module[i].normetv[1, j+1] = (modules.module[i].etv[1, j+1] - np.min((modules.module[i].etv[1, 0:j+1])))/(np.max((modules.module[i].etv[1,0:j+1]))-np.min((modules.module[i].etv[1,0:j+1])))
                
                #modules.module[i].normetp[0, j+1] = (modules.module[i].etp[0, j+1] / np.max(np.abs(modules.module[i].etp[0, 0:j+1])))#/2.0
                #modules.module[i].normetp[1, j+1] = (modules.module[i].etp[1, j+1] / np.max(np.abs(modules.module[i].etp[1, 0:j+1])))#/2.0
                #modules.module[i].normetv[0, j+1] = (modules.module[i].etv[0, j+1] / np.max(np.abs(modules.module[i].etv[0, 0:j+1])))#/2.0
                #modules.module[i].normetv[1, j+1] = (modules.module[i].etv[1, j+1] / np.max(np.abs(modules.module[i].etv[1, 0:j+1])))#/2.0
            '''input_data = []
            input_data.append(modules.module[i].torquesLF[0, j+1])                
            input_data.append(modules.module[i].torquesLF[1, j+1])
            input_data.append(modules.module[i].q1[j]) 
            input_data.append(modules.module[i].q2[j])
            input_data.append(modules.module[i].q1d[j]) 
            input_data.append(modules.module[i].q2d[j]) 
            input_data.append(modules.module[i].posr[0, j]) 
            input_data.append(modules.module[i].posr[1, j]) 
            input_data.append(modules.module[i].velr[0, j]) 
            input_data.append(modules.module[i].velr[1, j])             
            modules.input_lwpr = np.array(input_data)'''
         
            # Update models
            modules.mlcj[i].ML_update(modules.input_lwpr, np.array([
                                                      modules.module[i].posr[0, j+1],      modules.module[i].velr[0, j+1],
                                                      modules.module[i].posr[1, j+1],      modules.module[i].velr[1, j+1]
                                                      ])
                           )
            
            print("Current torquesLF:    ", modules.module[i].torquesLF[:, j+1])
            
            #print("C output: ", modules.module[i].outputC[:, j+1])
            
            modules.mlcj[i].ML_rfs()
            print("\n\n")
            t = (time.time() - begin_time)
            print("Modules_{} ".format(modules.ModuleID[i]), "- j: ", j, "- t: ", t)


    #####################################################################
    ####################### Save data into matlab #######################
    #####################################################################
    # Save with Matlab compatibility

    now = datetime.now()
    for i in range(len(modules.ModuleID)): 
        scipy.io.savemat('TestREC_1box_{0}modules_fab{1}_{2}_{3}_{4}.mat'.format(len(modules.ModuleID),
                                                                            modules.ModuleID[i], 
                                                                            now.strftime('%d-%m-%Y_%H:%M'), 
                                                                            modules.mlcj[i].model.num_rfs[0], 
                                                                            modules.module[i].init_D),
                     dict(q0 = modules.module[i].q1, q1 = modules.module[i].q2, q0d = modules.module[i].q1d, q1d = modules.module[i].q2d, 
                          velr0 = modules.module[i].velr[0], velr1 = modules.module[i].velr[1], posr0 = modules.module[i].posr[0], 
                          posr1 = modules.module[i].posr[1], errp0 = modules.module[i].epr[0], errp1 = modules.module[i].epr[1], errv0 = modules.module[i].eprv[0], errv1 = modules.module[i].eprv[1],
                          prederr0 = modules.module[i].epp[0], prederr1 = modules.module[i].epp[1], torquesLF0 = modules.module[i].
                          torquesLF[0], torquesLF1 = modules.module[i].torquesLF[1], DCNv0 = modules.module[i].outputDCN[2], DCNv1 = modules.module[i].outputDCN[3], 
                          DCNp0 = modules.module[i].outputDCN[0], DCNp1 = modules.module[i].outputDCN[1], vellwpr0 = modules.module[i].vLWPR[0], vellwpr1 = modules.module[i].vLWPR[1], 
                          poslwpr0 = modules.module[i].pLWPR[0], poslwpr1 = modules.module[i].pLWPR[1], weight_mod = modules.mlcj[i].weights_mod))
    
    print('Recurrent Thread-test for Module_{}: finishing'.format(modules.ModuleID))
    

#####################################################################
######################## Termination of test ########################
#####################################################################
# Termination of api and class usage

time.sleep(1)
print("Test - Done...")
time.sleep(1)
#SetUpModules.api.terminate()
