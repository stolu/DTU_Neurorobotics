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
sys.path.append("/users/stolu/workspace/DTU_neurorobotics/projects/lwpr_fable_fforward/scripts/fforward_N_modules")

import dynamics

#####################################################################
############################### Run test ################################ 
#####################################################################

def hybrid(modules, kp, ki, kv):
        modules.setModuleMotorsPos(0)
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
              
              # LWPR predictions
              (modules.module[i].output_lwpr, modules.module[i].outputDCN[:, j], modules.module[i].init_D) = modules.mlcj[i].ML_prediction(
                                                                                                              modules.input_lwpr,
                                                                                                              np.round(modules.module[i].normetp[:, j],2),
                                                                                                              np.round(modules.module[i].normetv[:, j],2))
              
              modules.module[i].pLWPR[0, j] = np.round(modules.module[i].output_lwpr[0],0)
              modules.module[i].vLWPR[0, j] = np.round(modules.module[i].output_lwpr[1],0)
              modules.module[i].pLWPR[1, j] = np.round(modules.module[i].output_lwpr[2],0)
              modules.module[i].vLWPR[1, j] = np.round(modules.module[i].output_lwpr[3],0)
              modules.module[i].outputDCN[0:2, j] = np.round(modules.module[i].outputDCN[0:2, j], 2)
              modules.module[i].outputDCN[2:4, j] = np.round(modules.module[i].outputDCN[2:4, j], 2)
              print("DCN output: ", modules.module[i].outputDCN[:, j])
              # Prediction errors
              '''modules.module[i].epr[:, j+1]  = (modules.module[i].posr[:, j] - modules.module[i].pLWPR[:, j])
              modules.module[i].eprv[:, j+1] = (modules.module[i].velr[:, j] - modules.module[i].vLWPR[:, j])'''
              
              # errors
              modules.module[i].epd[:, j]  = modules.module[i].q[:,j] - modules.module[i].pLWPR[:, j] + modules.module[i].etp[:, j] - modules.module[i].epr[:, j] #- (modules.module[i].posr[:, j] - modules.module[i].pLWPR[:, j])
              modules.module[i].epdv[:, j] = modules.module[i].qd[:,j] - modules.module[i].vLWPR[:, j] + modules.module[i].etv[:, j] - modules.module[i].eprv[:, j] #- (modules.module[i].velr[:, j] - modules.module[i].vLWPR[:, j])
              #modules.module[i].epd[:, j]  = (modules.module[i].outputDCN[0:2, j]) - modules.module[i].epr[:, j] + modules.module[i].etp[:, j]#- (modules.module[i].posr[:, j] - modules.module[i].pLWPR[:, j])
              #modules.module[i].epdv[:, j] = (modules.module[i].outputDCN[2:4, j]) - modules.module[i].eprv[:, j] + modules.module[i].etv[:, j]#- (modules.module[i].velr[:, j] - modules.module[i].vLWPR[:, j])
              # LF controller 
              
              # Set right corrections (sign)
              if modules.module[i].epd[0, j] < 0:
                  modules.module[i].outputDCN[0, j] = - modules.module[i].outputDCN[0, j]
              
              if modules.module[i].epdv[0, j] < 0:
                  modules.module[i].outputDCN[2, j] = - modules.module[i].outputDCN[2, j]
              
              if modules.module[i].epd[1, j] < 0:
                  modules.module[i].outputDCN[1, j] = - modules.module[i].outputDCN[1, j]
              
              if modules.module[i].epdv[1, j] < 0:
                  modules.module[i].outputDCN[3, j] = - modules.module[i].outputDCN[3, j]
                  
              modules.module[i].ep[:, j+1] = modules.module[i].epd[:, j] + modules.module[i].outputDCN[0:2, j]
              modules.module[i].ev[:, j+1] = ((modules.module[i].ep[:, j+1] - modules.module[i].ep[:, j]) / modules.module[0].dt + modules.module[i].outputDCN[2:4, j])
              modules.module[i].ea[:, j+1] = 0 #0.22
              
              # Feedback error learning
              if j > 0:
                  modules.module[i].D[:, j] = modules.module[i].D[:, j - 1] + (modules.module[i].ea[:, j+1] * ki[i]) + (modules.module[i].ep[:, j+1] * kp[i]) + (modules.module[i].ev[:, j+1] * kv[i])
              else:
                  modules.module[i].D[:, j] = (modules.module[i].ea[:, j+1] * ki[i]) + (modules.module[i].ep[:, j+1] * kp[i]) + (modules.module[i].ev[:, j+1] * kv[i])

              tau = dynamics.rne(modules.fab17, [modules.module[i].q1[j], modules.module[i].q2[j]], [0, 0], [1, 1], [0, 0, 0] )

              modules.module[i].torquesLF[0, j+1] = tau[0, 0] * modules.module[i].D[0, j]
              modules.module[i].torquesLF[1, j+1] = tau[0, 1] * modules.module[i].D[1, j]

              modules.module[i].torquestot[0, j] = modules.module[i].torquesLF[0, j+1]
              modules.module[i].torquestot[1, j] = modules.module[i].torquesLF[1, j+1]
              
              # Avoid torques higher than 100
              if modules.module[i].torquestot[0, j] > 100.0:
                  modules.module[i].torquestot[0, j] = 40.0
              if modules.module[i].torquestot[1, j] > 100.0:
                  modules.module[i].torquestot[1, j] = 40.0
              # Avoid torques smaller than -100
              if modules.module[i].torquestot[0, j] < -100.0:
                  modules.module[i].torquestot[0, j] = -40.0
              if modules.module[i].torquestot[1, j] < -100.0:
                  modules.module[i].torquestot[1, j] = -40.0
              
              #Set torques
              modules.setModuleMotorsTorque(j)
              print("Torques Tot:    ", np.round(modules.module[i].torquestot[:, j],2))
              
              #Get motors positions
              modules.getMotorsPosVel(j)
              print("desired position:", modules.module[i].q[:,j])
              print("Current Positions:    ", modules.module[i].posr[:, j+1])
              print("lwpr output - positions: ", modules.module[i].pLWPR[0:2, j])
              print("Current Velocities:    ", modules.module[i].velr[:, j+1])
              print("lwpr output - velocities: ", modules.module[i].vLWPR[0:2, j])
              
              #LWPR input data for update model
              input_data = []
              input_data.append(modules.module[i].torquesLF[0, j+1])                
              input_data.append(modules.module[i].torquesLF[1, j+1])
              input_data.append(modules.module[i].q1[j]) 
              input_data.append(modules.module[i].q2[j])
              input_data.append(modules.module[i].q1d[j]) 
              input_data.append(modules.module[i].q2d[j]) 
              input_data.append(modules.module[i].posr[0, j+1]) 
              input_data.append(modules.module[i].posr[1, j+1]) 
              input_data.append(modules.module[i].velr[0, j+1]) 
              input_data.append(modules.module[i].velr[1, j+1])
                              
              modules.input_lwpr = np.array(input_data)
              
              # LWPR Update models
              modules.mlcj[i].ML_update(modules.input_lwpr, np.array([
                                                        modules.module[i].posr[0, j+1],      modules.module[i].velr[0, j+1],
                                                        modules.module[i].posr[1, j+1],      modules.module[i].velr[1, j+1]
                                                        ])
                             )
              #Trajectory errors
              modules.module[i].etp[:, j+1] = modules.module[i].q[:, j] - modules.module[i].pLWPR[:, j]  #modules.module[i].posr[:, j+1] 
              modules.module[i].etv[:, j+1] = modules.module[i].qd[:, j] - modules.module[i].vLWPR[:, j] #modules.module[i].velr[:, j+1]
              print('etp:', modules.module[i].etp[:, j])
              
              #prediction errors
              modules.module[i].epr[:, j+1]  = (modules.module[i].posr[:, j+1] - modules.module[i].pLWPR[:, j])
              modules.module[i].eprv[:, j+1] = (modules.module[i].velr[:, j+1] - modules.module[i].vLWPR[:, j])
              if j>0:
                  # Normalizations
                  modules.module[i].normetp[0, j+1] = (modules.module[i].etp[0, j+1] - modules.module[i].etp[0].min())/(modules.module[i].etp[0].max()-modules.module[i].etp[0].min())
                  modules.module[i].normetp[1, j+1] = (modules.module[i].etp[1, j+1] - modules.module[i].etp[1].min())/(modules.module[i].etp[1].max()-modules.module[i].etp[1].min())
                  modules.module[i].normetv[0, j+1] = (modules.module[i].etv[0, j+1] - modules.module[i].etv[0].min())/(modules.module[i].etv[0].max()-modules.module[i].etv[0].min())
                  modules.module[i].normetv[1, j+1] = (modules.module[i].etv[1, j+1] - modules.module[i].etv[1].min())/(modules.module[i].etv[1].max()-modules.module[i].etv[1].min())
              print('normetp', np.round(modules.module[i].normetp[:, j+1],2) )
              print('normetv', np.round(modules.module[i].normetv[:, j+1],2) )
          
              modules.module[i].num_rfs = modules.mlcj[i].ML_rfs()
              end_time = time.time()
              print("\n\n")
              t = (end_time - begin_time)
              print("Modules_{} ".format(modules.ModuleID[i]), "- j: ", j, "- t: ", np.round(t,2))

              #####################################################################
              ####################### Save data into matlab #######################
              #####################################################################
              # Save with Matlab compatibility

        now = datetime.now()
        for i in range(len(modules.ModuleID)): 
                scipy.io.savemat('TestREC_1box_{0}modules_fab{1}_{2}_{3}_{4}.mat'.format(len(modules.ModuleID), modules.ModuleID[i], 
                                                                                   now.strftime('%d-%m-%Y_%H:%M'), 
                                                                                   modules.mlcj[i].model.num_rfs[0], 
                                                                                   modules.mlcj[i].model.init_D[0][0]),
                                 dict(q0 = modules.module[i].q1, q1 = modules.module[i].q2, q0d = modules.module[i].q1d, q1d = modules.module[i].q2d, 
                                      velr0 = modules.module[i].velr[0], velr1 = modules.module[i].velr[1], posr0 = modules.module[i].posr[0], 
                                      posr1 = modules.module[i].posr[1], errp0 = modules.module[i].etp[0], errp1 = modules.module[i].etp[1], 
                                      prederr0 = modules.module[i].epr[0], prederr1 = modules.module[i].epr[1], torquesLF0 = modules.module[i].torquesLF[0], 
                                      torquesLF1 = modules.module[i].torquesLF[1], DCNv0 = modules.module[i].outputDCN[2], DCNv1 = modules.module[i].outputDCN[3], 
                                      DCNp0 = modules.module[i].outputDCN[0], DCNp1 = modules.module[i].outputDCN[1],
                                      vellwpr0 = modules.module[i].vLWPR[0], vellwpr1 = modules.module[i].vLWPR[1], poslwpr0 = modules.module[i].pLWPR[0], 
                                      poslwpr1 = modules.module[i].pLWPR[1], weight_mod = modules.mlcj[i].weights_mod))


        
        return True

