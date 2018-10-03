#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""============================================================
* File Name: recurrent_n_modules.py
* Purpose: Recurrent control loop - Control several modules of Fable at the same
            time, using one single dongle and from just
            one script.Smith controller
* Creation Date : 05-02-2018
* Last Modified : 05-02-2018
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
sys.path.append("/users/stolu//workspace/DTU_neurorobotics/lib/Robot_toolboxPy/robot")
sys.path.append("/users/stolu//workspace/DTU_neurorobotics/projects/lwpr_fable_recurrent/scripts/recurrent_N_modules_newCerebellum4")

import dynamics
#import SetUpModules as SetUpModules
import SetUpModulesMultithreading as SetUpModules

#####################################################################
###################### Variables initialization #####################
#####################################################################
#exec()
platform = "mac"             # Select platform between "linux" or "mac"
#all_modules_ids = [74, 80]      # Select the list of modules ids[74, 80]
all_modules_ids = [80]      # Select the list of modules ids[74, 80]
n_iter = 20500               # Select the number of iterations
control_type = 1                 # Select the control architecture:
                                              # 0: fforward
                                              # 1: recurrent
                                              # 2: hybrid
eef_trajectory_type = 2   # Select the trajectory of the end effector:
                                              # 0: circle
                                              # 1: crawl
                                              # 2: eigth 
                                              # 3: eigth change Frequency
                                              # 4: eight change Amplitude

#modules = SetUpModules.MODULES(platform, all_modules_ids, n_iter, control_type, eef_trajectory_type)
modules = SetUpModules.MODULES_MULTITHREAD(platform, all_modules_ids, n_iter, control_type, eef_trajectory_type)

#####################################################################
############################# Run test ##############################
#####################################################################

modules.setModuleMotorsPos(0)
#modules.setModuleMotorMaxSpeed(100)
#modules.setModuleMotorMaxTorque(100)

time.sleep(1)

t = 0.0

for j in range(n_iter):
    
    start_time = time.time()
    
    # LWPR input data
    modules.createInputData_pos(j)
    modules.createInputData_vel(j)
    
    # LWPR prediction
    if j > 0:
        modules.predict(j)

    # PID + Cerebellum
    modules.performControl(j, t)

    modules.setModuleMotorsTorque(j)
    
    modules.getMotorsPos(j)
    
    modules.estimateErrors(j)
    
    modules.createInputData_pos(j+1)
    modules.createInputData_vel(j+1)
    # Update models
    if j > 0:
        modules.update(j)
        
    print("\n\n")
    t = (time.time() - start_time)
    print("Modules_{} ".format(all_modules_ids), "- j: ", j, "- t: ", t)


#####################################################################
####################### Save data into matlab #######################
#####################################################################
# Save with Matlab compatibility

now = datetime.now()
modules.save(now)
#for i in range(len(modules.module)): 
#    scipy.io.savemat('TestREC_1box_{0}modules_fab{1}_{2}_{3}_{4}.mat'.format(len(modules.module),
#                                                                        modules.module[i].ModuleID, 
#                                                                        now.strftime('%d-%m-%Y_%H:%M'), 
#                                                                        modules.module[i].mlcj.model.num_rfs[0], 
#                                                                        modules.module[i].module.init_D),
#                 dict(q0 = modules.module[i].module.q1, q1 = modules.module[i].module.q2, q0d = modules.module[i].module.q1d, q1d = modules.module[i].module.q2d, 
#                      velr0 = modules.module[i].module.velr[0], velr1 = modules.module[i].module.velr[1], posr0 = modules.module[i].module.posr[0], 
#                      posr1 = modules.module[i].module.posr[1], errp0 = modules.module[i].module.epr[0], errp1 = modules.module[i].module.epr[1], errv0 = modules.module[i].module.evr[0], errv1 = modules.module[i].module.evr[1],
#                      prederr0 = modules.module[i].module.epp[0], prederr1 = modules.module[i].module.epp[1], torquesLF0 = modules.module[i].module.
#                      torquesLF[0], torquesLF1 = modules.module[i].module.torquesLF[1], DCNv0 = modules.module[i].module.outputDCN[2], DCNv1 = modules.module[i].module.outputDCN[3], 
#                      DCNp0 = modules.module[i].module.outputDCN[0], DCNp1 = modules.module[i].module.outputDCN[1], vellwpr0 = modules.module[i].module.vLWPR[0], vellwpr1 = modules.module[i].module.vLWPR[1], 
#                      poslwpr0 = modules.module[i].module.pLWPR[0], poslwpr1 = modules.module[i].module.pLWPR[1], weight_mod = modules.module[i].mlcj.weights_mod))
#
#    print('Recurrent Thread-test for Module_{}: finishing'.format(modules.module[i].ModuleID))


#####################################################################
######################## Termination of test ########################
#####################################################################
# Termination of api and class usage

del modules
time.sleep(1)
print("Test - Done...")
time.sleep(1)
sys.exit()
