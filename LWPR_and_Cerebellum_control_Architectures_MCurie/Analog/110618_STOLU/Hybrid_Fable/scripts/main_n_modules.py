#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""============================================================
* File Name: main_n_modules.py
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
sys.path.append("/users/stolu//workspace/DTU_neurorobotics/lib/Robot_toolboxPy/robot")
sys.path.append("/users/stolu/workspace/DTU_neurorobotics/projects/lwpr_fable_forward_recurrent/scripts")

import SetUpModules
from fforward_n_modules import fforward
from recurrent_n_modules import recurrent

#####################################################################
###################### Variables initialization #####################
#####################################################################

platform = "mac"          # Select platform between "linux" or "mac"
all_modules_ids = [80]    # Select the list of modules ids[74, 80, ...]
n_iter = 20500                    # Select the number of iterations
control_type = 1                # Select the control architecture:
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
################################ Run test ################################
#####################################################################

if control_type == 0:
    fforward(modules, kp, ki, kv)
if control_type == 1:
    recurrent(modules, kp, ki, kv)
if control_type == 2:
    pass
    hybrid(modules, kp, ki, kv)
else:
    pass


#####################################################################
############################ Termination of test #############################
#####################################################################
# Termination of api and class usage

#modules.terminate()
time.sleep(1)
print("Test - Done...")

