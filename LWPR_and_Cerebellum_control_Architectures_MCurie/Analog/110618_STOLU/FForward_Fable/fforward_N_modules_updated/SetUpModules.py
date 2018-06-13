#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""============================================================
* File Name: Recurrent_modular.py
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

sys.path.append("/users/stolu/workspace/DTU_neurorobotics/lib/fable_1_api/python/api")
sys.path.append("/users/stolu/workspace/DTU_neurorobotics/lib/Robot_toolboxPy/robot")
sys.path.append("/users/stolu/workspace/DTU_neurorobotics/projects/lwpr_fable_recurrent/scripts/recurrent_N_modules")
sys.path.append("/users/stolu/workspace/DTU_neurorobotics/projects/lwpr_fable_fforward/scripts/fforward_N_modules")

import lwpr
import RafelLwprAndC1boxLwpr as rafel_class
import AfelLwprAndC as afel_class
import Robot
import fableAPI as api
import simfable

import dynamics
from datetime import datetime, date
import serial, time, random, math#, threading
#from threading import Thread

'''def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper'''

class MODULES():
    def __init__(self, platform, modulesids, n_iter, control_type, eef_trajectory_type):
        self.n_iter = n_iter
        self.njoints = 2
        self.nout = 4
        self.grav    = [0, 0, 9.81]
        self.fab     = simfable.moduleFable()
        self.fab17   = self.fab.Fab()
        self.ModuleID = modulesids
        self.module = []
        self.control_type = control_type
        self.eef_trajectory_type = eef_trajectory_type
        self.mlj = []
        self.api = api.FableAPI()
        
        
        if "linux" in platform.lower():
            self.api.setup(0)
        elif "mac" in platform.lower():
            self.api.setup(1)
                
        self.module = [MODULE_INI(self.n_iter, self.control_type) for x in range(len(self.ModuleID))]
        
        for i in range(len(self.ModuleID)):
            # Selection of end effector trajectory
            if self.eef_trajectory_type == 0:
                self.module[i].calc_trajectory_circle(self.fab17)
            elif self.eef_trajectory_type == 1:
                self.module[i].calc_trajectory_crawl(self.fab17)
            elif self.eef_trajectory_type == 2:
                self.module[i].calc_trajectory_8(self.fab17)
            else:
                return
            
                            
        # Selection of control architecture
        # FFORWARD
        if self.control_type == 0:
            self.N_LWPR_INPUTS_1MODULE = 6
            self.n_lwpr_inputs = len(self.ModuleID)*self.N_LWPR_INPUTS_1MODULE
            self.mlcj = [afel_class.MLandC(self.n_lwpr_inputs, self.njoints) for x in range(len(self.ModuleID))]
        # RECURRENT
        elif self.control_type == 1:
            self.N_LWPR_INPUTS_1MODULE = 10
            self.n_lwpr_inputs = len(self.ModuleID)*self.N_LWPR_INPUTS_1MODULE
            self.mlcj = [rafel_class.MLandC(self.n_lwpr_inputs, self.nout, self.njoints) for x in range(len(self.ModuleID))]
        # HYBRYD
        elif self.control_type == 2:
            #self.mlcj = [RAFEL.MLandC(self.n_lwpr_inputs, self.nout, self.njoints) for x in range(len(modulesids))]
            pass
        else:
            return

            
    def getMotorsPos(self, j):
        for i in range(len(self.ModuleID)):
            # Receive feedback positions from motors
            self.module[i].posr[0, j+1] =  self.api.getModuleMotorPosition(self.ModuleID[i], 0)   # degrees
            self.module[i].posr[1, j+1] =  self.api.getModuleMotorPosition(self.ModuleID[i], 1)
            self.module[i].velr[0, j+1] = -self.api.getModuleMotorSpeed(self.ModuleID[i], 0)      # grad/s
            self.module[i].velr[1, j+1] = -self.api.getModuleMotorSpeed(self.ModuleID[i], 1)
     
    def setModuleMotorsPos(self, j):
        for i in range(len(self.ModuleID)):
            self.api.setModuleMotorPosition(self.ModuleID[i], 0, self.module[i].q1[j])
            self.api.setModuleMotorPosition(self.ModuleID[i], 1, self.module[i].q2[j])
    
    def setModuleMotorsTorque(self, j):
        for i in range(len(self.ModuleID)):
            # Control in motor torques
            # FFORWARD
            if self.control_type == 0:
                self.api.setModuleMotorTorque(self.ModuleID[i], 0, -self.module[i].torquestot[0, j], 1, -self.module[i].torquestot[1, j], ack = False)
            # RECURRENT
            elif self.control_type == 1:
                self.api.setModuleMotorTorque(self.ModuleID[i], 0, -self.module[i].torquestot[0, j], 1, -self.module[i].torquestot[1, j], ack = False)
            # HYBRID
            elif self.control_type == 2:
                pass
            else:
                return
            #self.api.setModuleMotorTorque(self.ModuleID[i], 0, -self.torquesLF[0, j+1], 1, -self.torquesLF[1, j+1], ack = False)
    def __del__(self):
            print("Class object for Module_{} - Destroyed".format(self.ModuleID))

    def terminate(self):
        self.api.terminate()
        
class MODULE_INI():

    def __init__(self, n_iter,  control_type):
        self.n_iter = n_iter
        self.njoints = 2
        
        # In case of tuning of the LF
        # self.a = 1.0
        # self.b = 7.5
        # self.c = 6.4 #6.4
        # self.ki = [self.a, self.a]
        # self.kp = [self.b, self.b]
        # self.kv = [self.c, self.c]
        
        #constant c1,c2,c3= 1 for vertical down - constant c1= 2 c2,c3=2.5for vertical up position
#        c1 = 1 #2.0 
#        c2 = 1 #2.5
#        c3 = 1 #2.5
        
#        self.ki = [1.0*c1, 1.0*c1]
#        self.kp = [7.5*c2, 7.5*c2]
#        self.kv = [6.4*c3, 6.4*c3]
        self.grav = [0, 0, 9.81]

        # Variable definitions
        self.A1 = 700
        self.A2 = 700
        self.phase = math.pi / 2            # CRAWL math.pi / 2
        self.dt    = 0.01                 # CRAWL 0.03
        self.t0    = 0
 
        self.q1   = [0 for k in range(self.n_iter + 1)]
        self.q2   = [0 for a in range(self.n_iter + 1)]
        self.q = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.q1d  = [0 for c in range(self.n_iter + 1)]
        self.q2d  = [0 for v in range(self.n_iter + 1)]
        self.qd = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.q1dd = [0 for o in range(self.n_iter + 1)]
        self.q2dd = [0 for l in range(self.n_iter + 1)]
        self.x    = [0 for k in range(self.n_iter + 1)]
        self.y    = [0 for k in range(self.n_iter + 1)]
        num_rfs = 0
        self.posr = np.zeros((self.njoints, self.n_iter + 1), dtype = np.double)
        self.velr = np.zeros((self.njoints, self.n_iter + 1), dtype = np.double)

        self.input_lwpr = []

        self.accr = np.zeros((self.njoints, self.n_iter + 1), dtype = np.double)
        self.D    = np.zeros((self.njoints, self.n_iter + 1), dtype = np.double)
        self.erra = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.errp = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.errv = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.epr = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.eprv = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.etp = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.etv = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.ea   = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.ep   = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.ev   = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        
        self.torquesLF   = np.zeros((self.njoints, self.n_iter + 1), dtype = np.double)
	
        if control_type == 0:
            self.nout = 2
            
        elif control_type == 1:
            self.nout = 4
            
        elif control_type == 2:
            pass
        else:
            return

        self.Ctorques    = np.zeros((self.nout, self.n_iter + 1), dtype = np.double)
        self.outputDCN   = np.zeros((self.nout, self.n_iter + 1), dtype = np.double)
        self.torqLWPR    = np.zeros((self.nout, self.n_iter + 1), dtype = np.double)
        self.output_x = np.zeros((self.nout), dtype = np.double)
        self.outputC  = np.zeros((self.nout, self.n_iter + 1), dtype = np.double)
        self.weights_mod = np.zeros((self.njoints, self.n_iter + 1), dtype = np.double)
        
        #self.DCNv        = np.zeros((self.nout, self.n_iter + 1), dtype = np.double)
        self.torquestot  = np.zeros((self.njoints, self.n_iter + 1), dtype = np.double)

        self.posC = np.zeros((self.njoints, n_iter + 1), dtype=np.double)
        self.velC = np.zeros((self.njoints, n_iter + 1), dtype=np.double)
        self.pLWPR = np.zeros((self.njoints, self.n_iter + 1), dtype = np.double)
        self.vLWPR = np.zeros((self.njoints, self.n_iter + 1), dtype = np.double)

    # Circular trajectory
    def calc_trajectory_circle(self, fab):
        self.phase = math.pi / 2
        for i in range(self.n_iter):
            self.q1dd[i] = self.A1 * math.sin(2 * math.pi * self.t0)
            self.q1d[i]  = -1/(2*math.pi) * self.A1 * math.cos(2 * math.pi * self.t0)
            self.q1[i]   = -math.pow((1/(2*math.pi)),2) * self.A1 * math.sin(2 * math.pi * self.t0)

            self.q2dd[i] = self.A2 * math.sin(2 * math.pi * self.t0 + self.phase)
            self.q2d[i]  = -1/(2*math.pi) * self.A2 * math.cos(2 * math.pi * self.t0 + self.phase)
            self.q2[i]   = -math.pow((1/(2*math.pi)),2) * self.A2 * math.sin(2 * math.pi * self.t0 + self.phase)

            self.t0 += self.dt
            # Cartesian coordinates ...
            #self.x[i] = fab.l1 * math.cos(self.q1[i]) + fab.l2 * math.cos((self.q1[i] + self.q2[i]))
            #self.y[i] = fab.l1 * math.sin(self.q1[i]) + fab.l2 * math.sin((self.q1[i] + self.q2[i]))

        return self.q1, self.q2, self.q1d, self.q2d, self.q1dd, self.q2dd#, self.x, self.y

    # Circular trajectory
    def calc_trajectory_crawl(self, fab):
        self.phase = math.pi / 3                                 # CRAWL math.pi / 3
        for i in range(self.n_iter):
            self.q1dd[i] = -self.A1 * math.sin(2 * math.pi * self.t0)
            self.q1d[i]  = 1/(2*math.pi) * self.A1 * math.cos(2 * math.pi * self.t0)
            self.q1[i]   = math.pow((1/(2*math.pi)),2) * self.A1 * math.sin(2 * math.pi * self.t0)

            self.q2dd[i] = -self.A2 * math.sin(2 * math.pi * self.t0 + self.phase)
            self.q2d[i]  = 1/(2*math.pi) * self.A2 * math.cos(2 * math.pi * self.t0 + self.phase)
            self.q2[i]   = math.pow((1/(2*math.pi)),2) * self.A2 * math.sin(2 * math.pi * self.t0 + self.phase)

            self.t0 += self.dt

        return self.q1, self.q2, self.q1d, self.q2d, self.q1dd, self.q2dd#, self.x, self.y

    # Eight figure trajectory
    def calc_trajectory_8(self, fab):
        self.A1    = 7
        self.phase = math.pi / 2
        for i in range(self.n_iter):
            self.q1dd[i] = self.A1 * math.sin(2 * math.pi * self.t0)
            self.q1d[i]  = ((-1 / 2) * math.pi) * self.A1 * math.cos(2 * math.pi * self.t0)
            self.q1[i]   = (-math.pow(((1 / 2) * math.pi), 2)) * self.A1 * math.sin(2 * math.pi * self.t0)

            self.q2dd[i]  = self.A1 * math.cos(4 * math.pi * self.t0 + math.pi / 2)
            self.q2d[i]   = ((1 / 2) * math.pi)  * self.A1 * math.sin(4 * math.pi * self.t0 + math.pi / 2)
            self.q2[i]    = (-math.pow(((1 / 2) * math.pi), 2)) * self.A1 * math.cos(4 * math.pi * self.t0 + math.pi / 2)
            self.q[:, i]  = (self.q1[i], self.q2[i])
            self.qd[:, i] = (self.q1d[i], self.q2d[i])
            self.t0 += self.dt
                # Cartesian coordinates
        plt.plot(self.q1, self.q2)
        #plt.show()
        return self.q1, self.q2, self.q1d, self.q2d, self.q1dd, self.q2dd
