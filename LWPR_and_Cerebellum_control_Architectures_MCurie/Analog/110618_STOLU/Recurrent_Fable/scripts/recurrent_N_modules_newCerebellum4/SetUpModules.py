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

import sys, time, math, os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from _functools import partial

sys.path.append("/users/stolu/workspace/DTU_neurorobotics/lib/fable_1_api/python/api")
sys.path.append("/users/stolu/workspace/DTU_neurorobotics/lib/Robot_toolboxPy/robot")
sys.path.append("/users/stolu/workspace/DTU_neurorobotics/projects/lwpr_fable_recurrent/scripts/recurrent_N_modules_newCerebellum3")
#sys.path.append("/users/stolu/workspace/DTU_neurorobotics/projects/lwpr_fable_fforward/scripts/fforward_N_modules")

import lwpr
import RafelLwprAndC1boxLwpr as rafel_class
#import AfelLwprAndC as afel_class
import Robot
import fableAPI as api
import simfable

from multiprocessing import Pool, cpu_count
import functools
import multiprocessing
from threading import Thread
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
            elif self.eef_trajectory_type == 3:
                self.module[i].calc_trajectory_8_changefreq(self.fab17)
            elif self.eef_trajectory_type == 4:
                self.module[i].calc_trajectory_8_changeA(self.fab17)
            else:
                return
            
                            
        # Selection of control architecture
        # FFORWARD
        if self.control_type == 0:
            self.N_LWPR_INPUTS_1MODULE = 6
            #self.n_lwpr_inputs = len(self.ModuleID)*self.N_LWPR_INPUTS_1MODULE
            self.n_lwpr_inputs = self.N_LWPR_INPUTS_1MODULE#
            self.mlcj = [afel_class.MLandC(self.n_lwpr_inputs, self.njoints) for x in range(len(self.ModuleID))]
        # RECURRENT
        elif self.control_type == 1:
            self.N_LWPR_INPUTS_1MODULE = 10
            #self.n_lwpr_inputs = len(self.ModuleID)*self.N_LWPR_INPUTS_1MODULE
            self.n_lwpr_inputs = self.N_LWPR_INPUTS_1MODULE#len(self.ModuleID)*self.N_LWPR_INPUTS_1MODULE
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
            self.module[i].velr[0, j+1] = -self.api.getModuleMotorSpeed(self.ModuleID[i], 0)    # grad/s
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
                self.api.setModuleMotorTorque(self.ModuleID[i], 0, -self.module[i].torquestot[0, j+1], 1, -self.module[i].torquestot[1, j+1], ack = False)
            # HYBRID
            elif self.control_type == 2:
                pass
            else:
                return
        #self.api.setModuleMotorTorque(self.ModuleID[i], 0, -self.torquesLF[0, j+1], 1, -self.torquesLF[1, j+1], ack = False)
    def __del__(self):
            print("Class object for Module_{} - Destroyed".format(self.ModuleID))

#class MODULE(multiprocessing.Process):
class MODULE():
    def __init__(self, platform, moduleID, n_iter, control_type, eef_trajectory_type, port = 0):
        # must call this before anything else
#        multiprocessing.Process.__init__(self)

        #constant c1,c2,c3=1 for vertical down - constant c1= 2 c2,c3=2.5for vertical up position
        c1 = 1 #2.0 
        c2 = 1 #2.5
        c3 = 1 #2.5
        
        self.kp = 7.5*c2       # Select the kp pf the Learning Feedback controller
        self.ki = 1.0*c1       # Select the ki pf the Learning Feedback controller
        self.kv = 6.4*c3       # Select the kv pf the Learning Feedback controller
        self.meanerrorp = [0.0, 0.0]
        self.meanerrorv = [0.0, 0.0]
        self.n_iter = n_iter
        self.njoints = 2
        self.nout = 4
        self.grav    = [0, 0, 9.81]
        self.fab     = simfable.moduleFable()
        self.fab17   = self.fab.Fab()
        self.ModuleID = moduleID
        #self.module = []
        self.control_type = control_type
        self.eef_trajectory_type = eef_trajectory_type
        self.mlj = []
        self.api = api.FableAPI()
        self.moduleIsDone = False
            
        #if "linux" in platform.lower():
        #    self.api.setup(0)
        #elif "mac" in platform.lower():
        #    self.api.setup(1)
        if "linux" in platform.lower():
            self.api.setup(port)
        elif "mac" in platform.lower():
            self.api.setup(port+1)
        
        self.module = MODULE_INI(self.n_iter, self.control_type)
        
        # Selection of end effector trajectory
        if self.eef_trajectory_type == 0:
            self.module.calc_trajectory_circle(self.fab17)
        elif self.eef_trajectory_type == 1:
            self.module.calc_trajectory_crawl(self.fab17)
        elif self.eef_trajectory_type == 2:
            self.module.calc_trajectory_8(self.fab17)
        elif self.eef_trajectory_type == 3:
            self.module.calc_trajectory_8_changefreq(self.fab17)
        elif self.eef_trajectory_type == 4:
            self.module.calc_trajectory_8_changeA(self.fab17)
        else:
            return
            
                            
        # Selection of control architecture
        # FFORWARD
        if self.control_type == 0:
            self.N_LWPR_INPUTS_1MODULE = 6
            #self.n_lwpr_inputs = len(self.ModuleID)*self.N_LWPR_INPUTS_1MODULE
            self.n_lwpr_inputs = self.N_LWPR_INPUTS_1MODULE#
            self.mlcj = afel_class.MLandC(self.n_lwpr_inputs, self.njoints)
        # RECURRENT
        elif self.control_type == 1:
            self.N_LWPR_INPUTS_1MODULE = 10
            #self.n_lwpr_inputs = len(self.ModuleID)*self.N_LWPR_INPUTS_1MODULE
            self.n_lwpr_inputs = self.N_LWPR_INPUTS_1MODULE#len(self.ModuleID)*self.N_LWPR_INPUTS_1MODULE
            self.mlcj = rafel_class.MLandC(self.n_lwpr_inputs, self.nout, self.njoints)
        # HYBRYD
        elif self.control_type == 2:
            #self.mlcj = [RAFEL.MLandC(self.n_lwpr_inputs, self.nout, self.njoints) for x in range(len(modulesids))]
            pass
        else:
            return
        
        return
            
    def getMotorsPos(self, j):
        # Flag to inform the process is not finished
        self.moduleIsDone = False
        
        # Receive feedback positions from motors
        self.module.posr[0, j+1] =  self.api.getModuleMotorPosition(self.ModuleID, 0)   # degrees
        self.module.posr[1, j+1] =  self.api.getModuleMotorPosition(self.ModuleID, 1)
        self.module.velr[0, j+1] = -self.api.getModuleMotorSpeed(self.ModuleID, 0)    # grad/s
        self.module.velr[1, j+1] = -self.api.getModuleMotorSpeed(self.ModuleID, 1)
        
        # Flag to inform the process is finished
        self.moduleIsDone = True
    
    def setModuleMotorsPos(self, j):
        print('setModuleMotorPos for module ', self.ModuleID)
        # Flag to inform the process is not finished
        self.moduleIsDone = False
        
        # Set position
        self.api.setModuleMotorPosition(self.ModuleID, 0, self.module.q1[j])
        self.api.setModuleMotorPosition(self.ModuleID, 1, self.module.q2[j])
        
        # Flag to inform the process is finished
        self.moduleIsDone = True
        
    def setModuleMotorsTorque(self, j):
        # Flag to inform the process is not finished
        self.moduleIsDone = False

        # Control in motor torques
        # FFORWARD
        if self.control_type == 0:
            self.api.setModuleMotorTorque(self.ModuleID, 0, -self.module.torquestot[0, j], 1, -self.module.torquestot[1, j], ack = False)
            # Flag to inform the process is finished
            self.moduleIsDone = True
        # RECURRENT
        elif self.control_type == 1:
            self.api.setModuleMotorTorque(self.ModuleID, 0, -self.module.torquestot[0, j+1], 1, -self.module.torquestot[1, j+1], ack = False)
            # Flag to inform the process is finished
            self.moduleIsDone = True
        # HYBRID
        elif self.control_type == 2:
            # Flag to inform the process is finished
            self.moduleIsDone = True
            pass
        else:
            # Flag to inform the process is finished
            self.moduleIsDone = True
            return
        #self.api.setModuleMotorTorque(self.ModuleID[i], 0, -self.torquesLF[0, j+1], 1, -self.torquesLF[1, j+1], ack = False)
    
    def createInputData(self, index):
        input_data = []
        input_data.append(self.module.torquesLF[0, index])                
        input_data.append(self.module.torquesLF[1, index])
        input_data.append(self.module.q1[index]) 
        input_data.append(self.module.q2[index])
        input_data.append(self.module.q1d[index]) 
        input_data.append(self.module.q2d[index]) 
        input_data.append(self.module.posr[0, index]) 
        input_data.append(self.module.posr[1, index]) 
        input_data.append(self.module.velr[0, index]) 
        input_data.append(self.module.velr[1, index])
        '''elif(len(modules.ModuleID) == 2):
        Due the purpose of this part of the code, it can not be parallelised... 
        it needs to be implemented in a different function within the class MODULES_MULTITHREAD
        input_data = []
        input_data.append(modules.module[0].torquesLF[0, index])      
        input_data.append(modules.module[0].torquesLF[1, index])
        input_data.append(modules.module[0].q1[index])
        input_data.append(modules.module[0].q2[index])
        input_data.append(modules.module[0].q1d[index])
        input_data.append(modules.module[0].q2d[index])
        input_data.append(modules.module[0].posr[0, index])
        input_data.append(modules.module[0].posr[1, index]) 
        input_data.append(modules.module[0].velr[0, index]) 
        input_data.append(modules.module[0].velr[1, index])
        input_data.append(modules.module[1].torquesLF[0, index])    
        input_data.append(modules.module[1].torquesLF[1, index])
        input_data.append(modules.module[1].q1[index])
        input_data.append(modules.module[1].q2[index])
        input_data.append(modules.module[1].q1d[index])
        input_data.append(modules.module[1].q2d[index]) 
        input_data.append(modules.module[1].posr[0, index])
        input_data.append(modules.module[1].posr[1, index])
        input_data.append(modules.module[1].velr[0, index])
        input_data.append(modules.module[1].velr[1, index])'''
    
        self.input_lwpr = np.array(input_data)
    
    def predict(self, index):
        (self.module.output_x, self.module.outputC[:, index+1], 
         self.module.outputDCN[:, index+1], self.module.init_D) = self.mlcj.ML_prediction(  self.input_lwpr,
                                                                                            self.module.epr[:, index],
                                                                                            self.module.evr[:, index], self.meanerrorp, 
                                                                                            self.meanerrorv, 
                                                                                            self.module.normetp[:, index], 
                                                                                            self.module.normetv[:, index])
        self.module.pLWPR[0, index+1] = self.module.output_x[0]
        self.module.vLWPR[0, index+1] = self.module.output_x[1]
        self.module.pLWPR[1, index+1] = self.module.output_x[2]
        self.module.vLWPR[1, index+1] = self.module.output_x[3]
        
        #print("lwpr output: ", self.module.pLWPR[0:2, index+1])
        
        '''if self.module.etp[0, index] < 0:
            self.module.outputDCN[0, index+1] = - self.module.outputDCN[0, index+1]
              
        if self.module.etv[0, index] < 0:
            self.module.outputDCN[2, index+1] = - self.module.outputDCN[2, index+1]
              
        if self.module.etp[1, index] < 0:
            self.module.outputDCN[1, index+1] = - self.module.outputDCN[1, index+1]
              
        if self.module.etv[1, index] < 0:
            self.module.outputDCN[3, index+1] = - self.module.outputDCN[3, index+1]'''
        print("DCN output: ", self.module.outputDCN[:, index+1])
        #print("C output: ", self.module.output_C[:, index+1])
    
    def estimateErrors(self, index):
        self.module.etp[:, index] = self.module.q[:, index-1] - self.module.pLWPR[:, index] 
        self.module.etv[:, index] = self.module.qd[:, index-1] - self.module.vLWPR[:, index]
        
        # Prediction errors
        self.module.epp[:, index] = self.module.posr[:, index] - self.module.pLWPR[:, index]
        self.module.evv[:, index] = self.module.velr[:, index] - self.module.vLWPR[:, index]
        print('Prediction error:', self.module.epp[:, index-1])
        
        self.module.epr[:, index] = self.module.q[:, index-1] - self.module.posr[:, index] 
        self.module.evr[:, index] = self.module.qd[:, index-1] - self.module.velr[:, index] 
        print('Trajectory error:', self.module.epr[:, index])
        
        if index > 3:  
            '''self.meanerrorp[0] = np.mean(self.module.etp[0, 0:index])
            self.meanerrorp[1] = np.mean(self.module.etp[1, 0:index])
            self.meanerrorv[0] = np.mean(self.module.etv[0, 0:index])
            self.meanerrorv[1] = np.mean(self.module.etv[1, 0:index])
        
            # Normalizations
            self.module.normetp[0, index] = (self.module.etp[0, index] - np.min((self.module.etp[0, 0:index])))/(np.max((self.module.etp[0,0:index]))-np.min((self.module.etp[0,0:index])))
            self.module.normetp[1, index] = (self.module.etp[1, index] - np.min((self.module.etp[1, 0:index])))/(np.max((self.module.etp[1,0:index]))-np.min((self.module.etp[1,0:index])))
            self.module.normetv[0, index] = (self.module.etv[0, index] - np.min((self.module.etv[0, 0:index])))/(np.max((self.module.etv[0,0:index]))-np.min((self.module.etv[0,0:index])))
            self.module.normetv[1, index] = (self.module.etv[1, index] - np.min((self.module.etv[1, 0:index])))/(np.max((self.module.etv[1,0:index]))-np.min((self.module.etv[1,0:index])))'''
       
            self.meanerrorp[0] = np.mean(self.module.epr[0, 0:index])
            self.meanerrorp[1] = np.mean(self.module.epr[1, 0:index])
            self.meanerrorv[0] = np.mean(self.module.evr[0, 0:index])
            self.meanerrorv[1] = np.mean(self.module.evr[1, 0:index])
        
            # Normalizations
            self.module.normetp[0, index] = (self.module.epr[0, index] - np.min((self.module.epr[0, 0:index])))/(np.max((self.module.epr[0,0:index]))-np.min((self.module.epr[0,0:index])))
            self.module.normetp[1, index] = (self.module.epr[1, index] - np.min((self.module.epr[1, 0:index])))/(np.max((self.module.epr[1,0:index]))-np.min((self.module.epr[1,0:index])))
            self.module.normetv[0, index] = (self.module.evr[0, index] - np.min((self.module.evr[0, 0:index])))/(np.max((self.module.evr[0,0:index]))-np.min((self.module.evr[0,0:index])))
            self.module.normetv[1, index] = (self.module.evr[1, index] - np.min((self.module.evr[1, 0:index])))/(np.max((self.module.evr[1,0:index]))-np.min((self.module.evr[1,0:index])))
            
            #modules.module[i].module.normetp[0, j+1] = (modules.module[i].module.etp[0, j+1] / np.max(np.abs(modules.module[i].module.etp[0, 0:j+1])))#/2.0
    
    def update(self, index):
        self.mlcj.ML_update(self.input_lwpr, np.array([self.module.posr[0, index], self.module.velr[0, index], self.module.posr[1, index], self.module.velr[1, index]]))
        print("Current torquesLF:    ", self.module.torquesLF[:, index])
        self.mlcj.ML_rfs()
    
    def performControl(self, index):
         #THE BEST
        #self.module.epd[:, index]  =  (self.module.q[:,index] - self.module.pLWPR[:, index+1]) - self.module.epp[:, index] + (self.module.q[:, index] - self.module.pLWPR[:, index]) - self.module.epp[:, index-1] #+ self.module.etp[:,index] 
        self.module.epd[:, index]  =  (self.module.q[:,index] - self.module.pLWPR[:, index+1]) - self.module.epp[:, index] #+ (self.module.q[:, index] - self.module.pLWPR[:, index])# - self.module.epp[:, index-1]
        #self.module.epd[:, index]  =  (self.module.q[:,index] - self.module.pLWPR[:, index+1]) - self.module.epp[:, index] + self.module.etp[:,index] - self.module.epp[:, index-1] #The BEST

        #self.module.epd[:, index]  =  (self.module.q[:,index] - self.module.pLWPR[:, index+1]) - self.module.epp[:, index] # SMITH model 2
        #self.module.epd[:, index]  =  (self.module.q[:,index] - self.module.pLWPR[:, index+1]) + self.module.pLWPR[:, index] - self.module.posr[:, index] # SMITH model 3
        
        #self.module.epd[:, index]  = (self.module.q[:,index] - self.module.posr[:, index]) #+ self.module.epr[:, index] # No SMITH        
        
        self.module.ea[:, index+1] = 0.22
        self.module.ep[:, index+1] = self.module.epd[:, index] + self.module.outputDCN[0:2, index+1]
        self.module.ev[:, index+1] = ((self.module.ep[:, index+1] - self.module.ep[:, index]) / self.module.dt) + self.module.outputDCN[2:4, index+1]
        # Feedback error learning
        if index > 1:
            self.module.D[:, index] = self.module.D[:, index-1] + (self.module.ea[:, index+1] * self.ki) + (self.module.ep[:, index+1] * self.kp) + (self.module.ev[:, index+1] * self.kv)
        else:
            self.module.D[:, index] = (self.module.ea[:, index+1] * self.ki) + (self.module.ep[:, index+1] * self.kp) + (self.module.ev[:, index+1] * self.kv)

        tau = dynamics.rne(self.fab17, [self.module.q1[index], self.module.q2[index]], [0, 0], [1, 1], [0, 0, 0] )

        self.module.torquesLF[0, index+1] = tau[0, 0] * self.module.D[0, index]
        self.module.torquesLF[1, index+1] = tau[0, 1] * self.module.D[1, index]

        self.module.torquestot[0, index+1] = self.module.torquesLF[0, index+1] # + self.module.torqLWPR[0, index] + self.module.Ctorques[0, index]
        self.module.torquestot[1, index+1] = self.module.torquesLF[1, index+1] # + self.module.torqLWPR[1, index] + self.module.Ctorques[1, index]
    
    def __del__(self):
        print("Class object for Module_{} - Destroyed".format(self.ModuleID))

def set_pool(module, index):
    module, j = arg
    module.setModuleMotorsPos(j)
    return module

class MODULES_MULTITHREAD():
    def __init__(self, platform, modulesids, n_iter, control_type, eef_trajectory_type):
        self.module = [MODULE(platform, modulesids[i], n_iter, control_type, eef_trajectory_type, i) for i in range(len(modulesids))]
    
    def getMotorsPos(self, j):
        threads = []
        for module in self.module:
            t = Thread(target=module.getMotorsPos, args=(j,))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    
    def setModuleMotorsPos(self, j):
        threads = []
        for module in self.module:
            t = Thread(target=module.setModuleMotorsPos, args=(j,))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    
    def setModuleMotorsTorque(self, j):
        threads = []
        for module in self.module:
            t = Thread(target=module.setModuleMotorsTorque, args=(j,))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    
    def createInputData(self, index):
        threads = []
        for module in self.module:
            t = Thread(target=module.createInputData, args=(index,))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    
    def predict(self, index):
        threads = []
        for module in self.module:
            t = Thread(target=module.predict, args=(index,))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
            
    def estimateErrors(self, index):
        threads = []
        for module in self.module:
            t = Thread(target=module.estimateErrors, args=(index,))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
            
    def update(self, index):
        threads = []
        for module in self.module:
            t = Thread(target=module.update, args=(index,))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
            
    def performControl(self, index):
        threads = []
        for module in self.module:
            t = Thread(target=module.performControl, args=(index,))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    
    def __del__(self):
        print("Class object for Module_{} - Destroyed".format(self.module))
   
class MODULE_INI():

    def __init__(self, n_iter,  control_type):
        self.n_iter = n_iter
        self.njoints = 2
        self.nout = 4
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
        self.phase = math.pi / 2
        self.dt    = 0.01                 # 1 module 0.01 - 2 modules 0.02
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

        self.posr = np.zeros((self.njoints, self.n_iter + 1), dtype = np.double)
        self.velr = np.zeros((self.njoints, self.n_iter + 1), dtype = np.double)

        #self.input_lwpr = np.array([0])

        self.accr = np.zeros((self.njoints, self.n_iter + 1), dtype = np.double)
        self.D    = np.zeros((self.njoints, self.n_iter + 1), dtype = np.double)
        self.erra = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.errp = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.errv = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.epr = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.epd = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.evr = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.etp = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.etv = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.epp   = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.evv   = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.ea   = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.ep   = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.ev   = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.normetp = np.zeros((self.njoints, self.n_iter + 1), dtype = np.double)
        self.normetv = np.zeros((self.njoints, self.n_iter + 1), dtype = np.double)
        self.torquesLF   = np.zeros((self.njoints, self.n_iter + 1), dtype = np.double)
	
        if control_type == 0:
            self.Ctorques    = np.zeros((self.njoints, self.n_iter + 1), dtype = np.double)
            self.outputDCN        = np.zeros((self.njoints, self.n_iter + 1), dtype = np.double)
            self.torqLWPR    = np.zeros((self.njoints, self.n_iter + 1), dtype = np.double)
            self.output_x = np.zeros((self.njoints), dtype = np.double)
            self.outputC  = np.zeros((self.njoints, self.n_iter + 1), dtype = np.double)
        elif control_type == 1:
            self.Ctorques    = np.zeros((self.nout, self.n_iter + 1), dtype = np.double)
            self.outputDCN        = np.zeros((self.nout, self.n_iter + 1), dtype = np.double)
            self.torqLWPR    = np.zeros((self.nout, self.n_iter + 1), dtype = np.double)
            self.output_x = np.zeros((self.nout), dtype = np.double)
            self.outputC  = np.zeros((self.nout, self.n_iter + 1), dtype = np.double)
        elif control_type == 2:
            pass
        else:
            return

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
            self.x[i] = fab.l1 * math.cos(self.q1[i]) + fab.l2 * math.cos((self.q1[i] + self.q2[i]))
            self.y[i] = fab.l1 * math.sin(self.q1[i]) + fab.l2 * math.sin((self.q1[i] + self.q2[i]))

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
        #plt.plot(self.q1, self.q2)
        #plt.show()
        return self.q1, self.q2, self.q1d, self.q2d, self.q1dd, self.q2dd
    
    def calc_trajectory_8_changefreq(self, fab):
        self.A1 = 7
        #self.f0    = [1, 0.75, 0.5, 0.75, 1, 0.75, 0.5, 0.75, 1, 0.75, 0.5, 0.75, 1, 0.75, 0.5, 0.75, 1, 0.75, 0.5, 0.75, 1]
        self.f0    = [1.2, 1, 0.9, 1, 1.2, 1, 0.9, 1, 1.2, 1, 0.9, 1, 0.9, 1, 1.2,1, 0.9, 1, 1.2, 1, 0.9, 1, 1.2, 1, 0.9, 1, 1.2]
            
        self.phase = math.pi / 2
        aj = 1
        for i in range(self.n_iter):
            
            self.q1dd[i] = self.A1 * math.sin(2 * math.pi *self.f0[aj]* self.t0)
            self.q1d[i]  = ((-1 / 2) * math.pi) * self.A1 * math.cos(2 * math.pi *self.f0[aj]* self.t0)
            self.q1[i]   = (-math.pow(((1 / 2) * math.pi), 2)) * self.A1 * math.sin(2 * math.pi*self.f0[aj]* self.t0)

            self.q2dd[i]  = self.A1 * math.cos(4 * math.pi *self.f0[aj]* self.t0 + math.pi / 2)
            self.q2d[i]   = ((1 / 2) * math.pi)  * self.A1 * math.sin(4 * math.pi *self.f0[aj]* self.t0 + math.pi / 2)
            self.q2[i]    = (-math.pow(((1 / 2) * math.pi), 2)) * self.A1 * math.cos(4 * math.pi *self.f0[aj]* self.t0 + math.pi / 2)
            
            self.q[:, i]  = (self.q1[i], self.q2[i])
            self.qd[:, i] = (self.q1d[i], self.q2d[i])
            self.t0 += self.dt
            if i % 3000 == 0:
                aj += 1
                #print('Aj:', aj)
        #plt.plot(self.q1, self.q2)
        #plt.show()
        return self.q1, self.q2, self.q1d, self.q2d, self.q1dd, self.q2dd
    
    def calc_trajectory_8_changeA(self, fab):
        self.A    = [9, 7, 5, 7, 9, 7, 5, 7, 9, 7, 5, 7, 9, 7, 5, 7, 9, 7, 9, 7, 5, 7, 9, 7, 5, 7, 9, 7, 5, 7, 9]
        self.phase = math.pi / 2
        aj = 1
        for i in range(self.n_iter):
            
            self.q1dd[i] = self.A[aj] * math.sin(2 * math.pi * self.t0)
            self.q1d[i]  = ((-1 / 2) * math.pi) * self.A[aj] * math.cos(2 * math.pi * self.t0)
            self.q1[i]   = (-math.pow(((1 / 2) * math.pi), 2)) * self.A[aj] * math.sin(2 * math.pi * self.t0)

            self.q2dd[i]  = self.A[aj] * math.cos(4 * math.pi * self.t0 + math.pi / 2)
            self.q2d[i]   = ((1 / 2) * math.pi)  * self.A[aj] * math.sin(4 * math.pi * self.t0 + math.pi / 2)
            self.q2[i]    = (-math.pow(((1 / 2) * math.pi), 2)) * self.A[aj] * math.cos(4 * math.pi * self.t0 + math.pi / 2)
            self.q[:, i]  = (self.q1[i], self.q2[i])
            self.qd[:, i] = (self.q1d[i], self.q2d[i])
            self.t0 += self.dt
            if i % 3000 == 0:
                aj += 1
                print('Aj:', aj)
        #plt.plot(self.q1, self.q2)
        #plt.show()
        return self.q1, self.q2, self.q1d, self.q2d, self.q1dd, self.q2dd
