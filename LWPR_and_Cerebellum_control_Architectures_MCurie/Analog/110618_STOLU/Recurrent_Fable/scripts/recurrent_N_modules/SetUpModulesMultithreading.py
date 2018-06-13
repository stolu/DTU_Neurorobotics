#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""============================================================
* File Name: Recurrent_modular.py
* Purpose: Recurrent control loop - Control several modules of Fable at the same
            time, using one single dongle and from just
            one script
* Creation Date : 05-09-2017
* Last Modified : 05-09-2017
__author__      = "Silvia Tolu"
__credits__     = ["Silvia Tolu"]
__maintainer__     = "Silvia Tolu"
__email__     = ["stolu@elektro.dtu.dk"]

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

import functools
import multiprocessing
import dynamics
from datetime import datetime, date
import serial, time, random, math#, threading
#from threading import Thread

class Task:
    def __init__(self, function, index, num_modules, now=0):
        self.function = function
        self.index = index
        self.num_modules = num_modules
        self.now = now

class MODULE(multiprocessing.Process):
    def __init__(self, task_queue, platform, moduleID, n_iter, control_type, eef_trajectory_type, port = 0):
        # must call this before anything else
        multiprocessing.Process.__init__(self)
        
        # Pipe to receive module behaviour
        self.task_queue = task_queue

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
    
    def run(self):
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                self.task_queue.task_done()
                break
            if next_task.function == 'getMotorsPos':
                self.getMotorsPos(next_task.index)
            elif next_task.function == 'setModuleMotorsPos':
                self.setModuleMotorsPos(next_task.index)
            elif next_task.function == 'setModuleMotorsTorque':
                self.setModuleMotorsTorque(next_task.index)
            elif next_task.function == 'createInputData':
                self.createInputData(next_task.index)
            elif next_task.function == 'predict':
                self.predict(next_task.index)
            elif next_task.function == 'estimateErrors':
                self.estimateErrors(next_task.index)
            elif next_task.function == 'update':
                self.update(next_task.index)
            elif next_task.function == 'performControl':
                self.performControl(next_task.index)
            elif next_task.function == 'save':
                self.save(next_task.num_modules, next_task.now)
                
            self.task_queue.task_done()
            
    def getMotorsPos(self, j):
        # Receive feedback positions from motors
        self.module.posr[0, j+1] =  self.api.getModuleMotorPosition(self.ModuleID, 0)   # degrees
        self.module.posr[1, j+1] =  self.api.getModuleMotorPosition(self.ModuleID, 1)
        self.module.velr[0, j+1] = -self.api.getModuleMotorSpeed(self.ModuleID, 0)    # grad/s
        self.module.velr[1, j+1] = -self.api.getModuleMotorSpeed(self.ModuleID, 1)
    
    def setModuleMotorsPos(self, j):
        print('setModuleMotorPos for module ', self.ModuleID)
        
        # Set position
        self.api.setModuleMotorPosition(self.ModuleID, 0, self.module.q1[j])
        self.api.setModuleMotorPosition(self.ModuleID, 1, self.module.q2[j])
        
    def setModuleMotorsTorque(self, j):
        # Control in motor torques
        # FFORWARD
        if self.control_type == 0:
            self.api.setModuleMotorTorque(self.ModuleID, 0, -self.module.torquestot[0, j], 1, -self.module.torquestot[1, j], ack = False)
        # RECURRENT
        elif self.control_type == 1:
            self.api.setModuleMotorTorque(self.ModuleID, 0, -self.module.torquestot[0, j+1], 1, -self.module.torquestot[1, j+1], ack = False)
        # HYBRID
        elif self.control_type == 2:
            pass
        else:
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
            
            #self.module.normetp[0, j+1] = (self.module.etp[0, j+1] / np.max(np.abs(self.module.etp[0, 0:j+1])))#/2.0
    
    def update(self, index):
        self.mlcj.ML_update(self.input_lwpr, np.array([self.module.posr[0, index], self.module.velr[0, index], self.module.posr[1, index], self.module.velr[1, index]]))
        print("Current torquesLF:    ", self.module.torquesLF[:, index])
        self.mlcj.ML_rfs()
    
    def performControl(self, index):
        #THE BEST
        #self.module.epd[:, index]  =  (self.module.q[:,index] - self.module.pLWPR[:, index+1]) - self.module.epp[:, index] + (self.module.q[:, index] - self.module.pLWPR[:, index]) - self.module.epp[:, index-1] #+ self.module.etp[:,index] 
           
        self.module.epd[:, index]  =  (self.module.q[:,index] - self.module.pLWPR[:, index+1]) - self.module.epp[:, index] + self.module.etp[:,index] - self.module.epp[:, index-1] #The BEST

        #self.module.epd[:, index]  =  (self.module.q[:,index] - self.module.pLWPR[:, index+1]) - self.module.epp[:, index] # SMITH model 2
        #self.module.epd[:, index]  =  (self.module.q[:,index] - self.module.pLWPR[:, index+1]) + self.module.pLWPR[:, index] - self.module.posr[:, index] # SMITH model 3
        
        #self.module.epd[:, index]  = (self.module.q[:,index] - self.module.posr[:, index]) #+ self.module.epr[:, index] # No SMITH        
        
        self.module.ea[:, index+1] = 0.1#0.22
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
    
    def save(self, num_modules, now):
        scipy.io.savemat('TestREC_1box_{0}modules_fab{1}_{2}_{3}_{4}.mat'.format(num_modules,
                                                                        self.ModuleID, 
                                                                        now.strftime('%d-%m-%Y_%H:%M'), 
                                                                        self.mlcj.model.num_rfs[0], 
                                                                        self.module.init_D),
                 dict(q0 = self.module.q1, q1 = self.module.q2, q0d = self.module.q1d, q1d = self.module.q2d, 
                      velr0 = self.module.velr[0], velr1 = self.module.velr[1], posr0 = self.module.posr[0], 
                      posr1 = self.module.posr[1], errp0 = self.module.epr[0], errp1 = self.module.epr[1], errv0 = self.module.evr[0], errv1 = self.module.evr[1],
                      prederr0 = self.module.epp[0], prederr1 = self.module.epp[1], torquesLF0 = self.module.
                      torquesLF[0], torquesLF1 = self.module.torquesLF[1], DCNv0 = self.module.outputDCN[2], DCNv1 = self.module.outputDCN[3], 
                      DCNp0 = self.module.outputDCN[0], DCNp1 = self.module.outputDCN[1], vellwpr0 = self.module.vLWPR[0], vellwpr1 = self.module.vLWPR[1], 
                      poslwpr0 = self.module.pLWPR[0], poslwpr1 = self.module.pLWPR[1], weight_mod = self.mlcj.weights_mod))

        print('Recurrent Thread-test for Module_{}: finishing'.format(self.ModuleID))
    
    def __del__(self):
        print("Class object for Module_{} - Destroyed".format(self.ModuleID))

class MODULES_MULTITHREAD():
    def __init__(self, platform, modulesids, n_iter, control_type, eef_trajectory_type):
        self.tasks = multiprocessing.JoinableQueue()
        self.module = [MODULE(self.tasks, platform, modulesids[i], n_iter, control_type, eef_trajectory_type, i) for i in range(len(modulesids))]
        
        for module in self.module:
            module.start()
    
    def getMotorsPos(self, j):
        for i in range(len(self.module)):
            self.tasks.put(Task('getMotorsPos', j, len(self.module)))
        # Wait for all of the tasks to finish
        self.tasks.join()
    
    def setModuleMotorsPos(self, j):
        for i in range(len(self.module)):
            self.tasks.put(Task('setModuleMotorsPos', j, len(self.module)))
        # Wait for all of the tasks to finish
        self.tasks.join()
    
    def setModuleMotorsTorque(self, j):
        for i in range(len(self.module)):
            self.tasks.put(Task('setModuleMotorsTorque', j, len(self.module)))
        # Wait for all of the tasks to finish
        self.tasks.join()
    
    def createInputData(self, index):
        for i in range(len(self.module)):
            self.tasks.put(Task('createInputData', index, len(self.module)))
        # Wait for all of the tasks to finish
        self.tasks.join()
    
    def predict(self, index):
        for i in range(len(self.module)):
            self.tasks.put(Task('predict', index, len(self.module)))
        # Wait for all of the tasks to finish
        self.tasks.join()
            
    def estimateErrors(self, index):
        for i in range(len(self.module)):
            self.tasks.put(Task('estimateErrors', index, len(self.module)))
        # Wait for all of the tasks to finish
        self.tasks.join()
            
    def update(self, index):
        for i in range(len(self.module)):
            self.tasks.put(Task('update', index, len(self.module)))
        # Wait for all of the tasks to finish
        self.tasks.join()
            
    def performControl(self, index):
        for i in range(len(self.module)):
            self.tasks.put(Task('performControl', index, len(self.module)))
        # Wait for all of the tasks to finish
        self.tasks.join()
            
    def save(self, now):
        for i in range(len(self.module)):
            self.tasks.put(Task('save', 0, len(self.module), now))
        # Wait for all of the tasks to finish
        self.tasks.join()
    
#    def __del__(self):
#        print("Class object for Module_{} - Destroyed".format(self.module))
   
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
