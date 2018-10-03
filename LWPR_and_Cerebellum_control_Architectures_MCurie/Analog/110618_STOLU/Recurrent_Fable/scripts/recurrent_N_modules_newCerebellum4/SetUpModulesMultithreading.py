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
# Vel in radians ==> X * (1023/100 * 0.111 rpm *2*pi/60) ==> X* 0.1189

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
import serial, time, random, math #, threading
#from threading import Thread

class Task:
    def __init__(self, function, index, num_modules, t, now=0):
        self.function = function
        self.index = index
        self.num_modules = num_modules
        self.t = t
        self.now = now

class MODULE(multiprocessing.Process):
    def __init__(self, task_queue, platform, moduleID, n_iter, control_type, eef_trajectory_type, port = 0):
        # must call this before anything else
        multiprocessing.Process.__init__(self)
        
        # Pipe to receive module behaviour
        self.task_queue = task_queue

        #constant c1,c2,c3=1 for vertical down - constant c1= 2 c2,c3=2.5for vertical up position
        c1 = 1
        c2 = 1
        c3 = 1
        w0 = 4.5 #*1.5# 8.0 natural frequence 0.03 rad/s
        w1 = 5.5 #*1.5# 7.5
        self.integral = 1
        self.kp = [7.5, 10.5]       # Select the kp pf the Learning Feedback controller
        self.kp = [math.pow(w0, 2)*2.0, math.pow(w1, 2)*4.0]
        self.kv = [w0 * 0.40, w1 * 0.40]
        self.ki = [w0/40.0, w1/40.0]       # Select the ki pf the Learning Feedback controller
        
        '''self.kp = [math.pow(w0, 2)*0.6, math.pow(w1, 2)*0.9]
        self.kv = [self.kp[0]/8.0, self.kp[1]/8.0]
        self.ki = [self.kp[0]/6.0, self.kp[1]/6.0]       # Select the ki pf the Learning Feedback controller'''
        
        #self.kv = [6.4, 9.4]      # Select the kv pf the Learning Feedback controller
        #self.ki = [1.0, 1.0]
        self.counter = 0
        self.dummy_index = 1
        self.meanerrorp = [0.0, 0.0]
        self.meanerrorv = [0.0, 0.0]
        self.n_iter = n_iter
        self.njoints = 2
        self.nout = 2
        self.grav = [0, 0, 9.81]
        self.fab  = simfable.moduleFable()
        self.fab17 = self.fab.Fab()
        self.ModuleID = moduleID
        #self.module = []
        self.control_type = control_type
        self.eef_trajectory_type = eef_trajectory_type
        self.tau_norm_sign = 1
        
        self.mlcj_a = []
        self.mlcj_r = []
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
            self.mlcj_a = afel_class.MLandC(self.n_lwpr_inputs, self.njoints)
        # RECURRENT
        elif self.control_type == 1:
            self.N_LWPR_INPUTS_1MODULE_pos = 6 #6#10
            self.N_LWPR_INPUTS_1MODULE_vel = 6 #6#10
            #self.n_lwpr_inputs = len(self.ModuleID)*self.N_LWPR_INPUTS_1MODULE
            self.n_lwpr_inputs_pos = self.N_LWPR_INPUTS_1MODULE_pos#len(self.ModuleID)*self.N_LWPR_INPUTS_1MODULE
            self.n_lwpr_inputs_vel = self.N_LWPR_INPUTS_1MODULE_vel
            self.mlcj_r = rafel_class.MLandC(self.n_lwpr_inputs_pos,self.n_lwpr_inputs_vel, self.nout, self.njoints, 10.5, 90, 6.5, 90) # with 6 inputs
            #self.mlcj_r = rafel_class.MLandC(self.n_lwpr_inputs_pos,self.n_lwpr_inputs_vel, self.nout, self.njoints, 10.5, 100, 18.05, 100) # with 4 inputs
            
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
            elif next_task.function == 'setModuleMotorMaxSpeed':
                self.setModuleMotorMaxSpeed(next_task.index)
            elif next_task.function == 'createInputData_pos':
                self.createInputData_pos(next_task.index)
            elif next_task.function == 'createInputData_vel':
                self.createInputData_vel(next_task.index)
            elif next_task.function == 'predict':
                self.predict(next_task.index)
            elif next_task.function == 'estimateErrors':
                self.estimateErrors(next_task.index)
            elif next_task.function == 'update':
                self.update(next_task.index)
            elif next_task.function == 'performControl':
                self.performControl(next_task.index, next_task.t)
            elif next_task.function == 'save':
                self.save(next_task.num_modules, next_task.now)
                
            self.task_queue.task_done()
            
    def getMotorsPos(self, j):
        # Receive feedback positions from motors
        self.module.posr[0, j+1] = self.api.getModuleMotorPosition(self.ModuleID, 0)*math.pi/180.0   # The function returns degrees
        self.module.posr[1, j+1] = self.api.getModuleMotorPosition(self.ModuleID, 1)*math.pi/180.0
        self.module.velr[0, j+1] = -self.api.getModuleMotorSpeed(self.ModuleID, 0)*0.1189   # The function returns 100 percent
        self.module.velr[1, j+1] = -self.api.getModuleMotorSpeed(self.ModuleID, 1)*0.1189
        print("posr: ", self.module.posr[:, j+1])
        print("velr: ", self.module.velr[:, j+1])
        #print("velr0: ", self.module.velr[0, j+1])
        #print("velr1: ", self.module.velr[1, j+1])
        print("vel-des: ", self.module.qd[:, j])
        
    def setModuleMotorsPos(self, j):
        print('setModuleMotorPos for module ', self.ModuleID)
        # Set position
        self.api.setModuleMotorPosition(self.ModuleID, 0, self.module.q1[j]*180.0/math.pi)
        self.api.setModuleMotorPosition(self.ModuleID, 1, self.module.q2[j]*180.0/math.pi)
        
    def setModuleMotorMaxSpeed(self, j):
        self.api.setModuleMotorMaxSpeed(self.ModuleID, 0, j)
        self.api.setModuleMotorMaxSpeed(self.ModuleID, 1, j)
        
    def setModuleMotorsTorque(self, j):
        # Control in motor torques
        # FFORWARD
        if self.control_type == 0:
            self.api.setModuleMotorTorque(self.ModuleID, 0, (self.module.torquestot[0, j]* 100.0 / 1.5), 1, (self.module.torquestot[1, j]* 100.0 / 1.5), ack = False)
        # RECURRENT
        elif self.control_type == 1:
            self.api.setModuleMotorTorque(self.ModuleID, 0, (self.module.torquestot[0, j]* 100.0 / 1.5), 1, (self.module.torquestot[1, j]* 100.0 / 1.5), ack = False)
            print('Torquestot j0:', self.module.torquestot[0, j+1]*100/1.5)
            print('Torquestot j1:', self.module.torquestot[1, j+1]*100/1.5)
        # HYBRID
        elif self.control_type == 2:
            pass
        else:
            return
    
    def createInputData_pos(self, index):
        
        input_data_p = []
        self.module.normtorquestot[0, index] = self.module.torquestot[0, index] / 1.5
        self.module.normtorquestot[1, index] = self.module.torquestot[1, index] / 1.5
        
        self.module.normq1[index] = self.module.q1[index] / (math.pi/2)
        self.module.normq2[index] = self.module.q2[index] / (math.pi/2)
        
        self.module.normposr[0, index] = (self.module.posr[0, index] / (math.pi/2))
        self.module.normposr[1, index] = (self.module.posr[1, index] / (math.pi/2))
        
        input_data_p.append(self.module.normtorquestot[0, index])                
        input_data_p.append(self.module.normtorquestot[1, index])
        input_data_p.append(self.module.normq1[index]) 
        input_data_p.append(self.module.normq2[index])
        input_data_p.append(self.module.normposr[0, index]) 
        input_data_p.append(self.module.normposr[1, index]) 
        
        self.input_lwpr_pos = np.array(input_data_p)
        #print('input lwpr pos:', self.input_lwpr_pos)
    
    def createInputData_vel(self, index):
        
        input_data_v = []
        self.module.normtorquestot[0, index] = self.module.torquestot[0, index] / 1.5
        self.module.normtorquestot[1, index] = self.module.torquestot[1, index] / 1.5
        
        self.module.normq1d[index] = self.module.q1d[index] / (0.548)
        self.module.normq2d[index] = self.module.q2d[index] / (0.548)
        
        self.module.normvelr[0, index] = self.module.velr[0, index] / 100.0 * 0.1189
        self.module.normvelr[1, index] =  self.module.velr[1, index] / 100.0 * 0.1189
        
        input_data_v.append(self.module.normtorquestot[0, index])                
        input_data_v.append(self.module.normtorquestot[1, index])
        input_data_v.append(self.module.normq1d[index]) 
        input_data_v.append(self.module.normq2d[index])
        input_data_v.append(self.module.normvelr[0, index]) 
        input_data_v.append(self.module.normvelr[1, index])
    
        self.input_lwpr_vel = np.array(input_data_v)
        #print('input lwpr vel:', self.input_lwpr_vel)
        
    def predict(self, index):
        if index > 0:
            (self.module.pLWPR[:, index+1],
             self.module.outputDCNp[:, index+1], self.weights_mod_p, self.module.init_D) = self.mlcj_r.ML_prediction_pos(self.input_lwpr_pos,
                                                                                            self.module.epr[:, index],
                                                                                            self.meanerrorp,
                                                                                            self.module.normetv[:, index])
             
            (self.module.vLWPR[:, index+1],
             self.module.outputDCNv[:, index+1], self.weights_mod_v, self.module.init_D) = self.mlcj_r.ML_prediction_vel(self.input_lwpr_vel,
                                                                                            self.module.evr[:, index], 
                                                                                            self.meanerrorv, 
                                                                                            self.module.normetv[:, index])
            #self.module.pLWPR[0, index+1] = (self.module.output_x[0])
            #self.module.vLWPR[0, index+1] = (self.module.output_x[1])
            #self.module.pLWPR[1, index+1] = (self.module.output_x[2])
            #self.module.vLWPR[1, index+1] = (self.module.output_x[3])
        
        #print("lwpr output positions : ", self.module.pLWPR[:, index+1])
        #print("lwpr output velocities : ", self.module.vLWPR[:, index+1])
        
        print("DCN_positions: ", self.module.outputDCNp[:, index+1])
        print("DCN_velocities: ", self.module.outputDCNv[:, index+1])
        #print("C output: ", self.module.output_C[:, index+1])
    
    def estimateErrors(self, index):
        
        self.module.etp[:, index+1] = - self.module.q[:, index] + self.module.pLWPR[:, index+1]
        self.module.etv[:, index+1] = - self.module.qd[:, index] + self.module.vLWPR[:, index+1]
        #self.module.etv[1, index+1] = - self.module.qd[1, index] + self.module.vLWPR[1, index+1]
        
        # Prediction errors
        # should it be index + 100ms?
        self.module.epp[:, index+1] = -self.module.posr[:, index+1] + self.module.pLWPR[:, index+1]
        self.module.evv[:, index+1] = -(self.module.velr[:, index+1]) + self.module.vLWPR[:, index+1]
        #self.module.evv[1, index+1] = -(self.module.velr[1, index+1]) + self.module.vLWPR[1, index+1]
        print('Prediction position error:', self.module.epp[:, index+1])
        print('Prediction velocity error:', self.module.evv[:, index+1])
        # should it be put in index + 100ms??
        self.module.epr[:, index+1] = -self.module.q[:, index] + self.module.posr[:, index+1] 
        self.module.evr[:, index+1] = -self.module.qd[:, index] + (self.module.velr[:, index+1])
        #self.module.evr[1, index+1] = -self.module.qd[1, index] + (self.module.velr[1, index+1])
        #self.module.evr[:, index+1] = self.module.epr[:, index+1] - self.module.epr[:, index] / self.module.dt
        print('Trajectory position error:', self.module.epr[:, index+1])
        print('Trajectory velocity error:', self.module.evr[:, index+1])
        
        if index > 3:  
            '''self.meanerrorp[0] = np.mean(self.module.epr[0, 0:index+1])
            self.meanerrorp[1] = np.mean(self.module.epr[1, 0:index+1])
            self.meanerrorv[0] = np.mean(self.module.evr[0, 0:index+1])
            self.meanerrorv[1] = np.mean(self.module.evr[1, 0:index+1])'''
            
            if self.counter <= self.module.ntrj: # a quarter of traj.
                print('counter', self.counter)
                print(self.module.ntrj)
                self.meanerrorp[0] = np.mean(self.module.epr[0, 1:self.counter])
                self.meanerrorp[1] = np.mean(self.module.epr[1, 1:self.counter])
                self.meanerrorv[0] = np.mean(self.module.evr[0, 1:self.counter])
                self.meanerrorv[1] = np.mean(self.module.evr[1, 1:self.counter])
                self.counter = self.counter + 1
            else:
                self.meanerrorp[0] = np.mean(self.module.epr[0, index+1-self.module.ntrj:index+1])
                self.meanerrorp[1] = np.mean(self.module.epr[1, index+1-self.module.ntrj:index+1])
                self.meanerrorv[0] = np.mean(self.module.evr[0, index+1-self.module.ntrj:index+1])
                self.meanerrorv[1] = np.mean(self.module.evr[1, index+1-self.module.ntrj:index+1])
        
            # Normalizations
            if self.module.epr[0, index+1] > math.pi:
                self.module.epr[0, index+1] = math.pi
            elif self.module.epr[0, index+1] < -math.pi:
                self.module.epr[0, index+1] = -math.pi
            if self.module.epr[1, index+1] > math.pi:
                self.module.epr[1, index+1] = math.pi
            elif self.module.epr[1, index+1] < -math.pi:
                self.module.epr[1, index+1] = -math.pi
            if self.module.evr[0, index+1] > math.pi:
                self.module.evr[0, index+1] = math.pi
            elif self.module.evr[0, index+1] < -math.pi:
                self.module.evr[0, index+1] = -math.pi
            if self.module.evr[1, index+1] > math.pi:
                self.module.evr[1, index+1] = math.pi
            elif self.module.evr[1, index+1] < -math.pi:
                self.module.evr[1, index+1] = -math.pi
                
            if self.tau_norm_sign == 0:
                epr0_sign = (self.module.epr[0, index+1]/abs(self.module.epr[0, index+1]))
                epr1_sign = (self.module.epr[1, index+1]/abs(self.module.epr[1, index+1]))
                evr0_sign = (self.module.evr[0, index+1]/abs(self.module.evr[0, index+1]))
                evr1_sign = (self.module.evr[1, index+1]/abs(self.module.evr[1, index+1]))
            else:
                epr0_sign = 1.
                epr1_sign = 1.
                evr0_sign = 1.
                evr1_sign = 1.
            
            self.module.normetp[0, index+1] = epr0_sign*(1. - 0.)*(self.module.epr[0, index+1] - (-math.pi))/(math.pi - (-math.pi))
            self.module.normetp[1, index+1] = epr1_sign*(1. - 0.)*(self.module.epr[1, index+1] - (-math.pi))/(math.pi - (-math.pi))
            self.module.normetv[0, index+1] = evr0_sign*(1. - 0.)*(self.module.evr[0, index+1] - (-math.pi))/(math.pi - (-math.pi))
            self.module.normetv[1, index+1] = evr1_sign*(1. - 0.)*(self.module.evr[1, index+1] - (-math.pi))/(math.pi - (-math.pi))
            #print('normetp_0: ', self.module.normetp[0, index+1])
            #print('normetp_1: ', self.module.normetp[1, index+1])
            #print('normetv_0: ', self.module.normetv[0, index+1])
            #print('normetv_1: ', self.module.normetv[1, index+1])
            #self.module.normetp[0, index+1] = (self.module.epr[0, index+1] - np.min((self.module.epr[0, 0:index+1])))/(np.max((self.module.epr[0,0:index+1]))-np.min((self.module.epr[0,0:index+1])))
            #self.module.normetp[1, index+1] = (self.module.epr[1, index+1] - np.min((self.module.epr[1, 0:index+1])))/(np.max((self.module.epr[1,0:index+1]))-np.min((self.module.epr[1,0:index+1])))
            #self.module.normetv[0, index+1] = (self.module.evr[0, index+1] - np.min((self.module.evr[0, 0:index+1])))/(np.max((self.module.evr[0,0:index+1]))-np.min((self.module.evr[0,0:index+1])))
            #self.module.normetv[1, index+1] = (self.module.evr[1, index+1] - np.min((self.module.evr[1, 0:index+1])))/(np.max((self.module.evr[1,0:index+1]))-np.min((self.module.evr[1,0:index+1])))
            
            #self.module.normetp[0, j+1] = (self.module.etp[0, j+1] / np.max(np.abs(self.module.etp[0, 0:j+1])))#/2.0
    
    def update(self, index):
        
        self.mlcj_r.ML_update_pos(self.input_lwpr_pos, np.array([self.module.posr[0, index+1], self.module.posr[1, index+1]]))
        self.mlcj_r.ML_update_vel(self.input_lwpr_vel, np.array([self.module.velr[0, index+1], self.module.velr[1, index+1]]))
        self.mlcj_r.ML_rfs()
        #self.mlcj.ML_update(self.input_lwpr, np.array([self.module.normposr[0, index], self.module.normvelr[0, index], self.module.normposr[1, index], self.module.normvelr[1, index]]))
        #print("Current torquesLF:    ", self.module.torquesLF[:, index])
        
       
    def performControl(self, index, t):
        #print('t', t)
        tau = dynamics.rne(self.fab17, [self.module.q[0, index], self.module.q[1, index]], [0, 0], [1, 1], [0, 0, 0] )
        #print('tau: ', tau)
        if index >= self.module.ntrj: #5 points are equal to 0.1 ms delay
            # SMITH # THE BEST
            #self.module.epd[:, index+1]  =  (-self.module.q[:,index] + self.module.pLWPR[:, index+1]) + self.module.epp[:, index] + self.module.etp[:,index] + self.module.epp[:, index-1]# The BEST
            # should it be epp di index??
            self.module.epd[:, index+1]  =  (-self.module.q[:,index] + self.module.pLWPR[:, index+1]) + self.module.epp[:, index-2] #+ self.module.etp[:,index] + self.module.epp[:, index-1]# The BEST   
            self.module.epdd[:, index+1]  =  (-self.module.qd[:,index] + (self.module.vLWPR[:, index+1])) + self.module.evv[:, index-2] #+ self.module.etv[:,index] + self.module.evv[:, index-1] 
            #self.module.epdd[1, index+1]  =  (-self.module.qd[1,index] + (self.module.vLWPR[1, index+1])) + self.module.evv[1, index-1] #+ self.module.etv[:,index] + self.module.evv[:, index-1] 
            
            self.module.ep[:, index+1] = self.module.epd[:, index+1] + self.module.outputDCNp[:, index+1]
            #self.module.ev[:, index+1] = ((self.module.ep[:, index+1] - self.module.ep[:, index]) / self.module.dt) #+ self.module.outputDCN[2:4, index+1] #+ self.module.etv[:,index] + self.module.evv[:, index-1]
            
            self.module.ev[:, index+1] = self.module.epdd[:, index+1] + self.module.outputDCNv[:, index+1]#+ self.module.etv[:,index] + self.module.evv[:, index-1]
                
            #self.module.ev[:, index+1] = ((self.module.epr[:, index] - self.module.epr[:, index-1]) / self.module.dt)
            #self.module.ev[:, index+1] =  (-self.module.qd[:,index] + self.module.vLWPR[:, index+1]) - self.module.evv[:, index] - self.module.outputDCN[2:4, index+1]#+ self.module.etv[:,index] - self.module.evv[:, index-1]
            #self.module.ev[:, index+1] = (self.module.posr[:,index] - self.module.posr[:, index-1]) / self.module.dt #+ self.module.outputDCN[2:4, index+1]#+ self.module.etv[:,index] + self.module.evv[:, index-1] #self.module.dt - self.module.outputDCN[2:4, index+1]
                
            '''
            if self.dummy_index == 0:
                self.module.interrors = []
                self.module.integral_time = []
            '''   
            if self.integral == 0:
               
                if self.dummy_index <= self.module.N:#-1:
                    self.module.interrors0.append(self.module.ep[0, index+1])
                    self.module.interrors1.append(self.module.ep[1, index+1])
                    #print(self.module.interrors0)
                    self.module.integral_time.append(t)
                    self.module.ea[0, index+1] = ((self.module.integral_time[self.dummy_index-1] - self.module.integral_time[0])/self.dummy_index)*(self.module.interrors0[0]*0.5 + self.module.interrors0[self.dummy_index-1]*0.5 + np.sum(self.module.interrors0[1:-1]))
                    self.module.ea[1, index+1] = ((self.module.integral_time[self.dummy_index-1] - self.module.integral_time[0])/self.dummy_index)*(self.module.interrors1[0]*0.5 + self.module.interrors1[self.dummy_index-1]*0.5 + np.sum(self.module.interrors1[1:-1]))
                    self.dummy_index = self.dummy_index + 1
                else:
                     # self.dummy_index = 0
                    self.module.interrors0.pop(0)
                    self.module.interrors0.append(self.module.ep[0, index+1])
                    self.module.interrors1.pop(0)
                    self.module.interrors1.append(self.module.ep[1, index+1])
                    #print(self.module.interrors0)
                    self.module.integral_time.pop(0)
                    self.module.integral_time.append(t)   
                    self.module.ea[0, index+1] = ((self.module.integral_time[self.module.N-1] - self.module.integral_time[0])/self.module.N)*(self.module.interrors0[0]*0.5 + self.module.interrors0[self.module.N-1]*0.5 + np.sum(self.module.interrors0[1:-1]))
                    self.module.ea[1, index+1] = ((self.module.integral_time[self.module.N-1] - self.module.integral_time[0])/self.module.N)*(self.module.interrors1[0]*0.5 + self.module.interrors1[self.module.N-1]*0.5 + np.sum(self.module.interrors1[1:-1]))
            else:
                self.module.ea[:, index+1] = (self.module.ev[:, index+1] - self.module.ev[:, index]) / self.module.dt#self.module.dt #0.1#0.22 
        else:
            # NO SMITH
            self.module.epd[:, index+1]  =  (-self.module.q[:,index] + self.module.posr[:, index]) #+ self.module.epr[:,index]
        
            # NO SMITH
            #self.module.epd[:, index]  = (-self.module.q[:,index] + self.module.posr[:, index])
            self.module.ep[:, index+1] = self.module.epd[:, index+1] 
            #self.module.ev[:, index+1] = ((self.module.ep[:, index+1] - self.module.ep[:, index]) / self.module.dt)#self.module.dt)
            self.module.ev[:, index+1] = (-self.module.qd[:,index] + self.module.velr[:, index])#self.module.dt)
            #self.module.ev[:, index+1] = (self.module.posr[:,index] - self.module.posr[:, index-1]) / self.module.dt
            self.module.ea[:, index+1] = (self.module.ev[:, index+1] - self.module.ev[:, index]) / self.module.dt#self.module.dt #0.1#0.22
        
        # Feedback error learning
        '''
        if index > 1:
            self.module.D[:, index] = self.module.D[:, index-1] + (self.module.ea[:, index+1] * self.ki) + (self.module.ep[:, index+1] * self.kp) + (self.module.ev[:, index+1] * self.kv)
        else:
            self.module.D[:, index] = (self.module.ea[:, index+1] * self.ki) + (self.module.ep[:, index+1] * self.kp) + (self.module.ev[:, index+1] * self.kv)

        tau = dynamics.rne(self.fab17, [self.module.q1[index], self.module.q2[index]], [0, 0], [1, 1], [0, 0, 0] )
        print("tau: ", tau)
        self.module.torquesLF[0, index+1] = tau[0, 0] * self.module.D[0, index]
        self.module.torquesLF[1, index+1] = tau[0, 1] * self.module.D[1, index]
        #print("D0: ", self.module.D[0, index])
        #print("D1: ", self.module.D[1, index])
        '''
        # PID SMITH and C
        self.module.torquestot[0, index+1] = -tau[0, 0] * (((-self.module.ep[0, index+1] ) * self.kp[0]) + ((-self.module.ev[0, index+1])* self.kv[0]) - (self.module.ea[0, index+1]* self.ki[0]))
        self.module.torquestot[1, index+1] = -tau[0, 1] * (((-self.module.ep[1, index+1] ) * self.kp[1]) + ((-self.module.ev[1, index+1])* self.kv[1]) - (self.module.ea[1, index+1]* self.ki[1]))
        
        '''
        if self.module.torquestot[0, index] > 100.0:
            self.module.torquestot[0, index] = 100.0
        if self.module.torquestot[1, index] > 100.0:
            self.module.torquestot[1, index] = 100.0
            
        if self.module.torquestot[0, index] < -100.0:
            self.module.torquestot[0, index] = -100.0
        if self.module.torquestot[1, index] < -100.0:
            self.module.torquestot[1, index] = -100.0'''
        # PID SMITH NOT C
        
        #self.module.torquestot[0, index] = (-self.module.ep[0, index+1] * self.kp[0]) - (self.module.ev[0, index+1] * self.kv[0]) + (self.module.ea[0, index+1]* self.ki[0]) #- (self.module.ea[0, index+1])
        #self.module.torquestot[1, index] = (-self.module.ep[1, index+1] * self.kp[1]) - (self.module.ev[1, index+1] * self.kv[1]) + (self.module.ea[1, index+1]* self.ki[1])
        
        #print("torquestot j0: ", self.module.torquestot[0, index+1])
        #print("torquestot j1: ", self.module.torquestot[1, index+1])
        
        #print("kv*ev j0: ", self.module.ev[0, index+1] * self.kv[0])
        #print("kv*ev j1 ", self.module.ev[1, index+1] * self.kv[1])
        #print("kp*ep j0: ", self.module.ep[0, index+1] * self.kp[0]) # * self.kp)
        #print("kp*ep j1: ", self.module.ep[1, index+1] * self.kp[1]) # * self.kp)
        # show errors
        
    def save(self, num_modules, now):

        N = self.module.ntrj
        
        moving_avesp0 = []
        dummy = 0.0
        for idx in range(len(self.module.epr[0])): # enumerate(self.module.epr[0], 1):
            dummy_N = 0.0
            if (idx>N) and (idx+N) < len(self.module.epr[0]):
                for n in range (0, N):
                    dummy_N += abs(self.module.epr[0, idx-N+n])
                moving_avesp0.append(dummy_N / (1.*N))
            elif idx>len(self.module.epr[0])-N:
                it = len(self.module.epr[0])-idx-1
                for n in range(0,it):
                    dummy_N += abs(self.module.epr[0, idx-it+n])
                moving_avesp0.append(dummy_N / (1.*it + 2.))
            else:
                dummy += abs(self.module.epr[0,idx])
                moving_avesp0.append(dummy/(idx+1.))    
        print(moving_avesp0)
        
        moving_avesp1 = []
        dummy = 0.0
        for idx in range(len(self.module.epr[1])):
            dummy_N = 0.0
            if (idx>N) and (idx+N) < len(self.module.epr[1]):
                for n in range (0,N):
                    dummy_N += abs(self.module.epr[1,idx-N+n])
                moving_avesp1.append( dummy_N / (1.*N) )
            elif idx>len(self.module.epr[1])-N:
                it = len(self.module.epr[1])-idx-1
                for n in range(0,it):
                    dummy_N += abs(self.module.epr[1,idx-it+n])
                moving_avesp1.append(dummy_N / (1.*it + 2.))
            else:
                dummy += abs(self.module.epr[1,idx])
                moving_avesp1.append(dummy/(idx+1.))    
                 
        print(moving_avesp1)
        
        moving_avesv0 = []
        
        dummy = 0.0
        for idx in range(len(self.module.evr[0])):
            dummy_N = 0.0
            if (idx>N) and (idx+N) < len(self.module.evr[0]):
                for n in range (0,N):
                    dummy_N += abs(self.module.evr[0,idx-N+n])
                moving_avesv0.append( dummy_N/(1.*N) )
            elif idx>len(self.module.evr[0])-N:
                it = len(self.module.evr[0])-idx-1
                for n in range(0,it):
                    dummy_N += abs(self.module.evr[0,idx-it+n])
                moving_avesv0.append(dummy_N/ (1.*it+2.))
            else:
                dummy += abs(self.module.evr[0,idx])
                moving_avesv0.append(dummy/(idx+1.))    
                 
        moving_avesv1 = []
        dummy = 0.0
        for idx in range(len(self.module.evr[1])):
            dummy_N = 0.0
            if (idx>N) and (idx+N) < len(self.module.evr[1]):
                for n in range (0,N):
                    dummy_N += abs(self.module.evr[1,idx-N+n])
                moving_avesv1.append( dummy_N/(1.*N) )
            elif idx>len(self.module.evr[1])-N:
                it = len(self.module.evr[1])-idx-1
                for n in range(0,it):
                    dummy_N += abs(self.module.evr[1,idx-it+n])
                moving_avesv1.append(dummy_N / (1.*it+2.))
            else:
                dummy += abs(self.module.evr[1,idx])
                moving_avesv1.append(dummy/(idx+1.))    
                 
        #plt.plot(moving_avesp0)
        #plt.show()
        
        scipy.io.savemat('TestREC_1box_{0}modules_fab{1}_{2}_{3}_{4}.mat'.format(num_modules,
                                                                        self.ModuleID, 
                                                                        now.strftime('%d-%m-%Y_%H:%M'),
                                                                        self.mlcj_r.model_p.num_rfs[0], 
                                                                        self.mlcj_r.model_p.init_D[0][0]),
                 dict(q0 = self.module.q1, q1 = self.module.q2, q0d = self.module.qd[0, :], q1d = self.module.qd[1, :], 
                      velr0 = self.module.velr[0], velr1 = self.module.velr[1], posr0 = self.module.posr[0], 
                      posr1 = self.module.posr[1], errp0 = self.module.epr[0], errp1 = self.module.epr[1], errv0 = self.module.evr[0], errv1 = self.module.evr[1],
                      prederr0 = self.module.epp[0], prederr1 = self.module.epp[1], torquestot0 = self.module.
                      torquestot[0], torquestot1 = self.module.torquestot[1], DCNv0 = self.module.outputDCNv[0], DCNv1 = self.module.outputDCNv[1], 
                      DCNp0 = self.module.outputDCNp[0], DCNp1 = self.module.outputDCNp[1], vellwpr0 = self.module.vLWPR[0], vellwpr1 = self.module.vLWPR[1], 
                      poslwpr0 = self.module.pLWPR[0], poslwpr1 = self.module.pLWPR[1], weight_mod_p = self.weights_mod_p, weight_mod_v = self.weights_mod_v,
                      error_p0 = moving_avesp0, error_p1 = moving_avesp1, error_v0 = moving_avesv0, error_v1 = moving_avesv1))

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
            self.tasks.put(Task('getMotorsPos', j, len(self.module), 0))
        # Wait for all of the tasks to finish
        self.tasks.join()
    
    def setModuleMotorsPos(self, j):
        for i in range(len(self.module)):
            self.tasks.put(Task('setModuleMotorsPos', j, len(self.module), 0))
        # Wait for all of the tasks to finish
        self.tasks.join()
    
    def setModuleMotorsTorque(self, j):
        for i in range(len(self.module)):
            self.tasks.put(Task('setModuleMotorsTorque', j, len(self.module), 0))
        # Wait for all of the tasks to finish
        self.tasks.join()
        
    def setModuleMotorMaxSpeed(self, j):
        for i in range(len(self.module)):
            self.tasks.put(Task('setModuleMotorMaxSpeed', j, len(self.module), 0))
        # Wait for all of the tasks to finish
        self.tasks.join()
        
    def createInputData_pos(self, index):
        for i in range(len(self.module)):
            self.tasks.put(Task('createInputData_pos', index, len(self.module), 0))
        # Wait for all of the tasks to finish
        self.tasks.join()
        
    def createInputData_vel(self, index):
        for i in range(len(self.module)):
            self.tasks.put(Task('createInputData_vel', index, len(self.module), 0))
        # Wait for all of the tasks to finish
        self.tasks.join()
    
    def predict(self, index):
        for i in range(len(self.module)):
            self.tasks.put(Task('predict', index, len(self.module), 0))
        # Wait for all of the tasks to finish
        self.tasks.join()
            
    def estimateErrors(self, index):
        for i in range(len(self.module)):
            self.tasks.put(Task('estimateErrors', index, len(self.module), 0))
        # Wait for all of the tasks to finish
        self.tasks.join()
            
    def update(self, index):
        for i in range(len(self.module)):
            self.tasks.put(Task('update', index, len(self.module), 0))
        # Wait for all of the tasks to finish
        self.tasks.join()
            
    def performControl(self, index, t):
        for i in range(len(self.module)):
            self.tasks.put(Task('performControl', index, len(self.module), t))
        # Wait for all of the tasks to finish
        self.tasks.join()
            
    def save(self, now):
        for i in range(len(self.module)):
            self.tasks.put(Task('save', 0, len(self.module), 0, now))
        # Wait for all of the tasks to finish
        self.tasks.join()
    
#    def __del__(self):
#        print("Class object for Module_{} - Destroyed".format(self.module))
   
class MODULE_INI():

    def __init__(self, n_iter, control_type):
        self.n_iter = n_iter
        self.dtt    = 0.005
        self.ntrj = int(1 / self.dtt) #+ 1
        self.njoints = 2
        self.nout = 2
        self.N = int(self.ntrj/4.0)
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
        self.dt    = 0.1                 # 1 module 0.01 - 2 modules 0.02
        
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
        self.integral_time = [0 for m in range(self.N)]
        self.interrors0 = [0 for m in range(self.N)]
        self.interrors1 = [0 for m in range(self.N)]
        #self.input_lwpr = np.array([0])

        self.accr = np.zeros((self.njoints, self.n_iter + 1), dtype = np.double)
        self.D    = np.zeros((self.njoints, self.n_iter + 1), dtype = np.double)
        self.erra = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.errp = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.errv = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.epr = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.epd = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.epdd = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
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
        self.normtorquestot = np.zeros((self.njoints, self.n_iter + 1), dtype = np.double)
        self.normq1 = [0 for k in range(self.n_iter + 1)]
        self.normq1d = [0 for k in range(self.n_iter + 1)]
        self.normq2 = [0 for k in range(self.n_iter + 1)]
        self.normq2d = [0 for k in range(self.n_iter + 1)]
        self.normposr = np.zeros((self.njoints, self.n_iter + 1), dtype = np.double)
        self.normvelr = np.zeros((self.njoints, self.n_iter + 1), dtype = np.double)
        
        self.torquesLF   = np.zeros((self.njoints, self.n_iter + 1), dtype = np.double)
    
        if control_type == 0:
            self.Ctorques    = np.zeros((self.njoints, self.n_iter + 1), dtype = np.double)
            self.outputDCN   = np.zeros((self.njoints, self.n_iter + 1), dtype = np.double)
            self.torqLWPR    = np.zeros((self.njoints, self.n_iter + 1), dtype = np.double)
            self.output_x    = np.zeros((self.njoints), dtype = np.double)
            self.outputC     = np.zeros((self.njoints, self.n_iter + 1), dtype = np.double)
        elif control_type == 1:
            self.outputDCNp  = np.zeros((self.nout, self.n_iter + 1), dtype = np.double)
            self.outputDCNv  = np.zeros((self.nout, self.n_iter + 1), dtype = np.double)
            self.pLWPR       = np.zeros((self.nout, self.n_iter + 1), dtype = np.double)
            self.vLWPR       = np.zeros((self.nout, self.n_iter + 1), dtype = np.double)
            #self.output_x = np.zeros((self.nout), dtype = np.double)
            #self.outputC  = np.zeros((self.nout, self.n_iter + 1), dtype = np.double)
        elif control_type == 2:
            pass
        else:
            return

        #self.DCNv        = np.zeros((self.nout, self.n_iter + 1), dtype = np.double)
        self.torquestot  = np.zeros((self.njoints, self.n_iter + 1), dtype = np.double)
        #self.posC = np.zeros((self.njoints, n_iter + 1), dtype=np.double)
        #self.velC = np.zeros((self.njoints, n_iter + 1), dtype=np.double)
        

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
        '''
        self.A1    = math.radians(40.) # 10.
        self.phase = math.radians(90.)
        for i in range(self.n_iter):
           self.q1dd[i] = -4.*np.power(np.pi, 2) *self.A1 * np.sin(2 * np.pi *self.t0)#"accelleration"
           self.q1d[i] = 2.*np.pi*self.A1 * np.cos(2 * np.pi *self.t0) 
           self.q1[i] = self.A1 * np.sin(2 * np.pi *self.t0) 
               
           self.q2dd[i] = -16.*np.power(np.pi, 2) *self.A1 * np.cos(4* np.pi *self.t0 + self.phase)#"accelleration"
           self.q2d[i] = -4.*np.pi*self.A1 * np.sin(4 * np.pi *self.t0 + self.phase) 
           self.q2[i] = self.A1 * np.cos(4 * np.pi *self.t0 + self.phase)
           
           self.t0 += self.dt
        '''
        self.A1 = 20.
        self.phase = math.pi / 2
        for i in range(self.n_iter):
            self.q1dd[i] = (self.A1 * math.sin(2 * math.pi * self.t0)) * (math.pi/180.0)
            self.q1d[i]  = ((-1 /2*math.pi) * self.A1 * math.cos(2 * math.pi * self.t0)) * (math.pi/180.0)
            self.q1[i]   = (-math.pow(1/2*math.pi, 2) * self.A1 * math.sin(2 * math.pi * self.t0)) * (math.pi/180.0)

            self.q2dd[i]  = self.A1 * 2.0 * math.cos(4 * math.pi * self.t0 + math.pi / 2) * math.pi/180.0
            self.q2d[i]   = ((1 /4 * math.pi) * 2.0 * self.A1 * math.sin(4 * math.pi * self.t0 + math.pi / 2)) * (math.pi/180.0)
            self.q2[i]    = (-math.pow(1/4*math.pi, 2) * 2.0 * self.A1 * math.cos(4*math.pi*self.t0 + math.pi / 2)) * (math.pi/180.0)
            self.q[:, i]  = (self.q1[i], self.q2[i])
            self.qd[:, i] = (self.q1d[i], self.q2d[i])
            
            self.t0 += self.dtt
        
        self.ntrj = int(1/self.dtt) 
        print(self.ntrj)
        plt.plot(self.q1, self.q2)
        #plt.plot(self.q1)
        plt.show()
        
        return self.ntrj, self.q, self.q1, self.q2, self.q1d, self.q2d, self.q1dd, self.q2dd
    
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
