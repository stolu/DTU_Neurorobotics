#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""============================================================
* File Name     : AFEL_LWPRandC.py
* Purpose       : MLandC class for modular configuration
* Creation Date : Mon 23 Nov 2017 13:05 PM CET
* Last Modified : Tue 23 Nov 2017 15:00 PM CET
============================================================"""

__author__      = "Silvia Tolu"
__credits__     = ["Silvia Tolu"]
__maintainer__  = "Carlos Corchado and Silvia Tolu"
__email__       = ["stolu@elektro.dtu.dk"]


from lwpr import *
import sys, time, random, math
import numpy as np
from numpy import linalg as LA


class MLandC:

    def __init__(self, nin, nout, njoints):
        self.model = [0 for k in range(1)]
        self.njoints = njoints                               # njoints is number of links
        self.nin  = nin                                      # nin number of LWPR inputs
        self.nout  = nout
        
        self.w_pf_pct = [0 for k in range(njoints)]
        self.w_pf_pc = [0 for k in range(njoints)]
        self.w_pc_dcn = [0 for k in range(njoints)]
        self.w_pc_dcnt = [0 for k in range(njoints)]
        self.w_mf_dcn = [0 for k in range(njoints)]
        self.w_mf_dcnt = [0 for k in range(njoints)]
        self.w_io_dcn = [0 for k in range(njoints)]
        self.w_io_dcnt = [0 for k in range(njoints)]

        #exc
        self.ltpPF_PC_max = 1 * math.pow(10, -4)         # LTP -2 #5 * math.pow(10, -4)
        self.ltdPF_PC_max = -1 * math.pow(10, -5)        # LTD -1 #-1 * math.pow(10, -5)
        #inh
        self.ltpPC_DCN_max = 1 * math.pow(10, -5)        # LTP -7 #4*math.pow(10, -5)
        self.ltdPC_DCN_max = 1 * math.pow(10, -4)       # LTD -6 #-9*math.pow(10, -4)
        #exc
        self.ltpMF_DCN_max = 1 * math.pow(10, -4)        # LTP -5 #3.2 * math.pow(10, -4)
        self.ltdMF_DCN_max = 1 * math.pow(10, -5)       # LTD -6 #-2.2 * math.pow(10, -5) 
        #exc
        self.ltpIO_DCN_max = 1 * math.pow(10, -3)        # LTP #1 * math.pow(10, -3)
        self.ltdIO_DCN_max = 1 * math.pow(10, -6)       # LTD #-1 * math.pow(10, -6)
        
        self.alphaPF_PC = 1000 #self.ltd_max / self.ltp_max
        self.alphaPC_DCN = 1 # self.ltd_PC_DCN_max / self.ltp_PC_DCN_max
        self.alphaMF_DCN = 1000 # self.ltd_PC_DCN_max / self.ltp_PC_DCN_max
        self.alphaIO_DCN = 1000
        #self.ind1 = 1
        #self.ind2 = 1

        #self.beta = 8 * math.pow(10, -3)
        self.output_ml  = [0 for k in range(self.njoints)]
        self.output_PC   = [0 for k in range(self.njoints)]
        self.output_DCN = [0 for k in range(self.njoints)]
        self.weights_mod = [0 for k in range(self.njoints)]

        for i in range(njoints):
            self.w_pf_pct[i]         = np.array([0])
            self.w_pf_pc[i]          = np.array([0])
            self.model               = LWPR(self.nin, self.nout)        # nout number of output of the LWPR model
            self.model.init_D        = 0.0008 * np.eye(self.nin) # 0.0002 (best PC)  #CRAWL 0.00001 # crawl solo 0.0001
            self.model.init_alpha    = 300 * np.ones([self.nin, self.nin])
            self.model.w_gen         = 0.5  # 0.1 default best
            self.model.diag_only     = bool(1)
            self.model.update_D      = bool(0)
            self.model.meta          = bool(0)
            self.model.meta_rate     = 0.3
            self.model.add_threshold = 0.95
            self.model.init_lambda = 0.99
            self.model.tau_lambda  = 0.9
            self.model.final_lambda = 0.99999
            # self.model[i].kernel      = 'Gaussian'


    def ML_prediction(self, inputlwpr, fbacktorq):      # inputlwpr array of inputs
            # LWPR
            (self.output_ml, self.weights_mod) = self.model.predict(inputlwpr)
            #print('lwprml', self.output_ml[i])
            for i in range(self.njoints):
                # PC
                if len(self.w_pf_pct[i]) != len(self.weights_mod):
                    self.w_pf_pc[i]  = np.resize(self.w_pf_pc[i], len(self.weights_mod))
                    self.w_pf_pct[i] = np.resize(self.w_pf_pct[i], len(self.weights_mod))
            
                # Learning PC
                #self.w_pf_pc[i] = self.w_pf_pct[i] + (self.beta * (fbacktorq[i] * self.weights_mod))
                self.w_pf_pc[i] = self.w_pf_pct[i] + ((self.ltpPF_PC_max / np.power((fbacktorq[i] + 1), self.alphaPF_PC)) - (self.ltdPF_PC_max * fbacktorq[i])) #* self.weights_mod
                self.w_pf_pct[i] = self.w_pf_pc[i]
                #Normalization
                if LA.norm(self.w_pf_pc[i]) !=0:
                    self.w_pf_pc[i] = self.w_pf_pct[i] / LA.norm(self.w_pf_pc[i])
                self.output_PC[i] = self.w_pf_pc[i] * np.matrix(self.weights_mod).T
            
                #print('w_pf_pc', self.w_pf_pc[i])
                #print('wT', np.matrix(self.weights_mod[i]).T)
                #print('C', ((self.ltpPF_PC_max / np.power((fbacktorq[i] + 1), self.alphaPF_PC)) - (self.ltdPF_PC_max * fbacktorq[i])))
                #print('output C', self.output_PC[i])
            
                #if (self.output_PC[i] != -1 and self.output_PC[i] != 0):
                # PC - DCN
                self.w_pc_dcn[i] = self.w_pc_dcnt[i] + (((self.ltpPC_DCN_max * np.power(self.output_PC[i], self.alphaPC_DCN)) * (1- (1/np.power((self.output_DCN[i] + 1), self.alphaPC_DCN)))) - (self.ltdPC_DCN_max * (1 - self.output_PC[i])))
                self.w_pc_dcnt[i] = self.w_pc_dcn[i]
                #print('wPCDCN', self.w_pc_dcn[i])
                #Normalization
                self.w_pc_dcn[i] = self.w_pc_dcnt[i] / LA.norm(self.w_pc_dcn)
                # MF - DCN
                self.w_mf_dcn[i] = self.w_mf_dcnt[i] + (self.ltpMF_DCN_max / np.power((self.output_PC[i] + 1), self.alphaMF_DCN) - (self.ltdMF_DCN_max * self.output_PC[i]))
                self.w_mf_dcnt[i] = self.w_mf_dcn[i]
                #Normalization
                #print('wMFDCN1', self.w_mf_dcn[i])
                self.w_mf_dcn[i] = self.w_mf_dcnt[i] / LA.norm(self.w_mf_dcn)
                #print('wMFDCN2', self.w_mf_dcn[i])
                # IO - DCN
                self.w_io_dcn[i] = self.w_io_dcnt[i] + (self.ltpIO_DCN_max * fbacktorq[i]) - (self.ltdIO_DCN_max / (np.power((fbacktorq[i]+1), self.alphaIO_DCN)))
                self.w_io_dcnt[i] = self.w_io_dcn[i]
                #Normalization
                self.w_io_dcn[i] = self.w_io_dcnt[i] / LA.norm(self.w_io_dcn)
                #DCN Output
                self.output_DCN[i] = self.w_mf_dcn[i] - (self.output_PC[i] * self.w_pc_dcn[i]) + (fbacktorq[i] * self.w_io_dcn[i])
            
            return self.output_ml, self.output_DCN, self.weights_mod

    def ML_update(self, inputlwpr, train_LWPRoutput):    # inputlwpr array of inputs   # trainlwpr array of train output
        #for i in range(self.njoints):
        self.model.update(inputlwpr, np.array([train_LWPRoutput]))


    def ML_rfs(self):
        #for i in range(self.njoints):
        print("rfs: ", self.model.num_rfs)
