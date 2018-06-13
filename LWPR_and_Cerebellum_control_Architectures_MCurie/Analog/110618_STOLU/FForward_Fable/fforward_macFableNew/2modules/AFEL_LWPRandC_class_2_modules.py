#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""============================================================
* File Name     : AFEL_circle_class_2modules.py
* Purpose       : MLandC class for modular configuration
* Creation Date : Mon 5 Dec 2016 18:00:05 PM CET
* Last Modified : Tue 06 Dec 2016 03:05:26 PM CET
__author__      = "Ismael Baira Ojeda"
__credits__     = ["Ismael Baira Ojeda", "Silvia Tolu"]
__maintainer__  = "Ismael Baira Ojeda"
__email__       = ["i.bairao@gmail.com", "iboj@elektro.dtu.dk"]
============================================================"""


from lwpr import *
import sys, time, random, math
import numpy as np



class MLandC:

    def __init__(self, nin, njoints):
        self.model = [0 for k in range(njoints)]
        self.njoints = njoints                               # njoints is number of links
        self.nin  = nin                                      # nin number of LWPR inputs

        self.wt = [0 for k in range(njoints)]
        self.w = [0 for k in range(njoints)]
        self.w_pc_dcn = [0 for k in range(njoints)]
        self.w_mf_dcn = [0 for k in range(njoints)]

        # self.ltp_max = math.pow(10, -5)        # math.pow(10, -5)
        # self.ltd_max = math.pow(10, -5)        # math.pow(10, -5)
        self.ltp_max = math.pow(10, -6)         # DCN freq
        self.ltd_max = math.pow(10, -6)         # DCN freq
        self.alpha = 1
        self.ind1 = 1
        self.ind2 = 1

        self.beta = 5 * math.pow(10, -2)
        self.output_ml  = [0 for k in range(self.njoints)]
        self.output_C   = [0 for k in range(self.njoints)]
        self.output_DCN = [0 for k in range(self.njoints)]
        self.weights_mod = [0 for k in range(self.njoints)]

        for i in range(njoints):
            self.wt[i]                  = np.array([0])
            self.w[i]                   = np.array([0])
            self.model[i]               = LWPR(nin, 1)                  # nout = 1, nout number of output of the LWPR model
            self.model[i].init_D        = 0.000001 * np.eye(nin)         # 0.0002 (best PC)  #CRAWL 0.00001 # crawl solo 0.0001
            self.model[i].init_alpha    = 500 * np.ones([nin, nin])
            self.model[i].w_gen         = 0.1  # 0.1 default best
            self.model[i].diag_only     = bool(1)
            self.model[i].update_D      = bool(0)
            self.model[i].meta          = bool(0)
            self.model[i].meta_rate     = 0.3
            self.model[i].add_threshold = 0.95
            # self.model[i].init_lambda = 0.99
            # self.model[i].tau_lambda  = 0.9
            # self.model[i].final_lambda = 0.99999
            # self.model[i].kernel      = 'Gaussian'


    def ML_prediction(self, inputlwpr, fbacktorq):      # inputlwpr array of inputs
        for i in range(self.njoints):
            # LWPR
            (self.output_ml[i], self.weights_mod[i]) = self.model[i].predict(inputlwpr)

            # PC
            if len(self.wt[i]) != len(self.weights_mod[i]):
                self.w[i]  = np.resize(self.w[i], len(self.weights_mod[i]))
                self.wt[i] = np.resize(self.wt[i], len(self.weights_mod[i]))
            # Learning PC
            self.w[i] = self.wt[i] + (self.beta * (fbacktorq[i] * self.weights_mod[i]))
            self.output_C[i] = self.w[i] * np.matrix(self.weights_mod[i]).T
            self.wt[i] = self.w[i]

            # DCN
            self.output_DCN[i] = self.w_mf_dcn[i] - self.output_C[i] * self.w_pc_dcn[i]
            # Learning DCN
            if (self.output_C[i] != -1 and self.output_C[i] != 0):
                if(i==0):    # BEST
                    self.w_pc_dcn[i] = self.w_pc_dcn[i] + self.ltp_max * np.power(self.output_C[i]/self.ind1, self.alpha) * (1 - (1/ np.power((self.output_DCN[i] + 1), self.alpha))) - (self.ltd_max * (1 - self.output_C[i]/self.ind1))
                    self.w_mf_dcn[i] = self.w_mf_dcn[i] + (self.ltp_max / np.power(self.output_C[i]/self.ind1 + 1, self.alpha)) - (self.ltd_max * self.output_C[i]/self.ind1)
                else:
                    self.w_pc_dcn[i] = self.w_pc_dcn[i] + self.ltp_max * np.power(self.output_C[i]/self.ind2, self.alpha) * (1 - (1/ np.power((self.output_DCN[i] + 1), self.alpha))) - (self.ltd_max * (1 - self.output_C[i]/self.ind2))
                    self.w_mf_dcn[i] = self.w_mf_dcn[i] + (self.ltp_max / np.power(self.output_C[i]/self.ind2 + 1, self.alpha)) - (self.ltd_max * self.output_C[i]/self.ind2)

        return self.output_ml, self.output_C, self.output_DCN



    def ML_update(self, inputlwpr, train_LWPRoutput):    # inputlwpr array of inputs   # trainlwpr array of train output
        for i in range(self.njoints):
            self.model[i].update(inputlwpr, np.array([train_LWPRoutput[i]]))


    def ML_rfs(self):
        for i in range(self.njoints):
            print("rfs: ", self.model[i].num_rfs)
