__authord__ = 'Silvia Tolu'
__date__ = '20.07.2015'

from lwpr import *
import sys, time, random, math
import numpy as np

class MLandC:
    # nin number of LWPR inputs
    # nout number of output of the LWPR model
    # njoints is number of links
    def __init__(self, nin, njoints):
        self.model = [0 for k in range(njoints)]
        self.njoints = njoints
        self.nin = nin
        self.wt = [0 for k in range(njoints)]
        self.w = [0 for k in range(njoints)]
        self.beta = 5 * math.pow(10, -2)
        self.output_ml = [0 for k in range(self.njoints)]
        self.output_C = [0 for k in range(self.njoints)]
        self.weights_mod = [0 for k in range(self.njoints)]

        for i in range(njoints):
            self.wt[i] = np.array([0])
            self.w[i] = np.array([0])
            self.model[i] = LWPR(nin, 1)  # nout = 1
            self.model[i].init_D = 0.0003 * np.eye(nin) #0.001
            self.model[i].init_alpha = 500 * np.ones([nin, nin])
            # self.model[i].init_lambda = 0.99
            # self.model[i].tau_lambda = 0.9
            # self.model[i].final_lambda = 0.99999
            self.model[i].w_gen = 0.5
            self.model[i].diag_only = bool(1)
            self.model[i].update_D = bool(0)
            self.model[i].meta = bool(0)
            self.model[i].meta_rate = 0.3
            self.model[i].add_threshold = 0.95
            # self.model[i].kernel = 'Gaussian'

    def ML_prediction(self, inputlwpr, fbacktorq):
        # inputlwpr array of inputs
        for i in range(self.njoints):
            (self.output_ml[i], self.weights_mod[i]) = self.model[i].predict(inputlwpr)
            # if len(self.weights_mod[i]) != 0 and (self.output_ml[i] != 0):
            if len(self.wt[i]) != len(self.weights_mod[i]):
                self.w[i] = np.resize(self.w[i], len(self.weights_mod[i]))
                self.wt[i] = np.resize(self.wt[i], len(self.weights_mod[i]))
            self.w[i] = self.wt[i] + (self.beta * (fbacktorq[i] * self.weights_mod[i]))
            self.output_C[i] = self.w[i] * np.matrix(self.weights_mod[i]).T
            self.wt[i] = self.w[i]
        return self.output_ml, self.output_C


    def ML_update(self, inputlwpr, train_LWPRoutput):
        # inputlwpr array of inputs
        # trainlwpr array of train output
        # print(train_LWPRoutput)
        for i in range(self.njoints):
            self.model[i].update(inputlwpr, np.array([train_LWPRoutput[i]]))

    def ML_rfs(self):
        for i in range(self.njoints):
            print("rfs: ", self.model[i].num_rfs)
