__authord__ = 'Ismael Baira Ojeda'
__date__ = '23.03.2017'

from lwpr import LWPR
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
        self.w_pc_dcn = [0 for k in range(njoints)]
        self.w_mf_dcn = [0 for k in range(njoints)]

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
            self.wt[i] = np.array([0])
            self.w[i] = np.array([0])
            self.model[i] = LWPR(nin, 1)
            self.w_pc_dcn[i] = np.array([0])
            self.w_mf_dcn[i] = np.array([0])

            self.model[i].init_D = 0.00001 * np.eye(nin)
            self.model[i].init_alpha = 500 * np.ones([nin, nin])

            #self.model[i].init_lambda = 0.99
            #self.model[i].tau_lambda = 0.9
            #self.model[i].final_lambda = 0.99999
            #self.model[i].w_gen = 0.2
            self.model[i].diag_only = bool(1)
            self.model[i].update_D = bool(0)
            self.model[i].meta = bool(0)
            self.model[i].meta_rate = 0.3
            self.model[i].add_threshold = 0.95

    def ML_prediction(self, inputlwpr, fbacktorq):
        # inputlwpr array of inputs
        for i in range(self.njoints):
            # PC & MF
            (self.output_ml[i], self.weights_mod[i]) = self.model[i].predict(inputlwpr)

            if len(self.wt[i]) != len(self.weights_mod[i]):
                self.w[i] = np.resize(self.w[i], len(self.weights_mod[i]))
                self.wt[i] = np.resize(self.wt[i], len(self.weights_mod[i]))
            self.w[i] = self.wt[i] + (self.beta * (fbacktorq[i] * self.weights_mod[i]))

            self.output_C[i] = self.w[i] * np.matrix(self.weights_mod[i]).T
            self.wt[i] = self.w[i]

            # DCN
            self.output_DCN[i] = self.w_mf_dcn[i] - self.output_C[i] * self.w_pc_dcn[i] # w_mf_dcn[i]

            # Learning DCN
            if (self.output_C[i] != -1 and self.output_C[i] != 0):
                if(i==0):
                    self.w_pc_dcn[i] = self.w_pc_dcn[i] + self.ltp_max * np.power(self.output_C[i]/self.ind1, self.alpha) * (1 - (1/ np.power((self.output_DCN[i] + 1), self.alpha))) - (self.ltd_max * (1 - self.output_C[i]/self.ind1))
                    self.w_mf_dcn[i] = self.w_mf_dcn[i] + (self.ltp_max / np.power(self.output_C[i]/self.ind1 + 1, self.alpha)) - (self.ltd_max * self.output_C[i]/self.ind1)
                else:
                    self.w_pc_dcn[i] = self.w_pc_dcn[i] + self.ltp_max * np.power(self.output_C[i]/self.ind2, self.alpha) * (1 - (1/ np.power((self.output_DCN[i] + 1), self.alpha))) - (self.ltd_max * (1 - self.output_C[i]/self.ind2))
                    self.w_mf_dcn[i] = self.w_mf_dcn[i] + (self.ltp_max / np.power(self.output_C[i]/self.ind2 + 1, self.alpha)) - (self.ltd_max * self.output_C[i]/self.ind2)

        return self.output_ml, self.output_DCN  # self.output_C,


    def ML_update(self, inputlwpr, train_LWPRoutput):
        # inputlwpr array of inputs
        # trainlwpr array of train output
        # print(train_LWPRoutput)
        for i in range(self.njoints):
            self.model[i].update(inputlwpr, np.array([train_LWPRoutput[i]]))

    def ML_rfs(self):
        for i in range(self.njoints):
            print("rfs: ", self.model[i].num_rfs)
            #a = self.model.sub(1).rfs(0)
            return  self.model[i].num_rfs
