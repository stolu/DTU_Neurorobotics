__authord__ = 'Adrian Llopart'
__date__ = '12.07.2018'

from lwpr import *
import sys, time, random, math
import numpy as np
from numpy import linalg as LA
class MLandC:


    def __init__(self, nin, nout, njoints):  # inicializa mandando el n de entradas y de articulaciones
        self.model = [0 for k in range(njoints)]
        self.njoints = njoints
        self.nin = nin
        self.nout = nout
        self.wt = [0 for k in range(nout)]
        self.w = [0 for k in range(nout)]
        self.w_pf_pct = [0 for k in range(self.nout)]
        self.w_pf_pc = [0 for k in range(self.nout)]
        self.w_pc_dcn = np.zeros((nout), dtype = np.double)
        self.w_mf_dcn = np.zeros((nout), dtype = np.double)
        self.w_pc_dcnt = np.zeros((nout), dtype = np.double)
        self.w_mf_dcnt = np.zeros((nout), dtype = np.double)
        self.w_io_dcn = np.zeros((nout), dtype = np.double)
        self.w_io_dcnt = np.zeros((nout), dtype = np.double)
        
        
        
        #inh
        self.ltpPC_DCN_max = 1 * math.pow(10, -5)        # LTP -7 #4*math.pow(10, -5)-4
        self.ltdPC_DCN_max = 1 * math.pow(10, -4)       # LTD -6 #-9*math.pow(10, -4)-3
        #exc
        self.ltpMF_DCN_max = 1 * math.pow(10, -5)        # LTP -5 #3.2 * math.pow(10, -4)-4
        self.ltdMF_DCN_max = 1 * math.pow(10, -4)       # LTD -6 #-2.2 * math.pow(10, -5)-3
        #exc
        self.ltpIO_DCN_max = 1 * math.pow(10, -4)        # LTP #1 * math.pow(10, -4)
        self.ltdIO_DCN_max = 1 * math.pow(10, -6)       # LTD #-1 * math.pow(10, -6)
        
        #self.alpha = 1
        #self.alphaPF_PC = 1  #self.ltd_max / self.ltp_max
        self.alphaPC_DCN = 1. # self.ltd_PC_DCN_max / self.ltp_PC_DCN_max
        self.alphaMF_DCN = 1. # self.ltd_PC_DCN_max / self.ltp_PC_DCN_max
        self.alphaIO_DCN = 1.
        
        #self.beta = 1 * math.pow(10, -5) #5 * math.pow(10, -3)
        #self.beta = 1. * math.pow(10, -5)
        #self.beta = [5. * math.pow(10, -5), 5. * math.pow(10, -5), 5. * math.pow(10,-4), 1. * math.pow(10, -5), 1. * math.pow(10, -4), 1. * math.pow(10, -4)] 
        self.output_C = np.zeros((nout), dtype = np.double)
        self.output_DCN = np.zeros((nout), dtype = np.double)
        self.weights_mod = [0 for k in range(self.nout)] 

        # self.wt = np.zeros((self.nout, 1), dtype = np.double)
        # self.w = np.zeros((self.nout,1), dtype = np.double)
        for i in range(nout):
            self.wt[i] = np.array([0])
            self.w[i] = np.array([0])

        #self.w_pc_dcn = np.zeros((self.nout, 1), dtype = np.double)
        #self.w_mf_dcn = np.zeros((self.nout, 1), dtype = np.double)
        self.model = LWPR(nin, nout)  # nout = 1

        self.model.init_D = 0.5*np.eye(nin) #(0.09)
        self.model.init_alpha = 20*np.ones([nin, nin])  # learning rate

        self.model.diag_only = bool(1)
        self.model.update_D = bool(0)
        self.model.meta = bool(0)
        #self.model.meta_rate = 0.3
        #self.model.add_threshold = 0.95

        #self.model.w_gen = 0.5

        self.debug = 0        
        

    def ML_prediction(self, inputlwpr, fbacktorq):
        # inputlwpr array of inputs
        #print("\n ---> Inside ML_Prediction! <---")
        
        (self.output_x, self.weights_mod) = self.model.predict(inputlwpr)
        
        # print(self.output_x)
            # PC



        for i in range(self.nout):

        
            if len(self.wt[i]) != len(self.weights_mod):
                # print("weights: ",self.weights_mod)
                # print("w: ",self.w[i])
                self.w[i]  = np.resize(self.w[i], len(self.weights_mod))
                self.wt[i] = np.resize(self.wt[i], len(self.weights_mod))
                self.w_pf_pc[i]  = np.resize(self.w_pf_pc[i], len(self.weights_mod))
                self.w_pf_pct[i] = np.resize(self.w_pf_pct[i], len(self.weights_mod))
            

            self.w[i] = self.wt[i] + (self.beta * (fbacktorq[i] * self.weights_mod))
            self.wt[i] = self.w[i]

            #self.w_pf_pct[i] = self.w_pf_pc[i]
            #if LA.norm(self.w[i]) !=0:
            #    self.w[i] = self.wt[i] / LA.norm(self.wt[i])
            #print('wpfpc',self.w_pf_pc[i])
            #print('norma_wpfpc',LA.norm(self.w_pf_pc[i]))
            #if LA.norm(self.w_pf_pc[i]) !=0:
            #    self.w_pf_pc[i] = self.w_pf_pct[i] / LA.norm(self.w_pf_pc[i])
            #print('wpfpc2',self.w_pf_pc[i])
            self.output_C[i] = self.w[i] * np.matrix(self.weights_mod).T
            #self.output_C[i] = self.w_pf_pc[i] * np.matrix(self.weights_mod).T

            
            # DCN
            if (self.output_C[i] != -1 and self.output_C[i] != 0):
                self.w_pc_dcn[i] = self.w_pc_dcnt[i] + self.ltpPC_DCN_max * np.power(self.output_C[i], self.alphaPC_DCN) * (1 - (1/ np.power((self.output_DCN[i] + 1), self.alphaPC_DCN))) - (self.ltdPC_DCN_max * (1 - self.output_C[i]))
                self.w_pc_dcnt[i] = self.w_pc_dcn[i]

                #Normalization
                #self.w_pc_dcn[i] = self.w_pc_dcnt[i] / LA.norm(self.w_pc_dcnt[i])
                self.w_mf_dcn[i] = self.w_mf_dcnt[i] + (self.ltpMF_DCN_max / np.power(self.output_C[i] + 1, self.alphaMF_DCN)) - (self.ltdMF_DCN_max * self.output_C[i])
                self.w_mf_dcnt[i] = self.w_mf_dcn[i]
                #Normalization
                #print('wMFDCN1', self.w_mf_dcn[i])
                #self.w_mf_dcn[i] = self.w_mf_dcnt[i] / LA.norm(self.w_mf_dcnt[i])

                # IO - DCN
                self.w_io_dcn[i] = self.w_io_dcnt[i] + (self.ltpIO_DCN_max * fbacktorq[i]) - (self.ltdIO_DCN_max / (np.power((fbacktorq[i]+1), self.alphaIO_DCN)))
                self.w_io_dcnt[i] = self.w_io_dcn[i]

                #Normalization
                #self.w_io_dcn[i] = self.w_io_dcnt[i] / LA.norm(self.w_io_dcnt[i])
                #try:
                #    self.output_DCN[i] = (self.w_mf_dcn[i]) - (self.output_C[i] * self.w_pc_dcn[i]) + (mean_torq.data[i] * self.w_io_dcn[i])
                #except:

                self.output_DCN[i] = (self.w_mf_dcn[i]) - (self.output_C[i] * self.w_pc_dcn[i]) #+ (mean_torq * self.w_io_dcn[i])
                #self.output_DCN[i] = self.output_C[i] * self.w_pc_dcn[i] #+ (mean_torq * self.w_io_dcn[i])



            '''self.output_DCN[i] = self.w_mf_dcn[i] - self.output_C[i] * self.w_pc_dcn[i]
            # Learning DCN
            if (self.output_C[i] != -1 and self.output_C[i] != 0):
                self.w_pc_dcn[i] = self.w_pc_dcn[i] + self.ltp_max * np.power(self.output_C[i], self.alpha) * (1 - (1/ np.power((self.output_DCN[i] + 1), self.alpha))) - (self.ltd_max * (1 - self.output_C[i]))
                self.w_mf_dcn[i] = self.w_mf_dcn[i] + (self.ltp_max / np.power(self.output_C[i] + 1, self.alpha)) - (self.ltd_max * self.output_C[i]) '''

        # print("outputC: ", self.output_C)
        return self.output_x, self.output_C, self.output_DCN, self.model.init_D[0][0]

    def ML_update(self, inputlwpr, train_LWPRoutput):  
        #print("\n ---> Update! ")
        if self.debug == 1:
            print(" Update total control input"+str( np.array([train_LWPRoutput])))
        self.model.update(inputlwpr, np.array([train_LWPRoutput]) ) #train_LWPRoutput)
 
    def ML_rfs(self):  
            print("rfs: ", self.model.num_rfs)
            return #str(self.model.num_rfs)
   