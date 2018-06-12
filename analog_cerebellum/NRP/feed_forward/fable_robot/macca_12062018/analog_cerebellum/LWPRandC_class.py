__authord__ = 'Silvia Tolu'
__date__ = '20.07.2015'

from lwpr import *
import sys, time, random, math
import numpy as np
from numpy import linalg as LA
class MLandC:

    # nin number of LWPR inputs
    # nout number of output of the LWPR model
    # njoints is number of links
    def __init__(self, nin, njoints,nout):
        print("\n ---> INITTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT cerebellum in documentsT<---")
        self.model = [0 for k in range(njoints)]
        self.njoints = njoints
        self.nin = nin
        self.nout =  nout
        self.norm_out = (1.57*1)
        self.debug = 0
        self.wt = [0 for k in range(nout)] #(njoints)]
        self.w  = [0 for k in range(nout)] #(njoints)]
        
        self.beta = 7. * math.pow(10, -2) #5 * math.pow(10, -5)
        
        
        
        self.output_x    = np.zeros((nout), dtype = np.double)
        self.output_C    = np.zeros((nout), dtype = np.double)
        self.output_DCN  = np.zeros((nout), dtype = np.double)
        self.weights_mod = [0 for k in range(self.nout)] 
        
        self.w_pc_dcn = np.zeros((nout), dtype = np.double)
        self.w_pc_dcnt = np.zeros((nout), dtype = np.double)
        #inh
        self.ltpPC_DCN_max = 1. * math.pow(10, -5)        # LTP -7 #4*math.pow(10, -5)-4
        self.ltdPC_DCN_max = 1. * math.pow(10, -4)       # LTD -6 #-9*math.pow(10, -4)-3
        self.alphaPC_DCN = 1. # self.ltd_PC_DCN_max / self.ltp_PC_DCN_max
        for i in range(self.nout):#njoints):
            self.wt[i] = np.array([0])
            self.w[i]  = np.array([0])
            
        self.model            = LWPR(nin, self.nout)
        #self.model.init_D     = 0.1*np.eye(nin)  #50*np.eye(nin) 0.000055
        #self.model.init_alpha = 90.*np.ones([nin, nin])
        
        #self.model.w_gen = 0.3
        self.model.diag_only = bool(1)  #1
        self.model.update_D  = bool(0) #0
        self.model.meta      = bool(0) #0
        #self.model[i].init_lambda = 0.995
        #self.model[i].tau_lambda = 0.5
        #self.model[i].final_lambda = 0.9995
        #self.model.w_prune= .9
        #self.model.meta_rate = 0.3
        #self.model.add_threshold = 0.95
        # self.model[i].kernel = 'Gaussian'
        if self.debug == 1:
            print("\n init -- self.wt : "+str(self.wt)+"\n init -- self.w : "+str(self.w))+"\n init -- self.beta : "+str(self.beta)
            print("\n init self.model : "+str(self.model))
            print("\n init -- self.njoints : "+str(self.njoints)+"\n init -- self.nout : "+str(self.nin))+"\n init -- self.nin : "+str(self.nout)
            print("\n init -- self.model after lwpr: "+str(self.model )+"\n init -- self.model.init_D : "+str(self.model.init_D)+"\n init -- self.model.init_alpha : "+str(self.model.init_alpha))
        
            
            

    def ML_prediction(self, inputlwpr, fbacktorq):
        # inputlwpr array of inputs
        #print("\n ---> Inside ML_Prediction! <---")
        
        (self.output_x, self.weights_mod) = self.model.predict(inputlwpr)
        if self.debug == 1:
            print("\n ML_Prediction -- self.output_x : "+str(self.output_x)+"\n ML_Prediction -- self.weights_mod : "+str(self.weights_mod))
        
        for i in range(self.nout):
            if len(self.wt[i]) != len(self.weights_mod):
                self.w[i]  = np.resize(self.w[i],  len(self.weights_mod))
                self.wt[i] = np.resize(self.wt[i], len(self.weights_mod))
            
            # Learning PC
            # print("wt", self.wt)
            self.w[i] = self.wt[i] + (self.beta * (fbacktorq[i] * self.weights_mod))
                
            self.wt[i] = self.w[i]
            
            self.output_C[i] = self.w[i] * np.matrix(self.weights_mod).T
            # DCN
            if (self.output_C[i] != -1 and self.output_C[i] != 0):
                self.w_pc_dcn[i] = self.w_pc_dcnt[i] + self.ltpPC_DCN_max * np.power(self.output_C[i], self.alphaPC_DCN) * (1 - (1/ np.power((self.output_DCN[i] + 1), self.alphaPC_DCN))) - (self.ltdPC_DCN_max * (1 - self.output_C[i]))
                self.w_pc_dcnt[i] = self.w_pc_dcn[i]
                #Normalization
                #self.w_pc_dcn[i] = self.w_pc_dcnt[i] / LA.norm(self.w_pc_dcnt[i])
              

            self.output_DCN[i] = self.output_C[i] * self.w_pc_dcn[i]
        if self.debug == 1:    
            print("\n ML_Prediction -- self.w : "+str(self.w)+"\n ML_Prediction -- self.wt : "+str(self.wt))
            print("\n ML_Prediction -- self.output_C : "+str(self.output_C))
            
        return self.output_x, self.output_DCN #.output_C
        

    def ML_update(self, inputlwpr, train_LWPRoutput):  
        #print("\n ---> Update! ")
        if self.debug == 1:
            print(" Update total control input"+str( np.array([train_LWPRoutput])))
        self.model.update(inputlwpr, np.array([train_LWPRoutput]) ) #train_LWPRoutput)
 
    def ML_rfs(self):  
            print("rfs: ", self.model.num_rfs)
            return #str(self.model.num_rfs)
   