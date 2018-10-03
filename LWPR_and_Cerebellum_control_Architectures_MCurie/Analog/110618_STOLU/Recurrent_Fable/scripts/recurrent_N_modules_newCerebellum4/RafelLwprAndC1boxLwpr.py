__authord__ = 'Silvia Tolu'
__date__ = '20.07.2015'

from lwpr import *
import sys, time, random, math
import numpy as np
from numpy import linalg as LA
import scipy.io

class MLandC:

    # nin number of LWPR inputs
    # nout number of output of the LWPR model
    # njoints is number of links
    def __init__(self, nin_pos, nin_vel, nout, njoints, initd_p, alpha_p, initd_v, alpha_v):  # inicializa mandando el n de entradas y de articulaciones
        self.model_p = [0 for k in range(njoints)]
        self.model_v = [0 for k in range(njoints)]
        self.njoints = njoints
        self.nin_pos = nin_pos
        self.nin_vel = nin_vel
        self.nout = nout
        self.initd_p = initd_p
        self.alpha_p = alpha_p
        self.initd_v = initd_v
        self.alpha_v = alpha_v
        self.wt_p = [0 for k in range(nout)]
        self.w_p = [0 for k in range(nout)]
        self.w_pf_pct_p = [0 for k in range(self.nout)]
        self.w_pf_pc_p = [0 for k in range(self.nout)]
        self.w_pc_dcn_p = np.zeros((nout), dtype = np.double)
        self.w_mf_dcn_p = np.zeros((nout), dtype = np.double)
        self.w_pc_dcnt_p = np.zeros((nout), dtype = np.double)
        self.w_mf_dcnt_p = np.zeros((nout), dtype = np.double)
        self.w_io_dcn_p = np.zeros((nout), dtype = np.double)
        self.w_io_dcnt_p = np.zeros((nout), dtype = np.double)
        
        self.wt_v = [0 for k in range(nout)]
        self.w_v = [0 for k in range(nout)]
        self.w_pf_pct_v = [0 for k in range(self.nout)]
        self.w_pf_pc_v = [0 for k in range(self.nout)]
        self.w_pc_dcn_v = np.zeros((nout), dtype = np.double)
        self.w_mf_dcn_v = np.zeros((nout), dtype = np.double)
        self.w_pc_dcnt_v = np.zeros((nout), dtype = np.double)
        self.w_mf_dcnt_v = np.zeros((nout), dtype = np.double)
        self.w_io_dcn_v = np.zeros((nout), dtype = np.double)
        self.w_io_dcnt_v = np.zeros((nout), dtype = np.double)
        
        #exc
        #self.ltpPF_PC_max =  1.0 * math.pow(10, -3)   # LTP  #  3.0 * math.pow(10, -4)
        #self.ltdPF_PC_max =  1.0 * math.pow(10, -2)   # LTD  # -3.0 * math.pow(10, -3)
        #inh
        self.ltpPC_DCN_max_p = 1.0 * math.pow(10, -4)   # LTP  #  1.0 * math.pow(10, -4)
        self.ltdPC_DCN_max_p = 1.0 * math.pow(10, -3)   # LTD  #  1.0 * math.pow(10, -3) # never negative
        #exc
        self.ltpMF_DCN_max_p =  1.0 * math.pow(10, -4)  # LTP  #  1.0 * math.pow(10, -4)
        self.ltdMF_DCN_max_p =  -1.0 * math.pow(10, -3)  # LTD  #  -1.0 * math.pow(10, -3)
        #exc
        self.ltpIO_DCN_max_p =  1.2 * math.pow(10, -3)  # LTP  #  1.2 * math.pow(10, -3)
        self.ltdIO_DCN_max_p =  -1.1 * math.pow(10, -6)  # LTD  # -1.1 * math.pow(10, -6) # mantain the 1000 ratio in ltp/ltd
        
        self.ltpPC_DCN_max_v = 1.0 * math.pow(10, -5)   # LTP  #  1.0 * math.pow(10, -4)
        self.ltdPC_DCN_max_v = 1.0 * math.pow(10, -4)   # LTD  #  1.0 * math.pow(10, -3) # never negative
        #exc
        self.ltpMF_DCN_max_v =  1.0 * math.pow(10, -5)  # LTP  #  1.0 * math.pow(10, -4)
        self.ltdMF_DCN_max_v =  -1.0 * math.pow(10, -4)  # LTD  #  -1.0 * math.pow(10, -3)
        #exc
        self.ltpIO_DCN_max_v =  1.2 * math.pow(10, -3)  # LTP  #  1.2 * math.pow(10, -3)
        self.ltdIO_DCN_max_v =  -1.1 * math.pow(10, -6)  # LTD  # -1.1 * math.pow(10, -6) # mantain the 1000 ratio in ltp/ltd
        
        #self.alpha = 1
        #self.alphaPF_PC = 4  # self.ltd_max / self.ltp_max
        self.alphaPC_DCN_p = 10 # self.ltd_PC_DCN_max / self.ltp_PC_DCN_max
        self.alphaMF_DCN_p = 10 # self.ltd_PC_DCN_max / self.ltp_PC_DCN_max
        self.alphaIO_DCN_p = 10
        
        self.alphaPC_DCN_v = 10 # self.ltd_PC_DCN_max / self.ltp_PC_DCN_max
        self.alphaMF_DCN_v = 10 # self.ltd_PC_DCN_max / self.ltp_PC_DCN_max
        self.alphaIO_DCN_v = 10
        
        self.beta_p = 5 * math.pow(10, -3) #7 * math.pow(10, -3)
        self.beta_v = 5 * math.pow(10, -4) #7 * math.pow(10, -3)
        
        self.output_x_p = np.zeros((nout), dtype = np.double)
        self.output_x_v = np.zeros((nout), dtype = np.double)
        self.output_C_p = np.zeros((nout), dtype = np.double)
        self.output_C_v = np.zeros((nout), dtype = np.double)
        
        self.output_DCN_p = np.zeros((nout), dtype = np.double)
        self.output_DCN_v = np.zeros((nout), dtype = np.double)
        self.weights_mod_p = [0 for k in range(self.nout)] 
        self.weights_mod_v = [0 for k in range(self.nout)] 

        # self.wt = np.zeros((self.nout, 1), dtype = np.double)
        # self.w = np.zeros((self.nout,1), dtype = np.double)
        for i in range(nout):
            self.wt_p[i] = np.array([0])
            self.w_p[i] = np.array([0])
            self.wt_v[i] = np.array([0])
            self.w_v[i] = np.array([0])
            
        #self.w_pc_dcn = np.zeros((self.nout, 1), dtype = np.double)
        #self.w_mf_dcn = np.zeros((self.nout, 1), dtype = np.double)
        self.model_p = LWPR(self.nin_pos, nout)
        self.model_v = LWPR(self.nin_vel, nout)
        self.model_p.init_D = self.initd_p * np.eye(self.nin_pos) #6.5 5 rfs #(0.0055 - 15 rfs)
        self.model_v.init_D = self.initd_v * np.eye(self.nin_vel) #6.5 5 rfs #(0.0055 - 15 rfs)
        self.model_p.init_alpha = self.alpha_p * np.ones([self.nin_pos, self.nin_pos])  # 100 learning rate
        self.model_v.init_alpha = self.alpha_v * np.ones([self.nin_vel, self.nin_vel])  # 90 learning rate
        self.model_p.diag_only = bool(1)
        self.model_v.diag_only = bool(1)
        self.model_p.update_D = bool(0)
        self.model_v.update_D = bool(0)
        self.model_p.meta = bool(0)
        self.model_v.meta = bool(0)
        self.model_p.w_prune = 0.9
        #self.model_v.w_prune = 0.9
        self.model_p.w_gen = 0.5
        self.model_v.w_gen = 0.5
        self.model_p.meta_rate = 0.3
        self.model_v.meta_rate = 0.3
        #self.model_v.add_threshold = 0.95


    def ML_prediction_pos(self, inputlwpr, fbackpos, meanerrp, normp):
        # inputlwpr array of inputsc
        (self.output_x_p, self.weights_mod_p) = self.model_p.predict(inputlwpr)
        
        # print(self.output_x)
            # PC
        for i in range(self.nout):
            if len(self.wt_p[i]) != len(self.weights_mod_p):
                # print("weights: ",self.weights_mod)
                # print("w: ",self.w[i])
                self.w_p[i]  = np.resize(self.w_p[i], len(self.weights_mod_p))
                self.wt_p[i] = np.resize(self.wt_p[i], len(self.weights_mod_p))
                self.w_pf_pc_p[i]  = np.resize(self.w_pf_pc_p[i], len(self.weights_mod_p))
                self.w_pf_pct_p[i] = np.resize(self.w_pf_pct_p[i], len(self.weights_mod_p))
            
            # Learning PC
            # print("wt", self.wt)
            self.w_p[i] = self.wt_p[i] + (self.beta_p * (fbackpos[i] * self.weights_mod_p))
            self.output_C_p[i] = self.w_p[i] * np.matrix(self.weights_mod_p).T
            self.wt_p[i] = self.w_p[i]
            #self.w_pf_pc[i] = self.w_pf_pct[i] + ((self.ltpPF_PC_max / np.power((abs(fbackpos[i]) + 1), self.alphaPF_PC)) - (self.ltdPF_PC_max * fbackpos[i])) * self.weights_mod
            #self.w_pf_pc[i]  = self.w_pf_pct[i] + self.ltpPF_PC_max / (abs(fbackpos[i]) + 1.)**(self.alphaPF_PC) - self.ltdPF_PC_max*abs(fbackpos[i]) * self.weights_mod
            #self.w_pf_pct[i] = self.w_pf_pc[i]
            
            #if LA.norm(self.w[i]) !=0:
            #    self.w[i] = self.wt[i] / LA.norm(self.wt[i])
            #print('wpfpc',self.w_pf_pc[i])
            #print('norma_wpfpc',LA.norm(self.w_pf_pc[i]))
            #if LA.norm(self.w_pf_pc[i]) !=0:
            #    self.w_pf_pc[i] = self.w_pf_pct[i] / LA.norm(self.w_pf_pc[i])
            #print('wpfpc2',self.w_pf_pc[i])
            
            #self.output_C[i] = self.w[i] * np.matrix(self.weights_mod).T
            #self.output_C[i] = self.w_pf_pc[i] * np.matrix(self.weights_mod).T
            
            # DCN
            if (self.output_C_p[i] != -1 and self.output_C_p[i] != 0):
                #self.w_pc_dcn[i] = self.w_pc_dcnt[i] + self.ltpPC_DCN_max * np.power(self.output_C[i], self.alphaPC_DCN) * (1 - (1 / np.power((abs(self.output_DCN[i]) + 1), self.alphaPC_DCN))) - (self.ltdPC_DCN_max * (1 - self.output_C[i]))
                self.w_pc_dcn_p[i]  = self.w_pc_dcnt_p[i] + ( self.ltpPC_DCN_max_p*abs(self.output_C_p[i])**self.alphaPC_DCN_p )/( 1. + abs(self.output_DCN_p[i]) )**self.alphaPC_DCN_p - self.ltdPC_DCN_max_p*( abs( self.output_C_p[i] ) )#1. - self.output_pc[i])
            
                self.w_pc_dcnt_p[i] = self.w_pc_dcn_p[i]
                #Normalization
                #self.w_pc_dcn[i] = self.w_pc_dcnt[i] / LA.norm(self.w_pc_dcnt[i])
                self.w_mf_dcn_p[i] = self.w_mf_dcnt_p[i] + (self.ltpMF_DCN_max_p / np.power(abs(self.output_C_p[i]) + 1, self.alphaMF_DCN_p)) - (self.ltdMF_DCN_max_p * self.output_C_p[i])
                #self.w_mf_dcn[i]  = self.w_mf_dcnt[i] + self.ltpMF_DCN_max/( abs(self.output_C[i]) + 1.)**self.alphaMF_DCN - self.ltdMF_DCN_max*abs(self.output_C[i])
                self.w_mf_dcnt_p[i] = self.w_mf_dcn_p[i]
                #Normalization
                #print('wMFDCN1', self.w_mf_dcn[i])
                #self.w_mf_dcn[i] = self.w_mf_dcnt[i] / LA.norm(self.w_mf_dcnt[i])
                # IO - DCN
                self.w_io_dcn_p[i] = self.w_io_dcnt_p[i] + (self.ltpIO_DCN_max_p * abs(fbackpos[i])) - (self.ltdIO_DCN_max_p / (np.power((abs(fbackpos[i])+1), self.alphaIO_DCN_p)))
                # Normalization error p
                #self.w_io_dcn[i] = self.w_io_dcnt[i] + (self.ltpIO_DCN_max * normp[i]) - (self.ltdIO_DCN_max / (np.power((normp[i]+1), self.alphaIO_DCN)))
                self.w_io_dcnt_p[i] = self.w_io_dcn_p[i]
                #Normalization
                #self.w_io_dcn[i] = self.w_io_dcnt[i] / LA.norm(self.w_io_dcnt[i])
                #self.output_DCN_p[i] = (self.w_mf_dcn_p[i]) - (self.output_C_p[i] * self.w_pc_dcn_p[i]) + (meanerrp[i] * self.w_io_dcn_p[i])
                self.output_DCN_p[i] = (self.w_mf_dcn_p[i]) - (self.output_C_p[i] * self.w_pc_dcn_p[i]) + (fbackpos[i] * self.w_io_dcn_p[i])

        # print("outputC: ", self.output_C)
        #return self.output_x, self.output_C, self.output_DCN, self.model.init_D[0][0]
        return self.output_x_p, self.output_DCN_p, self.weights_mod_p, self.model_p.init_D[0][0]

    def ML_prediction_vel(self, inputlwpr, fbackvel, meanerrv, normv):
        # inputlwpr array of inputsc
        (self.output_x_v, self.weights_mod_v) = self.model_v.predict(inputlwpr)
        
        # print(self.output_x)
        # PC
        for i in range(self.nout):
            if len(self.wt_v[i]) != len(self.weights_mod_v):
                # print("weights: ",self.weights_mod)
                # print("w: ",self.w[i])
                self.w_v[i]  = np.resize(self.w_v[i], len(self.weights_mod_v))
                self.wt_v[i] = np.resize(self.wt_v[i], len(self.weights_mod_v))
                self.w_pf_pc_v[i]  = np.resize(self.w_pf_pc_v[i], len(self.weights_mod_v))
                self.w_pf_pct_v[i] = np.resize(self.w_pf_pct_v[i], len(self.weights_mod_v))
            
            # Learning PC
            self.w_v[i] = self.wt_v[i] + (self.beta_v * (fbackvel[i-int(self.nout/2)] * self.weights_mod_v))
            self.output_C_v[i] = self.w_v[i] * np.matrix(self.weights_mod_v).T
            self.wt_v[i] = self.w_v[i]
            #self.w_pf_pc[i] = self.w_pf_pct[i] + ((self.ltpPF_PC_max / np.power((abs(fbackvel[i-int(self.nout/2)]) + 1), self.alphaPF_PC)) - (self.ltdPF_PC_max * fbackvel[i-int(self.nout/2)])) * self.weights_mod
            #self.w_pf_pc[i]  = self.w_pf_pct[i] + self.ltpPF_PC_max / (abs(fbackvel[i-int(self.nout/2)] + 1.)**(self.alphaPF_PC) - self.ltdPF_PC_max*abs(fbackvel[i-int(self.nout/2)])) * self.weights_mod

            
            #self.w_pf_pct[i] = self.w_pf_pc[i]
            
            #if LA.norm(self.w[i]) !=0:
            #    self.w[i] = self.wt[i] / LA.norm(self.wt[i])
            #print('wpfpc',self.w_pf_pc[i])
            #print('norma_wpfpc',LA.norm(self.w_pf_pc[i]))
            #if LA.norm(self.w_pf_pc[i]) !=0:
            #    self.w_pf_pc[i] = self.w_pf_pct[i] / LA.norm(self.w_pf_pc[i])
            #print('wpfpc2',self.w_pf_pc[i])
            
            #self.output_C[i] = self.w[i] * np.matrix(self.weights_mod).T
            #self.output_C[i] = self.w_pf_pc[i] * np.matrix(self.weights_mod).T
            
            # DCN
            if (self.output_C_v[i] != -1 and self.output_C_v[i] != 0):
                #self.w_pc_dcn[i] = self.w_pc_dcnt[i] + self.ltpPC_DCN_max * np.power(self.output_C[i], self.alphaPC_DCN) * (1 - (1 / np.power((abs(self.output_DCN[i]) + 1), self.alphaPC_DCN))) - (self.ltdPC_DCN_max * (1 - self.output_C[i]))
                self.w_pc_dcn_v[i]  = self.w_pc_dcnt_v[i] + ( self.ltpPC_DCN_max_v*abs(self.output_C_v[i])**self.alphaPC_DCN_p )/( 1. + abs(self.output_DCN_v[i]) )**self.alphaPC_DCN_v - self.ltdPC_DCN_max_v*( abs( self.output_C_v[i] ) )#1. - self.output_pc[i])
            
                self.w_pc_dcnt_v[i] = self.w_pc_dcn_v[i]
                #Normalization
                #self.w_pc_dcn[i] = self.w_pc_dcnt[i] / LA.norm(self.w_pc_dcnt[i])
                self.w_mf_dcn_v[i] = self.w_mf_dcnt_v[i] + (self.ltpMF_DCN_max_v / np.power(abs(self.output_C_v[i]) + 1, self.alphaMF_DCN_v)) - (self.ltdMF_DCN_max_v * self.output_C_v[i])
                #self.w_mf_dcn[i]  = self.w_mf_dcnt[i] + self.ltpMF_DCN_max/( abs(self.output_C[i]) + 1.)**self.alphaMF_DCN - self.ltdMF_DCN_max*abs(self.output_C[i])
                self.w_mf_dcnt_v[i] = self.w_mf_dcn_v[i]
                #Normalization
                #print('wMFDCN1', self.w_mf_dcn[i])
                #self.w_mf_dcn[i] = self.w_mf_dcnt[i] / LA.norm(self.w_mf_dcnt[i])
                 # IO - DCN
                self.w_io_dcn_v[i] = self.w_io_dcnt_v[i] + (self.ltpIO_DCN_max_v * abs(fbackvel[i-int(self.nout/2)])) - (self.ltdIO_DCN_max_v / (np.power((abs(fbackvel[i-int(self.nout/2)])+1), self.alphaIO_DCN_v)))
                # Normalization error v
                #self.w_io_dcn[i] = self.w_io_dcnt[i] + (self.ltpIO_DCN_max * normv[i-int(self.nout/2)]) - (self.ltdIO_DCN_max / (np.power((normv[i-int(self.nout/2)]+1), self.alphaIO_DCN)))
                self.w_io_dcnt_v[i] = self.w_io_dcn_v[i]
                #Normalization
                #self.w_io_dcn[i] = self.w_io_dcnt[i] / LA.norm(self.w_io_dcn[i]) 
                self.output_DCN_v[i] = (self.w_mf_dcn_v[i]) - (self.output_C_v[i] * self.w_pc_dcn_v[i]) + (meanerrv[i-int(self.nout/2)] * self.w_io_dcn_v[i])
                #self.output_DCN_v[i] = (self.w_mf_dcn_v[i]) - (self.output_C_v[i] * self.w_pc_dcn_v[i]) + (fbackvel[i-int(self.nout/2)] * self.w_io_dcn_v[i])

        # print("outputC: ", self.output_C)
        #return self.output_x, self.output_C, self.output_DCN, self.model.init_D[0][0]
        return self.output_x_v, self.output_DCN_v, self.weights_mod_v, self.model_v.init_D[0][0]
    
    def ML_update_pos(self, inputlwpr, train_LWPRoutput):  # lo que tiene que aprender : 
        self.model_p.update(inputlwpr, train_LWPRoutput)
    
    def ML_update_vel(self, inputlwpr, train_LWPRoutput):  # lo que tiene que aprender : 
        self.model_v.update(inputlwpr, train_LWPRoutput)

    def ML_rfs(self):  
        print("rfs_p: ", self.model_p.num_rfs)
        print("rfs_v: ", self.model_v.num_rfs)
