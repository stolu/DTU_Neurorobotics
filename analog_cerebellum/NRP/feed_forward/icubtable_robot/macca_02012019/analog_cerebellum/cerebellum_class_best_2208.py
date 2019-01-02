# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 13:03:03 2018

@author: silvia-neurorobotics
"""

import sys

sys.path.append("/usr/local/lib/python2.7/dist-packages")
import numpy as np
from random import *
from lwpr import *

class Cerebellum:
    
    
    def __init__(self,n_uml, n_out, n_input_mossy, n_input_pf):


        self.n_uml = n_uml # number of unite learning machine
        self.n_out_mf = n_out # number of output per uml
        self.n_out_pf =  self.n_uml
        self.debug_update = 0
        self.debug_prediction = 0
        self.rfs_print = 1
        # *** Teaching Signal ***
        self.min_teach = 0.
        self.max_teach = 0.
        self.tau_normalization = 1
        self.tau_norm_sign = 1
        self.mean_torq_dcn = 1
        
        
        self.output_DCN  = [0.001 for i in range( 0, self.n_uml)]#np.zeros((nout), dtype = np.double)
 
        # *** Mossy Fiber Learning Parameters ***
        self.uml_mossy        = []#0 for k in range(0,self.n_uml)]
        self.n_input_mossy    = n_input_mossy
        self.input_mossy      = []
        self.init_D_mf        = []
        self.init_alpha_mf    = []
        self.w_gen_mf = []
        self.init_lambda_mf   = []
        self.tau_lambda_mf    = []
        self.final_lambda_mf  = []
        self.w_prune_mf       = []
        self.meta_rate_mf     = []
        self.add_threshold_mf = []
        
        self.output_x_mossy    = []#np.zeros((nout), dtype = np.double)
        self.output_C_mossy    = [] # np.zeros((nout), dtype = np.double)
        self.wt_mossy = []
        self.w_mossy  = []
        self.weights_mod_mossy = []#[0 for k in range(self.nout)]
        for i in range(0,self.n_uml):
            self.init_D_mf.append(0.5)
            self.init_alpha_mf.append(90.)
            self.w_gen_mf.append(0.3)
            self.init_lambda_mf.append(0.9)
            self.tau_lambda_mf.append(0.5)
            self.final_lambda_mf.append(0.995)
            self.w_prune_mf.append(0.95)
            self.meta_rate_mf.append(0.3)
            self.add_threshold_mf.append(0.95)
        self.kernel_mf = 'Gaussian'
        self.diag_only_mf  = bool(1)  #1
        self.update_D_mf   = bool(0) #0
        self.meta_mf       = bool(0) #0
        self.beta_mf       = 7. * 10**(-3)
        
         
        
        
        # *** Parallel Fiber Learning Parameters ***
        
        self.n_input_pf    = n_input_pf
        self.input_pf      = [ 0. for n in range(self.n_input_pf) ]
        self.init_D_pf     = 0.5 
        self.init_alpha_pf = 100.
        self.w_gen_pf      = 0.4
        self.diag_only_pf  = bool(1)  #1
        self.update_D_pf   = bool(0) #0
        self.meta_pf       = bool(0) #0
        self.init_lambda_pf  = 0.9
        self.tau_lambda_pf   = 0.5
        self.final_lambda_pf = 0.995
        self.w_prune_pf      = 0.95
        self.meta_rate_pf    = 0.3
        self.add_threshold_pf = 0.95
        self.kernel_pf        = 'Gaussian'
        self.beta_pf = 7. * 10**(-3)
        self.wt_pf = [0 for k in range(self.n_out_pf)] 
        self.w_pf  = [0 for k in range(self.n_out_pf)] 
        self.weights_mod_pf = []
        self.output_x_pf    = [0 for k in range(self.n_out_pf)]#np.zeros((nout), dtype = np.double)
        self.output_pc    = [0 for k in range(self.n_out_pf)] # np.zeros((nout), dtype = np.double)
        self.IO_fbacktorq   = []
        # *** Synaptic weights ***

        self.w_pf_pct = []
        self.w_pf_pc = []
        self.w_pc_dcn = []
        self.w_pc_dcnt = []
        
        self.w_io_dcn = []
        self.w_io_dcnt = []

        self.w_mf_dcn = []
        self.w_mf_dcnt = []
        
        # *** Plasticity ***
        #exc
        self.ltpPF_PC_max = 1 * 10**(-4) 
        self.ltdPF_PC_max = 1 * 10**(-3) 
        #inh
        self.ltpPC_DCN_max = 1 * 10**(-4) 
        self.ltdPC_DCN_max = 1 * 10**(-3) 
        #exc
        self.ltpMF_DCN_max = 1 * 10**(-5) 
        self.ltdMF_DCN_max = 1 * 10**(-4) 
        #exc
        self.ltpIO_DCN_max = 1 * 10**(-4)
        self.ltdIO_DCN_max = 1 * 10**(-3) 
             
        #self.alpha = 1
        self.alphaPF_PC  = 1.  #self.ltd_max / self.ltp_max
        self.alphaPC_DCN = 1. # self.ltd_PC_DCN_max / self.ltp_PC_DCN_max
        self.alphaMF_DCN = 1. # self.ltd_PC_DCN_max / self.ltp_PC_DCN_max
        self.alphaIO_DCN = 1.     
        
        
    def create_models(self):
        
        #PARALLEL FIBERS
        #parallel fiber layer is unique and shared
        print("\n ** Creating Parallel Fibers Layer **")   
        self.uml_pf            = LWPR( self.n_input_pf, self.n_out_pf)
        self.uml_pf.init_D     = self.init_D_pf*np.eye(self.n_input_pf)
        self.uml_pf.init_alpha = self.init_alpha_pf*np.ones([self.n_input_pf, self.n_input_pf])
        self.uml_pf.w_gen      = self.w_gen_pf
        self.uml_pf.diag_only  = self.diag_only_pf 
        self.uml_pf.update_D   = self.update_D_pf
        self.uml_pf.meta       = self.meta_pf
        #self.uml_pf.init_lambda   = init_lambda_pf
        #self.uml_pf.tau_lambda    = tau_lambda_pf
        #self.uml_pf.final_lambda  = final_lambda_pf
        self.uml_pf.w_prune       = self.w_prune_pf
        #self.uml_pf.meta_rate     = meta_rate_pf
        #self.uml_pf.add_threshold = add_threshold_pf
        #self.uml_pf.kernel        = kernel_pf 
        
        #        MOSSY FIBERS
        #        Mossy fiber layer is specialized for each joint 
        print("\n ** Creating Mossy Fibers Layer **") 
        for i in range(0, self.n_uml):
            
            self.input_mossy.append([0.])
            self.w_mossy.append(np.array([0.]))
            self.wt_mossy.append(np.array([0.]))
            
            self.weights_mod_mossy.append( np.array([]))
            self.IO_fbacktorq.append(0.)
            # PF-PC synaptic weights
            # previous
            self.w_pf_pct.append(0.)
            # current
            self.w_pf_pc.append(0.)
            
            # PF-DCN synaptic weights
            # previous
            self.w_pc_dcnt.append(0.)
            # current
            self.w_pc_dcn.append(0.)
            
            # IO-DCN synaptic weights
            # previous
            self.w_io_dcnt.append(0.)
            # current
            self.w_io_dcn.append(0.)
            
            # MF-DCN synaptic weights
            # previous
            self.w_mf_dcnt.append(0.)
            # current
            self.w_mf_dcn.append(0.)
            
            
            self.output_x_mossy.append([]) #np.zeros((nout), dtype = np.double)
            self.output_C_mossy.append(0.) # np.zeros((nout), dtype = np.double)
            print("\ creating mossy fiber joint "+str(i))
            self.uml_mossy.append( LWPR( self.n_input_mossy[i], self.n_out_mf) )
            print("\ mossy fiber joint "+str(i)+" is created")
            print(self.uml_mossy[i])
            self.uml_mossy[i].init_D     = self.init_D_mf[i]*np.eye( self.n_input_mossy[i] )
            self.uml_mossy[i].init_alpha = self.init_alpha_mf[i]*np.ones([ self.n_input_mossy[i], self.n_input_mossy[i] ])            
            self.uml_mossy[i].w_gen      = self.w_gen_mf[i]
            self.uml_mossy[i].diag_only  = self.diag_only_mf
            self.uml_mossy[i].update_D   = self.update_D_mf
            self.uml_mossy[i].meta       = self.meta_mf
            #self.uml_mossy[i].init_lambda   = init_lambda_mf[i]
            #self.uml_mossy[i].tau_lambda    = tau_lambda_mf[i]
            #self.uml_mossy[i].final_lambda  = final_lambda_mf[i]
            #self.uml_mossy[i].w_prune       = w_prune_mf[i]
            #self.uml_mossy[i].meta_rate     = meta_rate_mf[i]
            #self.uml_mossy[i].add_threshold = add_threshold_mf[i]
            #self.uml_mossy[i].kernel        = kernel_mf   
        
    def update_models(self, update_teach_pf, update_teach_mf):
        
        if self.debug_update == 1:
            print(" \n Updating pf input "+str( np.array([ n for n in self.input_pf ])  )+" \n "+str(np.array([ teach for teach in update_teach_pf ])) )
        self.uml_pf.update( np.array([ n for n in self.input_pf ]) , np.array([ teach for teach in update_teach_pf ]) )
        if self.rfs_print == 1:
            print("\n rfs parallel fibers :"+str(self.uml_pf.num_rfs[0]) )
        if self.debug_update == 1:        
            print(" \n Updating pf ")
        for i in range(0, self.n_uml):
            if self.debug_update == 1:
                print(" \n Updating mf  input "+str(np.array([ n for n in self.input_mossy[i] ]) )+" \n " +str( np.array([ update_teach_mf[i] ])) )
            # input mossy is an array or list of arrays, update_teach is an array of n_out values
            self.uml_mossy[i].update( np.array([ n for n in self.input_mossy[i] ]) , np.array([ update_teach_mf[i] ]) )
            if self.rfs_print == 1:
                print("\n rfs Mossy fibers joint "+str(i)+": "+str(self.uml_mossy[i].num_rfs[0]) )
            if self.debug_update == 1:
                print(" \n Updating mf ")

    def saturate_signal(self,signal, min_bound, max_bound):
        if signal > max_bound:
            return max_bound
        elif signal < min_bound:
            return min_bound
        else:
            return signal
            
    def norma(self, signal, min_signal, max_signal, sign):
        # Normalization formula : sign*( max_des - min_des)*(x - x_min)/(x_max - x_min) + min_des
        if sign == 1:
            sign_signal = abs(signal)/signal
        else:
            sign_signal = 1.
        return sign_signal*(1. - 0.)*( self.saturate_signal(signal, min_signal, max_signal) - min_signal)/(max_signal - min_signal)



    def prediction(self, fbacktorq):   
        
        # Parallel Fiber prediction
        (self.output_x_pf, self.weights_mod_pf) = self.uml_pf.predict( np.array([ n for n in self.input_pf ]) )
        
        if self.debug_prediction == 1:
            print(" \n self.output_x_pf "+str(self.output_x_pf))
            print(" \n self.weights_mod_pf "+str( self.weights_mod_pf ))
        # Mossy fiber Prediction
        for i in range(0,self.n_uml):
            (self.output_x_mossy[i], self.weights_mod_mossy[i]) = self.uml_mossy[i].predict( np.array([ n for n in self.input_mossy[i] ]) )
            if self.debug_prediction == 1:
                print(" \n self.output_x_mf "+str(self.output_x_mossy))
                print(" \n self.weights_mod_mf "+str( self.weights_mod_mossy ))
            # this passage is to prevent errors when summing vectors later
            #if len(self.wt_pf[i]) != len(self.weights_mod_pf):

            #   self.w_pf[i]     = np.resize( self.w_pf[i],     len(self.weights_mod_pf))
            #   self.wt_pf[i]    = np.resize( self.wt[i],       len(self.weights_mod_pf))
                
                #self.w_mossy[i]  = np.resize( self.w_mossy[i],  len(self.weights_mod_mossy))
                #self.wt_mossy[i] = np.resize( self.wt_mossy[i], len(self.weights_mod_mossy))
                
                #self.w_pf_pc[i]  = np.resize( self.w_pf_pc[i],  len(self.weights_mod_pf))
                #self.w_pf_pct[i] = np.resize( self.w_pf_pct[i], len(self.weights_mod_pf))
                
        for i in range(0,self.n_uml):
            
            # Parallel Fibers - Prkinje with IO modulation
            if self.tau_normalization == 1:                
                self.IO_fbacktorq[i] = self.norma( abs(fbacktorq[i]) , self.min_teach,self.max_teach, self.tau_norm_sign)
            else:
                self.IO_fbacktorq[i] = fbacktorq[i] 
            if self.debug_prediction == 1:
                print(" \n self.IO_fbacktorq joint "+str(i)+" "+str(self.IO_fbacktorq[i]))
            
            self.w_pf_pc[i]  = self.w_pf_pct[i]  + self.ltpPF_PC_max /( abs(fbacktorq[i]) + 1.)**(self.alphaPF_PC) - self.ltdPF_PC_max*abs(fbacktorq[i])
#            self.w_pf_pc[i]  = self.w_pf_pct[i]  + self.ltpPF_PC_max /( self.IO_fbacktorq[i] + 1.)**(self.alphaPF_PC) - self.ltdPF_PC_max*self.IO_fbacktorq[i] 

            self.w_pf_pct[i]  = self.w_pf_pc[i]  
            if self.debug_prediction == 1:
                print(" \n self.w_pf_pc[i]  joint "+str(i)+" "+str( self.w_pf_pc[i]  ))
            
            # output signal PC
            self.output_pc[i]    = self.w_pf_pc[i]*self.output_x_pf[i] #*np.sum(self.weights_mod_pf)
            if self.debug_prediction == 1:
                print(" \n self.output_pc[i]  joint "+str(i)+" "+str( self.output_pc[i]  ))
            #if self.mean_torq_dcn == 1:
                #self.w_pf[i]    = self.wt_pf[i]    + ( self.beta_pf * mean_torq[i] * self.weights_mod_pf       )
                #self.w_mossy[i] = self.wt_mossy[i] + ( self.beta_mf * mean_torq[i] * self.weights_mod_mossy[i] )
            #else:
                #self.w_pf[i]    = self.wt[i]       + ( self.beta_pf * fbacktorq[i] * self.weights_mod_pf       )
                #self.w_mossy[i] = self.wt_mossy[i] + ( self.beta_mf * fbacktorq[i] * self.weights_mod_mossy[i] ) # this doesnt make sense
                
            #self.wt_pf[i]          = self.w_pf[i]
            #self.wt_mossy[i]       = self.w_mossy[i] 
            
            
            #self.output_C_pf[i]    = self.w_pf[i] * np.transpose( self.weights_mod_pf )#np.matrix( w for w in self.weights_mod[i]).T
            #self.output_C_mossy[i] = self.w_mossy[i] * np.transpose( self.weights_mod_mossy[i] ) #np.matrix(self.weights_mod_mossy).T # no sense!!!

            # PC-DCN - is excited by the normalized value of Pc output
            #norm_pc = self.norma( abs(self.output_pc[i]) , 0. , 10., self.tau_norm_sign)
            #if self.debug_prediction == 1:
                #print("\n norm_pc "+str(norm_pc))
            

            self.w_pc_dcn[i]  = self.w_pc_dcnt[i] + ( self.ltpPC_DCN_max*abs(self.output_pc[i])**self.alphaPC_DCN )/( 1. + abs(self.output_DCN[i]) )**self.alphaPC_DCN - self.ltdPC_DCN_max*( abs( self.output_pc[i] ) )#1. - self.output_pc[i])
            self.w_pc_dcnt[i] = self.w_pc_dcn[i]
            if self.debug_prediction == 1:
                print(" \n self.w_pc_dcn[i] "+str(self.w_pc_dcn[i] ))
            
            # MF - DCN
            
            self.w_mf_dcn[i]  = self.w_mf_dcnt[i] + self.ltpMF_DCN_max/( abs(self.output_pc[i]) + 1.)**self.alphaMF_DCN - self.ltdMF_DCN_max*abs(self.output_pc[i])
            self.w_mf_dcnt[i] = self.w_mf_dcn[i]
            if self.debug_prediction == 1:
                print(" \n self.w_mf_dcnt[i] "+str( self.w_mf_dcnt[i]))

            self.output_C_mossy[i] = self.w_mf_dcn[i]*self.output_x_mossy[i][0]#*np.sum(self.weights_mod_mossy[i])
            if self.debug_prediction == 1:
                print("\n self.output_C_mossy[i] "+str(self.output_C_mossy[i]))
            # DCN out            
            self.output_DCN[i] = self.output_C_mossy[i] - self.w_pc_dcn[i]*self.output_pc[i] #+ self.output_x_mossy[i][0]
            if self.debug_prediction == 1:
                print(" \n self.output_DCN[i] "+str( self.output_DCN[i]))
        return self.output_DCN
            #if (self.output_C_pf[i] != -1. and self.output_C_pf[i] != 0.):
                
            #    self.w_pc_dcn[i]  = self.w_pc_dcnt[i] + self.ltpPC_DCN_max * np.power(self.output_C_pf[i], self.alphaPC_DCN) * (1. - (1./ np.power( self.output_DCN[i] + 1., self.alphaPC_DCN) )) - (self.ltdPC_DCN_max * (1. - self.output_C_pf[i]) )
            #    self.w_pc_dcnt[i] = self.w_pc_dcn[i]

                #Normalization
                #self.w_pc_dcn[i] = self.w_pc_dcnt[i] / LA.norm(self.w_pc_dcnt[i])
                #self.w_mf_dcn[i] = self.w_mf_dcnt[i] + (self.ltpMF_DCN_max / np.power(self.output_C[i] + 1, self.alphaMF_DCN)) - (self.ltdMF_DCN_max * self.output_C[i])
           #     self.w_mf_dcn[i]  = self.w_mf_dcnt[i] + (self.ltpMF_DCN_max / np.power(self.output_C_mossy[i] + 1. , self.alphaMF_DCN) ) - (self.ltdMF_DCN_max * self.output_C_mossy[i])
           #     self.w_mf_dcnt[i] = self.w_mf_dcn[i]
            
            
#            if self.tau_normalization == 1:
#                if fbacktorq[i]> self.max_teach:
#                    fbacktorq[i] = 1.5
#                elif fbacktorq[i] < self.max_teach:
#                    fbacktorq[i] = - 1.5
#                if self.tau_norm_sign == 1:
#                    fbacktorq_sign = (fbacktorq[i]/abs(fbacktorq[i]))
#                else:
#                    fbacktorq_sign = 1. # (fbacktorq[i]/abs(fbacktorq[i]))
#                
#                # Normalization formula : sign*( max_des - min_des)*(x - x_min)/(x_max - x_min) + min_des
#                IO_fbacktorq = fbacktorq_sign*(1. - 0.)*(fbacktorq[i] - self.min_teach)/(self.max_teach - self.min_teach)
##                # IO - DCN
##                self.w_io_dcn[i] = self.w_io_dcnt[i] + (self.ltpIO_DCN_max * IO_fbacktorq) - (self.ltdIO_DCN_max / (np.power((IO_fbacktorq+1), self.alphaIO_DCN)))
##            else:
##                self.w_io_dcn[i] = self.w_io_dcnt[i] + (self.ltpIO_DCN_max * fbacktorq[i]) - (self.ltdIO_DCN_max / (np.power((fbacktorq[i]+1), self.alphaIO_DCN)))
##            
#            self.w_io_dcnt[i] = self.w_io_dcn[i]
#            
#            self.output_DCN[i] = (self.output_x_mossy[i][0]*self.w_mf_dcn[i]) - (self.output_C_pf[i] * self.w_pc_dcn[i]) + (mean_torq[i] * self.w_io_dcn[i]) + self.output_x_pf[i]
#
