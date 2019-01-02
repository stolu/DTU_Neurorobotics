# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 13:03:03 2018

@author: silvia-neurorobotics
"""

import sys
sys.path.append("/usr/local/lib")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import numpy as np
from random import *
from lwpr import *

class Cerebellum:
    
    
    def __init__(self,n_uml, n_out, n_input_mossy, n_input_mossy_vel, n_input_pf):
        
        self.name = "Cerebellum"
        
        self.n_uml = n_uml # number of unite learning machine
        self.n_out_mf = n_out # number of output per uml
        self.n_out_pf =  self.n_uml
        self.debug_update = 0
        self.debug_prediction = 0
        self.rfs_print = 1
        # *** Teaching Signal ***
        self.min_teach_pos = [0. for i in range( 0, self.n_uml)]
        self.max_teach_pos = [np.pi for i in range( 0, self.n_uml)]
        
        self.min_teach_vel = [0. for i in range( 0, self.n_uml)]
        self.max_teach_vel = [np.pi for i in range( 0, self.n_uml)]
        
        self.min_signal_pos    = [0. for i in range( 0, self.n_uml)]       
        self.max_signal_pos    = [1.5 for i in range( 0, self.n_uml)]
        self.min_signal_vel    = [0. for i in range( 0, self.n_uml)]       
        self.max_signal_vel    = [1.5 for i in range( 0, self.n_uml)]
        
        self.signal_normalization = 0
        self.tau_norm_sign = 1
        self.IO_on = True
        self.PC_on = True
        self.MF_on = True
        
        
        
        self.output_DCN  = [0. for i in range( 0, self.n_uml)]#np.zeros((nout), dtype = np.double)
        self.output_IO_DCN  = [0. for i in range( 0, self.n_uml)]        
        self.output_IO_DCN_vel  = [0. for i in range( 0, self.n_uml)]  
        # *** Mossy Fiber Learning Parameters ***
        self.uml_mossy        = []#0 for k in range(0,self.n_uml)]
        self.uml_mossy_vel        = []#0 for k in range(0,self.n_uml)]
        self.n_input_mossy    = n_input_mossy
        self.n_input_mossy_vel    = n_input_mossy_vel
        self.input_mossy      = []
        self.input_mossy_vel      = []
        
        self.init_D_mf        = [ [], []]
        self.init_alpha_mf    = [ [], []]
        self.w_gen_mf = [ [], []]
        self.init_lambda_mf   = [ [], []]
        self.tau_lambda_mf    = [ [], []]
        self.final_lambda_mf  = [ [], []]
        self.w_prune_mf       = [ [], []]
        self.meta_rate_mf     = [ [], []]
        self.add_threshold_mf = [ [], []]
        
        self.output_x_mossy    = []#np.zeros((nout), dtype = np.double)
        self.output_C_mossy    = [] # np.zeros((nout), dtype = np.double)
        
        self.output_x_mossy_vel    = []#np.zeros((nout), dtype = np.double)
        self.output_C_mossy_vel    = [] # np.zeros((nout), dtype = np.double)
        
        self.wt_mossy = []
        self.w_mossy  = []
        self.weights_mod_mossy = []#[0 for k in range(self.nout)]
        
        self.wt_mossy_vel = []
        self.w_mossy_vel  = []
        self.weights_mod_mossy_vel = []#[0 for k in range(self.nout)]
        
        for q in range(0,2):
            for i in range(0,self.n_uml):
                self.init_D_mf[q].append(0.5)
                self.init_alpha_mf[q].append(90.)
                self.w_gen_mf[q].append(0.3)
                self.init_lambda_mf[q].append(0.9)
                self.tau_lambda_mf[q].append(0.5)
                self.final_lambda_mf[q].append(0.995)
                self.w_prune_mf[q].append(0.95)
                self.meta_rate_mf[q].append(0.3)
                self.add_threshold_mf[q].append(0.95)
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
        self.wt_pf = [0. for k in range(self.n_out_pf)] 
        self.w_pf  = [0. for k in range(self.n_out_pf)] 
        self.weights_mod_pf = []
        self.output_x_pf    = [0. for k in range(self.n_out_pf)]#np.zeros((nout), dtype = np.double)
        self.output_pc    = [0. for k in range(self.n_out_pf)] # np.zeros((nout), dtype = np.double)
        self.output_pc_vel    = [0. for k in range(self.n_out_pf)] # np.zeros((nout), dtype = np.double)
        self.input_pc_dcn_pos     = [0. for k in range(self.n_out_pf)]
        self.input_pc_dcn_vel     = [0. for k in range(self.n_out_pf)]
        self.output_pc_dcn_pos    = [0. for k in range(self.n_out_pf)]
        self.output_pc_dcn_vel    = [0. for k in range(self.n_out_pf)]
        self.output_pc_dcn_pos_norm    = [0. for k in range(self.n_out_pf)]
        self.output_pc_dcn_vel_norm    = [0. for k in range(self.n_out_pf)]        
        
        self.IO_fbacktorq   = []
        self.vel_IO_fbacktorq   = []
        self.IO_fbacktorq_t   = []
        self.vel_IO_fbacktorq_t   = []
        
        # *** Synaptic weights ***

        self.w_pf_pct = []
        self.w_pf_pc = []
        
        self.w_pf_pct_vel = []
        self.w_pf_pc_vel = []
        
        self.w_pc_dcn = []
        self.w_pc_dcnt = []
        
        self.w_pc_dcn_vel = []
        self.w_pc_dcnt_vel = []
        
        self.w_io_dcn = []
        self.w_io_dcnt = []
        
        self.w_io_dcn_vel = []
        self.w_io_dcnt_vel = []
        
        self.w_mf_dcn = []
        self.w_mf_dcnt = []
        
        self.w_mf_dcn_vel = []
        self.w_mf_dcnt_vel = []
        
        
        
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
        self.alphaPF_PC_vel = 1.
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
            self.input_mossy_vel.append([0.])
            
            self.w_mossy.append(np.array([0.]))
            self.wt_mossy.append(np.array([0.]))            
            self.weights_mod_mossy.append( np.array([]))

            self.w_mossy_vel.append(np.array([0.]))
            self.wt_mossy_vel.append(np.array([0.]))            
            self.weights_mod_mossy_vel.append( np.array([]))
            
            self.IO_fbacktorq.append(0.)
            self.IO_fbacktorq_t.append(0.)
            self.vel_IO_fbacktorq.append(0.)
            self.vel_IO_fbacktorq_t.append(0.)
            # PF-PC synaptic weights
            # previous
            self.w_pf_pct.append(0.)
            # current
            self.w_pf_pc.append(0.)
            self.w_pf_pct_vel.append(0.)
            # current
            self.w_pf_pc_vel.append(0.)
            
            # PF-DCN synaptic weights
            # previous
            self.w_pc_dcnt.append(0.)
            # current
            self.w_pc_dcn.append(0.)
            
            # previous
            self.w_pc_dcnt_vel.append(0.)
            # current
            self.w_pc_dcn_vel.append(0.)
            
            # IO-DCN synaptic weights
            # previous
            self.w_io_dcnt.append(0.)
            # current
            self.w_io_dcn.append(0.)
            #velocities 
            # previous
            self.w_io_dcnt_vel.append(0.)
            # current
            self.w_io_dcn_vel.append(0.)
            
            # MF-DCN synaptic weights
            # previous
            self.w_mf_dcnt.append(0.)
            # current
            self.w_mf_dcn.append(0.)
            
            # previous
            self.w_mf_dcnt_vel.append(0.)
            # current
            self.w_mf_dcn_vel.append(0.)
            
            
            self.output_x_mossy.append([]) #np.zeros((nout), dtype = np.double)
            self.output_C_mossy.append(0.) # np.zeros((nout), dtype = np.double)
            
            self.output_x_mossy_vel.append([]) #np.zeros((nout), dtype = np.double)
            self.output_C_mossy_vel.append(0.) # np.zeros((nout), dtype = np.double)
            
            print("\n "+self.name+" creating mossy fiber joint "+str(i))
            self.uml_mossy.append( LWPR( self.n_input_mossy[i], self.n_out_mf) )
            self.uml_mossy_vel.append( LWPR( self.n_input_mossy_vel[i], self.n_out_mf) )
            print("\n "+self.name+" mossy fiber joint "+str(i)+" is created")
            print(self.uml_mossy[i])
            print(self.uml_mossy_vel[i])
            
            self.uml_mossy[i].init_D     = self.init_D_mf[0][i]*np.eye( self.n_input_mossy[i] )
            self.uml_mossy[i].init_alpha = self.init_alpha_mf[0][i]*np.ones([ self.n_input_mossy[i], self.n_input_mossy[i] ])            
            self.uml_mossy[i].w_gen      = self.w_gen_mf[0][i]
            self.uml_mossy[i].diag_only  = self.diag_only_mf
            self.uml_mossy[i].update_D   = self.update_D_mf
            self.uml_mossy[i].meta       = self.meta_mf
            #self.uml_mossy[i].init_lambda   = init_lambda_mf[i]
            #self.uml_mossy[i].tau_lambda    = tau_lambda_mf[i]
            #self.uml_mossy[i].final_lambda  = final_lambda_mf[i]
            #self.uml_mossy[i].w_prune       = self.w_prune_mf[0][i]
            #self.uml_mossy[i].meta_rate     = meta_rate_mf[i]
            #self.uml_mossy[i].add_threshold = add_threshold_mf[i]
            #self.uml_mossy[i].kernel        = kernel_mf   
            
            self.uml_mossy_vel[i].init_D     = self.init_D_mf[1][i]*np.eye( self.n_input_mossy_vel[i] )
            self.uml_mossy_vel[i].init_alpha = self.init_alpha_mf[1][i]*np.ones([ self.n_input_mossy_vel[i], self.n_input_mossy_vel[i] ])            
            self.uml_mossy_vel[i].w_gen      = self.w_gen_mf[1][i]
            self.uml_mossy_vel[i].diag_only  = self.diag_only_mf
            self.uml_mossy_vel[i].update_D   = self.update_D_mf
            self.uml_mossy_vel[i].meta       = self.meta_mf
            #self.uml_mossy_vel[i].init_lambda   = init_lambda_mf[i]
            #self.uml_mossy_vel[i].tau_lambda    = tau_lambda_mf[i]
            #self.uml_mossy_vel[i].final_lambda  = final_lambda_mf[i]
            #self.uml_mossy_vel[i].w_prune       = self.w_prune_mf[1][i]
            #self.uml_mossy_vel[i].meta_rate     = meta_rate_mf[i]
            #self.uml_mossy_vel[i].add_threshold = add_threshold_mf[i]
            #self.uml_mossy_vel[i].kernel        = kernel_mf  
        
    def update_models(self, update_teach_pf, update_teach_mf):
        
        if self.debug_update == 1:
            print("\n "+self.name+" Updating pf input "+str( np.array([ n for n in self.input_pf ])  )+" \n "+str(np.array([ teach for teach in update_teach_pf ])) )
        self.uml_pf.update( np.array([ n for n in self.input_pf ]) , np.array([ teach for teach in update_teach_pf ]) )
        if self.rfs_print == 1:
            print("\n "+self.name+" rfs parallel fibers :"+str(self.uml_pf.num_rfs[0]) )
        if self.debug_update == 1:        
            print("\n "+self.name+" Updating pf ")
        #for i in range(0, self.n_uml):
        #    if self.debug_update == 1:
                #print("\n "+self.name+" Updating mf  input "+str(np.array([ n for n in self.input_mossy[i] ]) )+" \n " +str( np.array([ update_teach_mf[i] ])) )
                
            # input mossy is an array or list of arrays, update_teach is an array of n_out values
            #self.uml_mossy[i].update( np.array([ n for n in self.input_mossy[i] ]) , np.array([ update_teach_mf[i] ]) )
            #self.uml_mossy_vel[i].update( np.array([ n for n in self.input_mossy_vel[i] ]) , np.array([ update_teach_mf[i] ]) )
            
            #if self.rfs_print == 1:
                #print("\n "+self.name+" rfs Mossy fibers joint "+str(i)+": "+str(self.uml_mossy[i].num_rfs[0]) )
                #print("\n "+self.name+" rfs Mossy fibers vel joint "+str(i)+": "+str(self.uml_mossy_vel[i].num_rfs[0]) )
            #if self.debug_update == 1:
                #print("\n "+self.name+" Updating mf ")

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
            sign_signal = np.sign(signal)
        else:
            sign_signal = 1.
        return sign_signal*(1. - 0.)*( self.saturate_signal(signal, min_signal, max_signal) - min_signal)/(max_signal - min_signal)



    def prediction(self, fbacktorq, pos_error, vel_error):   
        
        # ----------------------------------------------------------------------------#
        # ----------------------- ***  Parallel Fiber prediction *** -----------------#
        # ----------------------------------------------------------------------------#
        
        (self.output_x_pf, self.weights_mod_pf) = self.uml_pf.predict( np.array([ n for n in self.input_pf ]) )
        
        if self.debug_prediction == 1:
            print("\n "+self.name+" self.output_x_pf "+str(self.output_x_pf))
            print("\n "+self.name+" self.weights_mod_pf "+str( self.weights_mod_pf ))

            
        
        # ----------------------------------------------------------------------------#
        # ------------------------- *** Mossy Fiber prediction *** -------------------#
        # ----------------------------------------------------------------------------#    
        #for i in range(0,self.n_uml):
            #(self.output_x_mossy[i], self.weights_mod_mossy[i]) = self.uml_mossy[i].predict( np.array([ n for n in self.input_mossy[i] ]) )
            #(self.output_x_mossy_vel[i], self.weights_mod_mossy_vel[i]) = self.uml_mossy_vel[i].predict( np.array([ n for n in self.input_mossy_vel[i] ]) )
            #if self.debug_prediction == 1:
            #    print("\n "+self.name+" self.output_x_mf "+str(self.output_x_mossy))
            #    print("\n "+self.name+" self.weights_mod_mf "+str( self.weights_mod_mossy ))
            #    print("\n "+self.name+" self.output_x_mf_vel "+str(self.output_x_mossy_vel))
            #    print("\n "+self.name+" self.weights_mod_mf _vel"+str( self.weights_mod_mossy_vel ))
        
        
   
        
        for i in range(0,self.n_uml):
            # ----------------------------------------------------------------------------#
            # ---------- *** Parallel Fibers - Purkinje with IO modulation *** -----------#
            # ----------------------------------------------------------------------------# 
            
            
            # ------------------ *** Inferior Olive *** -----------------------# 
            if self.signal_normalization == 1:                
                self.IO_fbacktorq[i] = ( self.norma( abs(fbacktorq[i]) , self.min_teach_pos[i], self.max_teach_pos[i] , self.tau_norm_sign) + self.IO_fbacktorq_t[i] ) /2. # media errore precedente e corrente
                self.IO_fbacktorq_t[i] = self.IO_fbacktorq[i]
                self.vel_IO_fbacktorq[i] =  (self.norma(  abs(vel_error[i]) , self.min_teach_vel[i] , self.max_teach_vel[i], self.tau_norm_sign) + self.vel_IO_fbacktorq_t[i] )/2.
                self.vel_IO_fbacktorq_t[i] = self.vel_IO_fbacktorq[i]
            else:
                self.IO_fbacktorq[i] = abs(pos_error[i])#abs(fbacktorq[i] )
                self.vel_IO_fbacktorq[i] = abs(vel_error[i]/10.)
            if self.debug_prediction == 1:
                print(" \n ****************** "+self.name+" Joint "+str(i)+" ****************** ")
                print(" \n self.IO_fbacktorq joint "+str(i)+" "+str(self.IO_fbacktorq[i]))
            
            
            
            # ------------------ *** Plasticity Pf-Pc *** -----------------------#
            self.w_pf_pc[i]  = self.w_pf_pct[i]  + self.ltpPF_PC_max /( self.IO_fbacktorq[i] + 1.)**(self.alphaPF_PC) - self.ltdPF_PC_max*self.IO_fbacktorq[i]
            self.w_pf_pct[i]  = self.w_pf_pc[i]  
            
            self.w_pf_pc_vel[i]  = self.w_pf_pct_vel[i]  + self.ltpPF_PC_max /( self.vel_IO_fbacktorq[i] + 1.)**(self.alphaPF_PC_vel) - self.ltdPF_PC_max*self.vel_IO_fbacktorq[i]
            self.w_pf_pct_vel[i]  = self.w_pf_pc_vel[i] 

            if self.debug_prediction == 1:
                print("\n "+self.name+" self.w_pf_pc[i]  joint "+str(i)+" "+str( self.w_pf_pc[i]  ))
                print("\n "+self.name+" self.w_pf_pc_vel[i]  joint "+str(i)+" "+str( self.w_pf_pc_vel[i]  ))
            
            # ------------------ *** output signal Purkinje Cell *** -----------------------#  
            self.output_pc[i]        = self.w_pf_pc[i]*self.output_x_pf[i] 
            self.output_pc_vel[i]    = self.w_pf_pc_vel[i]*self.output_x_pf[i]            
            
            if self.debug_prediction == 1:
                print("\n "+self.name+" self.output_pc[i]  joint "+str(i)+" "+str( self.output_pc[i]  ))
                print("\n "+self.name+" self.output_pc_vel[i]  joint "+str(i)+" "+str( self.output_pc_vel[i]  ))

            # ----------------------------------------------------------------------------#
            # -------------- *** Purkinje - Deep Cerebellar Nuclei *** -------------------#
            # ----------------------------------------------------------------------------#  
            
            if self.signal_normalization == 1: 
                self.input_pc_dcn_pos[i] = self.norma( abs( self.output_pc[i] )     , self.min_signal_pos[i] , self.max_signal_pos[i] , self.tau_norm_sign)
                self.input_pc_dcn_vel[i] = self.norma( abs( self.output_pc_vel[i] ) , self.min_signal_vel[i] , self.max_signal_vel[i] , self.tau_norm_sign)
            else:
                self.input_pc_dcn_pos[i] = abs(self.output_pc[i])
                self.input_pc_dcn_vel[i] = abs(self.output_pc_vel[i])
            
            # ----------------------- *** Plasticity Pc-DCN *** -------------------------# --> modulated by dcn(t-1) and pc
            self.w_pc_dcn[i]  = self.w_pc_dcnt[i] + ( self.ltpPC_DCN_max*(self.input_pc_dcn_pos[i]**self.alphaPC_DCN) )/( 1. + self.norma( abs( self.output_DCN[i] ) ,self.min_signal_pos[i] , self.max_signal_pos[i] , self.tau_norm_sign) )**self.alphaPC_DCN - self.ltdPC_DCN_max*( self.input_pc_dcn_pos[i]  )
            self.w_pc_dcnt[i] = self.w_pc_dcn[i]
            
            self.w_pc_dcn_vel[i]  = self.w_pc_dcnt_vel[i] + ( self.ltpPC_DCN_max*(self.input_pc_dcn_vel[i]**self.alphaPC_DCN) )/( 1. + self.norma( abs( self.output_DCN[i] ) ,self.min_signal_vel[i] , self.max_signal_vel[i] , self.tau_norm_sign) )**self.alphaPC_DCN - self.ltdPC_DCN_max*( self.input_pc_dcn_vel[i]  )#1. - self.output_pc[i])
            self.w_pc_dcnt_vel[i] = self.w_pc_dcn_vel[i]
            
            # ------------- *** Input signal from Purkinje Cell to DCN ***----------------#
            self.output_pc_dcn_pos[i] = self.output_pc[i]*self.w_pc_dcn[i]           
            self.output_pc_dcn_vel[i] = self.output_pc_vel[i]*self.w_pc_dcn_vel[i]
            
            if self.debug_prediction == 1:
                print("\n "+self.name+" self.w_pc_dcn[i] "+str(self.w_pc_dcn[i] ))
            
            
            # ----------------------------------------------------------------------------#
            # ------------- *** Mossy Fibers - Deep Cerebellar Nuclei *** ----------------#
            # ----------------------------------------------------------------------------#            
            
            if self.signal_normalization == 1: 
                self.output_pc_dcn_pos_norm[i] = self.norma( abs( self.output_pc[i] )     , self.min_signal_pos[i] , self.max_signal_pos[i] , self.tau_norm_sign)
                self.output_pc_dcn_vel_norm[i] = self.norma( abs( self.output_pc_vel[i] ) , self.min_signal_vel[i] , self.max_signal_vel[i] , self.tau_norm_sign)
            else:
                self.output_pc_dcn_pos_norm[i] = abs(self.output_pc[i])
                self.output_pc_dcn_vel_norm[i] = abs(self.output_pc_vel[i])
            
            # ------------------------ *** Plasticity MF-DCN *** -------------------------#    
            self.w_mf_dcn[i]  = self.w_mf_dcnt[i] + self.ltpMF_DCN_max/( self.output_pc_dcn_pos_norm[i]  + 1.)**self.alphaMF_DCN - self.ltdMF_DCN_max*self.output_pc_dcn_pos_norm[i] 
            self.w_mf_dcnt[i] = self.w_mf_dcn[i]
            
            self.w_mf_dcn_vel[i]  = self.w_mf_dcnt_vel[i] + self.ltpMF_DCN_max/( self.output_pc_dcn_vel_norm[i]  + 1.)**self.alphaMF_DCN - self.ltdMF_DCN_max*self.output_pc_dcn_vel_norm[i] 
            self.w_mf_dcnt_vel[i] = self.w_mf_dcn_vel[i]
            
            if self.debug_prediction == 1:
                print("\n "+self.name+" self.w_mf_dcn[i] "+str( self.w_mf_dcn[i]))
                print("\n "+self.name+" self.w_mf_dcn_vel[i] "+str( self.w_mf_dcn_vel[i]))
            
            # --------------------- *** Input signal from MF to DCN ***--------------------#
            self.output_C_mossy[i] = self.w_mf_dcn[i]*self.input_mossy[i]#self.output_x_mossy[i][0]
            self.output_C_mossy_vel[i] = self.w_mf_dcn_vel[i]*self.input_mossy_vel[i]#self.output_x_mossy_vel[i][0]

            if self.debug_prediction == 1:
                print("\n "+self.name+" self.output_C_mossy[i] "+str(self.output_C_mossy[i]))
            
            
            # ----------------------------------------------------------------------------#
            # ------------- *** Inferior Olive - Deep Cerebellar Nuclei *** --------------#
            # ----------------------------------------------------------------------------#            
            
            # ------------------------ *** Plasticity IO-DCN *** -------------------------#             
            self.w_io_dcn[i] = self.w_io_dcnt[i] + self.ltpIO_DCN_max*self.IO_fbacktorq[i] - self.ltdIO_DCN_max/( (self.IO_fbacktorq[i] + 1.)**self.alphaIO_DCN  ) 
            self.w_io_dcnt[i] = self.w_io_dcn[i]
            
            self.w_io_dcn_vel[i] = self.w_io_dcnt_vel[i] + self.ltpIO_DCN_max*self.vel_IO_fbacktorq[i] - self.ltdIO_DCN_max/( (self.vel_IO_fbacktorq[i] + 1.)**self.alphaIO_DCN  )    
            self.w_io_dcnt_vel[i] = self.w_io_dcn_vel[i]
            
            # --------------------- *** Input signal from IO to DCN ***--------------------#
            self.output_IO_DCN[i] =  self.w_io_dcn[i]*self.IO_fbacktorq[i]
            self.output_IO_DCN_vel[i] =  self.w_io_dcn_vel[i] *self.vel_IO_fbacktorq[i]
            
            if self.debug_prediction == 1:
                print("\n "+self.name+" self.w_io_dcn[i] "+str( self.w_io_dcn[i] ))
                print("\n "+self.name+" self.output_IO_DCN[i] "+str( self.output_IO_DCN[i] ))
            
            
            # ----------------------------------------------------------------------------#
            # ----------------------- *** Deep Cerebellar Nuclei *** ---------------------#
            # ----------------------------------------------------------------------------#            
            # DCN = - (PC_pos + PC_vel) + (MF_pos + MF_vel) + (IO_pos + IO_vel)            
            #self.output_DCN[i] =  - (self.w_pc_dcn[i]*self.output_pc[i] + self.w_pc_dcn_vel[i]*self.output_pc_vel[i] ) + (self.output_C_mossy[i]  +  self.output_C_mossy_vel[i]) +self.output_IO_DCN[i] +self.output_IO_DCN_vel[i]# self.output_x_mossy[i][0]  #+ self.output_x_mossy[i][0]
            #self.output_DCN[i] =  - (self.output_pc_dcn_pos[i] + self.output_pc_dcn_vel[i] ) + (self.output_C_mossy[i]  +  self.output_C_mossy_vel[i]) +self.output_IO_DCN[i] +self.output_IO_DCN_vel[i]
            self.output_DCN[i] =  [ 0. , -self.output_pc_dcn_pos[i] - self.output_pc_dcn_vel[i] ][self.PC_on == True]  + [ 0., self.output_C_mossy[i]  +  self.output_C_mossy_vel[i] ][ self.MF_on == True ] + [ 0. , self.output_IO_DCN[i] +self.output_IO_DCN_vel[i] ][ self.IO_on == True ]
            
            if self.debug_prediction == 1:
                print("\n "+self.name+" self.output_DCN[i] "+str( self.output_DCN[i]))
        
        return self.output_DCN
