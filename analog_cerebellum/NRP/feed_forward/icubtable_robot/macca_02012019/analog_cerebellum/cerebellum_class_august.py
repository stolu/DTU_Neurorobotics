# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 15:02:54 2018

@author: silvia-neurorobotics
"""
from lwpr import *
import sys, math
import numpy as np

from random import *
import math
sys.path.append("/usr/local/lib/python2.7/dist-packages")


#################################
R= Random()

def cross_2D(x1, x2):
   return max(math.exp(-10.0 * x1 * x1), math.exp(-50.0 * x2 * x2),
               1.25 * math.exp(-5.0*(x1 * x1 + x2 * x2)))
############################

n_uml = 2 # number of unite learning machine
n_out = 2 # number of output per uml
debug = 1
n_input_mossy = [3,3] 


n_input_pf = sum(n_input_mossy)*n_uml
input_mossy = [ [2.,4.,3.], [2.,4.,3.] ]
update_teach = 4.

init_D_mf=[0.5, 0.5]
init_alpha_mf = [500.,500.]
w_gen_mf = [0.6,0.6]
init_lambda_mf = [0.9,0.9]
tau_lambda_mf = [0.5,0.5]
final_lambda_mf = [0.995, 0.995]
w_prune_mf = [0.9,0.9]
meta_rate_mf = [0.3,0.3]
add_threshold_mf = [0.95,0.959]
kernel_mf = 'Gaussian'
init_D_pf=0.5
init_alpha_pf = 100.
w_gen_pf = 0.4

#print uml_model


################################################## INIT #############################
# create model

# PARALLEL FIBERS
#uml_pf = [0] # parallel fiber layer is unique and shared
uml_pf = LWPR(n_input_pf,n_out) # parallel fiber layer is unique and shared
uml_pf.init_D     = init_D_pf*np.eye(n_input_pf)  #50*np.eye(nin) 0.000055
uml_pf.init_alpha = init_alpha_pf*np.ones([n_input_pf, n_input_pf])
uml_pf.w_gen = w_gen_pf
uml_pf.diag_only = bool(1)  #1
uml_pf.update_D  = bool(0) #0
uml_pf.meta      = bool(0) #0
#uml_pf.init_lambda = 0.995
#uml_pf.tau_lambda = 0.5
#uml_pf.final_lambda = 0.9995
#uml_pf.w_prune= .9
#uml_pf.meta_rate = 0.3
#uml_pf.add_threshold = 0.95
#uml_pf.kernel = 'Gaussian'

# MOSSY FIBERS-GRANULAR
#input_mossy = []
uml_model = [0 for k in range(n_uml)]
for i in range(0,n_uml):
    #input_mossy.append([])
    uml_model[i] = LWPR(n_input_mossy[i],n_out)
    
    uml_model[i].init_D     = init_D_mf[i]*np.eye(n_input_mossy[i])  #50*np.eye(nin) 0.000055
    uml_model[i].init_alpha = init_alpha_mf[i]*np.ones([n_input_mossy[i], n_input_mossy[i]])
    
    uml_model[i].w_gen = w_gen_mf[i]
    uml_model[i].diag_only = bool(1)  #1
    uml_model[i].update_D  = bool(0) #0
    uml_model[i].meta      = bool(0) 
    #uml_model[i].init_lambda = init_lambda_mf[i]
    #uml_model[i].tau_lambda = tau_lambda_mf[i]
    #uml_model[i].final_lambda = final_lambda_mf[i]
    #uml_model[i].w_prune= w_prune_mf[i]
    #uml_model[i].meta_rate = meta_rate_mf[i]
    #uml_model[i].add_threshold = add_threshold_mf[i]
    #uml_model[i].kernel = kernel_mf     



def predict_model(uml_model,n_uml,input_mossy):
    for i in range(0,n_uml):
        d,f = uml_model[i].predict( np.array([ n for n in input_mossy[i] ]) )
        print d
        print f
        print("rfs "+str(uml_model[i].num_rfs[0]) )



def update_model(uml_model,n_uml,input_mossy, update_teach):  
    for i in range(0,n_uml):
        # input mossy is an array or list, update_teach is a float
        uml_model[i].update( np.array([ n for n in input_mossy[i] ]) , np.array([ update_teach , update_teach+3.]) )


# *************   MAIN  ***************

for t in range(0,1):
    for mossy in input_mossy:
        for inp in mossy:
            inp =R.uniform(-1, 1) + R.gauss(0, 0.3)
        #print inp
    update_teach = cross_2D(input_mossy[0][0], input_mossy[1][1]) + R.gauss(0, 0.1) 
    predict_model(uml_model, n_uml,input_mossy)
    update_model(uml_model,n_uml,input_mossy, update_teach)
    
from cerebellum_class import*

cereb = Cerebellum(n_uml,n_out,n_input_mossy,n_input_pf)
cereb.create_models()
