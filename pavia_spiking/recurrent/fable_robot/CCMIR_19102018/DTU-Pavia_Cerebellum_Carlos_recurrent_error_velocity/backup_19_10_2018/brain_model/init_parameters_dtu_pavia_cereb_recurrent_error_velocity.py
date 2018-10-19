import os

########################################################################
########################################################################
########################################################################
#INPUT THE PATH WHERE ALL YOUR TESTS RESULTS ARE LOCATED
RESULTS_PATH = "/home/dtu-neurorobotics/Documents/workspace/Projects/DTU-Pavia_Cerebellum_Carlos_recurrent_error_velocity/cereb_tuning/results"
#INPUT THE NEW TEST NUMBER FOR THE NEW TEST
TEST = 10


#THIS PATH HAS TO BE MANUALLY CREATED; THIS IS TO ENSURE THAT FILES ARE NOT OVERWRITTEN 
#GO TO RESULTS_PATH AND CREATE A DIRECTORY NAMED TEST (where TEST is the number of the test)
OUTPUT_PATH = RESULTS_PATH + "/" +str(TEST) + "/"
########################################################################
########################################################################
########################################################################



############ RECORDING CELLS ############
RECORDING_CELLS = True
############ NUMBER OF MODULES, JOINTS AND INPUT_MF ############
n_modules   = 1
n_joints    = 2
n_inputs_MF = 3

############ NUMBER OF NEURONS ############
MF_n  = 120#300		#24#240   # multiple of n_inputs_MF
GC_n  = 750#6000	#150#1500  # multiple of n_joints
IO_n  = 24 #72		#6#48	  # multiple of n_joints
PC_n  = 24 #72		#6#48	  # multiple of n_joints
DCN_n = 12 #36		#4#24	  # multiple of n_joints

############ NEURONS PARAMETERS ############
# V_m double - Membrane potential in mV 
# E_L double - Leak reversal potential in mV. 
# C_m double - Capacity of the membrane in pF 
# t_ref double - Duration of refractory period in ms. 
# V_th double - Spike threshold in mV. 
# V_reset double - Reset potential of the membrane in mV. 
# E_ex double - Excitatory reversal potential in mV. 
# E_in double - Inhibitory reversal potential in mV. 
# g_L double - Leak conductance in nS; 
# tau_syn_ex double - Time constant of the excitatory synaptic exponential function in ms. 
# tau_syn_in double - Time constant of the inhibitory synaptic exponential function in ms. 
# I_e double - Constant external input current in pA. 

#### MF_p ####
MF_p = {'t_ref'       :  1.0,
		'C_m'	 	  :  2.0,
		'V_th' 	      : -40.0,
		'V_reset' 	  : -70.0,
		'g_L'		  :  0.2,
		'tau_syn_ex'  :  0.5,
		'tau_syn_in'  :  10.0}
#### GC_p ####
GC_p = {'t_ref'       :  1.0,
		'C_m'	 	  :  2.0,
		'V_th' 	      : -40.0,
		'V_reset' 	  : -70.0,
		'g_L'		  :  0.2,
		'tau_syn_ex'  :  0.5,
		'tau_syn_in'  :  10.0}
#### IO_p ####
IO_p = {'t_ref'       :  1.0,
		'C_m'	 	  :  2.0,
		'V_th' 	      : -40.0,
		'V_reset' 	  : -70.0,
		'g_L'		  :  0.2,
		'tau_syn_ex'  :  0.5,
		'tau_syn_in'  :  10.0}
#### PC_p ####
PC_p ={'t_ref'        :  2.0,
	   'C_m'	 	  :  400.0,
	   'V_th' 	      : -52.0,
	   'V_reset' 	  : -70.0,
	   'g_L'		  :  16.0,
	   'tau_syn_ex'   :  0.5,
	   'tau_syn_in'   :  1.6}

#### DCN_p ####
DCN_p ={'t_ref'       :  1.0,
		'C_m'	 	  :  2.0,
		'V_th' 	      : -40.0,
		'V_reset' 	  : -70.0,
		'g_L'		  :  0.2,
		'tau_syn_ex'  :  0.5,
		'tau_syn_in'  :  10.0}

############ SYNAPSES PARAMETERS ############
PLAST1_sinexp   = True # Set to True in order to have sinexp plasticity in PF
#### MF_GC_p ####
MF_GC_p = {'weight'   :  0.00024, #NOT FROM PAVIA -> 0.75 GR fire at 7 Hz ####0.3,#1.5e-4, #0.625
		   'delay'    :  1.0}
#### GC_PC_p ####
if PLAST1_sinexp:
	GC_PC_p = {}
	for module in range(n_modules):
		for joint in range(n_joints):
			if joint==0:
				GC_PC_p['weight'+"_"+str(module)+"_"+str(joint)+"_q"]  =  20.0
				GC_PC_p['delay'+"_"+str(module)+"_"+str(joint)+"_q"]   =  1.0
				GC_PC_p['A_minus'+"_"+str(module)+"_"+str(joint)+"_q"] =  -0.35    #LTD double - Amplitude of weight change for depression  # double - Amplitude of weight change for depression
				GC_PC_p['A_plus'+"_"+str(module)+"_"+str(joint)+"_q"]  =  0.03#0.0242   #0.015,  #LTP double - Amplitude of weight change for potentiation # double - Amplitude of weight change for facilitation 
				GC_PC_p['Wmin'+"_"+str(module)+"_"+str(joint)+"_q"]    =  0.0    # double - Minimal synaptic weight 
				GC_PC_p['Wmax'+"_"+str(module)+"_"+str(joint)+"_q"]    =  40.0   # double - Maximal synaptic weight
				GC_PC_p['stdp_delay'+"_"+str(module)+"_"+str(joint)+"_q"]    =  150.0   # double - Maximal synaptic weight
				
				GC_PC_p['weight'+"_"+str(module)+"_"+str(joint)+"_qd"]  =  20.0
				GC_PC_p['delay'+"_"+str(module)+"_"+str(joint)+"_qd"]   =  1.0
				GC_PC_p['A_minus'+"_"+str(module)+"_"+str(joint)+"_qd"] =  -0.1#-0.38    #LTD double - Amplitude of weight change for depression  # double - Amplitude of weight change for depression
				GC_PC_p['A_plus'+"_"+str(module)+"_"+str(joint)+"_qd"]  =  0.06#0.07#0.0242   #0.015,  #LTP double - Amplitude of weight change for potentiation # double - Amplitude of weight change for facilitation 
				GC_PC_p['Wmin'+"_"+str(module)+"_"+str(joint)+"_qd"]    =  0.0    # double - Minimal synaptic weight 
				GC_PC_p['Wmax'+"_"+str(module)+"_"+str(joint)+"_qd"]    =  40.0   # double - Maximal synaptic weight
				GC_PC_p['stdp_delay'+"_"+str(module)+"_"+str(joint)+"_qd"]    =  150.0   # double - Maximal synaptic weight
				
				
			elif joint==1:
				GC_PC_p['weight'+"_"+str(module)+"_"+str(joint)+"_q"]  =  20.0
				GC_PC_p['delay'+"_"+str(module)+"_"+str(joint)+"_q"]   =  1.0
				GC_PC_p['A_minus'+"_"+str(module)+"_"+str(joint)+"_q"] =  -0.35#-0.22#-0.42    #LTD double - Amplitude of weight change for depression  # double - Amplitude of weight change for depression
				GC_PC_p['A_plus'+"_"+str(module)+"_"+str(joint)+"_q"]  =  0.03#0.03  #0.015,  #LTP double - Amplitude of weight change for potentiation # double - Amplitude of weight change for facilitation 
				GC_PC_p['Wmin'+"_"+str(module)+"_"+str(joint)+"_q"]    =  0.0    # double - Minimal synaptic weight 
				GC_PC_p['Wmax'+"_"+str(module)+"_"+str(joint)+"_q"]    =  40.0   # double - Maximal synaptic weight
				GC_PC_p['stdp_delay'+"_"+str(module)+"_"+str(joint)+"_q"]    =  150.0   # double - Maximal synaptic weight
				
				GC_PC_p['weight'+"_"+str(module)+"_"+str(joint)+"_qd"]  =  20.0
				GC_PC_p['delay'+"_"+str(module)+"_"+str(joint)+"_qd"]   =  1.0
				GC_PC_p['A_minus'+"_"+str(module)+"_"+str(joint)+"_qd"] =  -0.1#-0.42    #LTD double - Amplitude of weight change for depression  # double - Amplitude of weight change for depression
				GC_PC_p['A_plus'+"_"+str(module)+"_"+str(joint)+"_qd"]  =  0.065#0.07  #0.015,  #LTP double - Amplitude of weight change for potentiation # double - Amplitude of weight change for facilitation 
				GC_PC_p['Wmin'+"_"+str(module)+"_"+str(joint)+"_qd"]    =  0.0    # double - Minimal synaptic weight 
				GC_PC_p['Wmax'+"_"+str(module)+"_"+str(joint)+"_qd"]    =  40.0   # double - Maximal synaptic weight
				GC_PC_p['stdp_delay'+"_"+str(module)+"_"+str(joint)+"_qd"]    =  150.0   # double - Maximal synaptic weight
				
else:
	GC_PC_p = {'weight'   :  0.025,
		   	   'delay'    :  1.0}


#### IO_PC_p ####
IO_PC_p = {'weight'   :  1.0,#10.0, #1.0
		   'delay'    :  1.0}  #0.1
#### PC_DCN_p ####
PC_DCN_p = {}
for module in range(n_modules):
	for joint in range(n_joints):
		if joint==0:
			PC_DCN_p["PC_DCN_p"+"_"+str(module)+"_"+str(joint)+"_q"] = {'weight'  :  0.0000207,#000016,#0.000015557,
										'delay'   :  1.0}  #0.1
                        PC_DCN_p["PC_DCN_p"+"_"+str(module)+"_"+str(joint)+"_qd"] = {'weight'  :  0.0000207,#000016,#0.000015557,
										'delay'   :  1.0}  #0.1
		elif joint==1:
			PC_DCN_p["PC_DCN_p"+"_"+str(module)+"_"+str(joint)+"_q"] = {'weight'  :  0.0000201,#0.000015557,
										'delay'   :  1.0}  #0.1
                        PC_DCN_p["PC_DCN_p"+"_"+str(module)+"_"+str(joint)+"_qd"] = {'weight'  :  0.0000201,#0.000015557,
										'delay'   :  1.0}  #0.1
#### MF_DCN_p ####
MF_DCN_p = {}
for module in range(n_modules):
	for joint in range(n_joints):
		if joint==0:
			MF_DCN_p["MF_DCN_p"+"_"+str(module)+"_"+str(joint)] = {'weight'  :  0.000029,#0.000028,
                                                                                'delay'   :  10.0}  #0.1
		elif joint==1:
			MF_DCN_p["MF_DCN_p"+"_"+str(module)+"_"+str(joint)] = {'weight'  :  0.0000299,#0.000027,
										'delay'   :  10.0}  #0.1

################################################################