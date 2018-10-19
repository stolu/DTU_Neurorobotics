########################################################### 
			 # DTU SPIKING CEREB MODEL V2.0 #
########################################################### 

# -*- coding: utf-8 -*-
"""
Loads a Spiking Network described in a SNN file

"""

__author__ = 'Carlos Corchado Miralles'


# -----------------------------------------------------------------------#
# ** libraries **
#from hbp_nrp_cle.brainsim import simulator as sim
import pyNN.nest as sim
import numpy as np
import logging
import pickle
import os

import sys
sys.path.append("/home/dtu-neurorobotics/Documents/NRP/Models/brain_model/DTU_spiking_cereb_hybrid")
#from init_parameters_dtu_pavia_cereb import*

sim.setup(timestep=0.1, min_delay=0.1, max_delay=1001.0, threads=8, rng_seeds=[1234])
# 8 threads: recomended number. Reference: http://www.nest-simulator.org/wp-content/uploads/2015/02/NEST_by_Example.pdf

# Remapping variables to allow reloading of module by importing it whole (Easier for development)
import init_parameters_dtu_pavia_cereb_hybrid as ip
reload(ip)
# Map variables:

MF_n = ip.MF_n
GC_n = ip.GC_n
IO_n = ip.IO_n
PC_n = ip.PC_n
DCN_n = ip.DCN_n

n_inputs_MF = ip.n_inputs_MF

# -----------------------------------------------------------------------#
# ** global variables  **
#global MF_n, GC_n, IO_n, PC_n, DCN_n

try:
	sim.nest.Install("albertomodule")
except Exception as e:
	print(str(e))


sim.nest.SetKernelStatus({"overwrite_files": True,  "data_path": ip.OUTPUT_PATH})

msd = 1000 # master seed
n_vp = sim.nest.GetKernelStatus('total_num_virtual_procs')
msdrange1 = range(msd, msd+n_vp )
pyrngs = [np.random.RandomState(s) for s in msdrange1]
msdrange2=range(msd+n_vp+1, msd+1+2*n_vp)
sim.nest.SetKernelStatus({'grng_seed' : msd+n_vp,
					  'rng_seeds' : msdrange2})
# -----------------------------------------------------------------------#
# ** Cerebellum NN **
def create_cerebellum():
	"""
	Initializes PyNN with the neuronal network
	"""

	#################### NEURON MODEL DEFINITIONS ########################################
	# Number of neurons in each layer per muscle --> 4 joints to command
	# MF: Mossy Fibers
	# GC: Granular Cells --> in common among the 4 cerebellar modules
	# IO: Inferior Olives
	# PC: Purkinje Cells
	# DCN: Deep cerebellar nuclei

	########################################################
	########################################################
	########################################################
	########################################################
	
	"""
														 error signal
															   |
															   |
															   |	
															  IO 
															   |
															   |
														teaching signal
															   +          
															   +          
															   +          
	 GC-------------excit, plast:stdp_sin_exp-------------->>>>PC         
	 ^                                                         |          
	 ^                                                         |          
	 ^                                                         |          
	 |                                                         |          
	 |                                                         |          
   excit                                                     inhib        
	 |                                                         *          
	 |                                                         *          
	 |                                                         *                    
	 |                                                         *          
	 MF-------------------------excit---------------------->>>>DCN  
	 ^
	 ^
	 ^
	 |
	 |
	 |
	 desire and actual states
	 
	 """
	########################################################
	########################################################
	########################################################
	########################################################
	
	# global definition

	logger = logging.getLogger(__name__) 

	global MF_n, GC_n, IO_n, PC_n, DCN_n


	#################### NEURONS DEFINITIONS ########################################
	# type of neurons for each population
	MF_t  = sim.native_cell_type('iaf_cond_exp')
	GC_t  = sim.native_cell_type('iaf_cond_exp')
	IO_t  = sim.native_cell_type('iaf_cond_exp')
	PC_t  = sim.native_cell_type('iaf_cond_exp')#_cs')
	DCN_t = sim.native_cell_type('iaf_cond_exp')
	vt_t  = sim.native_cell_type('volume_transmitter_alberto')

	MF_r   = {}
	GC_r   = {}
	IO   = {}
	PC_r   = {}
	DCN_r  = {}
	vt_r   = {}

	MF_ff   = {}
	GC_ff   = {}
	PC_ff   = {}
	DCN_ff  = {}
	vt_ff   = {}

	################
	###### MF ######
	################
	for module in range(ip.n_modules):
		for joint in range(ip.n_joints):
			#Create independent subgroups of Volume Transmitter
			#Recurrent
			vt_r["vt_"+str(module)+"_"+str(joint)+"_pos"] = sim.create(vt_t, {}, PC_n/2)
			vt_r["vt_"+str(module)+"_"+str(joint)+"_neg"] = sim.create(vt_t, {}, PC_n/2)
			#Feedforward
			vt_ff["vt_"+str(module)+"_"+str(joint)+"_pos"] = sim.create(vt_t, {}, PC_n/2)
			vt_ff["vt_"+str(module)+"_"+str(joint)+"_neg"] = sim.create(vt_t, {}, PC_n/2)

			vti_r = {}
			vti_ff = {}
			for P in range(PC_n/2):
				#Recurrent
				vti_r["vti_"+str(module)+"_"+str(joint)+"_pos"] = sim.PopulationView(vt_r["vt_"+str(module)+"_"+str(joint)+"_pos"], np.array([P]))
				vti_r["vti_"+str(module)+"_"+str(joint)+"_neg"] = sim.PopulationView(vt_r["vt_"+str(module)+"_"+str(joint)+"_neg"], np.array([P]))
					
				sim.nest.SetStatus(tuple(vti_r["vti_"+str(module)+"_"+str(joint)+"_pos"].all_cells), {"vt_num": P})
				sim.nest.SetStatus(tuple(vti_r["vti_"+str(module)+"_"+str(joint)+"_neg"].all_cells), {"vt_num": P})

				#Feedforward
				vti_ff["vti_"+str(module)+"_"+str(joint)+"_pos"] = sim.PopulationView(vt_ff["vt_"+str(module)+"_"+str(joint)+"_pos"], np.array([P]))
				vti_ff["vti_"+str(module)+"_"+str(joint)+"_neg"] = sim.PopulationView(vt_ff["vt_"+str(module)+"_"+str(joint)+"_neg"], np.array([P]))
				
				sim.nest.SetStatus(tuple(vti_ff["vti_"+str(module)+"_"+str(joint)+"_pos"].all_cells), {"vt_num": P})
				sim.nest.SetStatus(tuple(vti_ff["vti_"+str(module)+"_"+str(joint)+"_neg"].all_cells), {"vt_num": P})

			#Create independent subgroups of MFs for the different types of inputs
			#Recurrent
			MF_r["MF_"+str(module)+"_"+str(joint)+"_cur_q"]  = sim.create(MF_t, ip.MF_p, ip.MF_n/n_inputs_MF) # Mossy Fiber module "module" joint "joint", current position 
			MF_r["MF_"+str(module)+"_"+str(joint)+"_des_q"]  = sim.create(MF_t, ip.MF_p, ip.MF_n/n_inputs_MF) # Mossy Fiber module "module" joint "joint", desired position
			MF_r["MF_"+str(module)+"_"+str(joint)+"_des_torque"]  = sim.create(MF_t, ip.MF_p, ip.MF_n/n_inputs_MF) # Mossy Fiber module "module" joint "joint", desired torque
			
			#Feedforward
			MF_ff["MF_"+str(module)+"_"+str(joint)+"_des_q"]  = sim.create(MF_t, ip.MF_p, ip.MF_n/n_inputs_MF) # Mossy Fiber module "module" joint "joint", desired position
			MF_ff["MF_"+str(module)+"_"+str(joint)+"_des_qd"] = sim.create(MF_t, ip.MF_p, ip.MF_n/n_inputs_MF) # Mossy Fiber module "module" joint "joint", desired velocity
			MF_ff["MF_"+str(module)+"_"+str(joint)+"_cur_q"]  = sim.create(MF_t, ip.MF_p, ip.MF_n/n_inputs_MF) # Mossy Fiber module "module" joint "joint", current position
			
			
			#Create the final MF population (combination the subgroups)
			#Recurrent
			MF_r["MF_"+str(module)+"_"+str(joint)] = MF_r["MF_"+str(module)+"_"+str(joint)+"_cur_q"] + MF_r["MF_"+str(module)+"_"+str(joint)+"_des_q"] + MF_r["MF_"+str(module)+"_"+str(joint)+"_des_torque"]
			#Feedforward
			MF_ff["MF_"+str(module)+"_"+str(joint)] = MF_ff["MF_"+str(module)+"_"+str(joint)+"_des_q"] + MF_ff["MF_"+str(module)+"_"+str(joint)+"_des_qd"] + MF_ff["MF_"+str(module)+"_"+str(joint)+"_cur_q"]

	################
	###### GC ######
	################
	#Recurrent
	GC_r["GC"] = sim.create(GC_t, ip.GC_p, GC_n)
	#Feedforward
	GC_ff["GC"] = sim.create(GC_t, ip.GC_p, GC_n)

	################   ################   #################
	###### IO ######   ###### PC ######   ###### DCN ######
	################   ################   #################

	for module in range(ip.n_modules):
		for joint in range(ip.n_joints):
			#Create independent subgroups of IOs for the +|- errors
			IO["IO_"+str(module)+"_"+str(joint)+"_pos"] = sim.create(IO_t, ip.IO_p, IO_n/2) # Agonist torque contribution
			IO["IO_"+str(module)+"_"+str(joint)+"_neg"] = sim.create(IO_t, ip.IO_p, IO_n/2) # Antagonist torque contribution
			
			#Create independent subgroups of PCs for the +|- errors
			#Recurrent
			PC_r["PC_"+str(module)+"_"+str(joint)+"_pos"] = sim.create(PC_t, ip.PC_p, PC_n/2) # Agonist torque contribution
			PC_r["PC_"+str(module)+"_"+str(joint)+"_neg"] = sim.create(PC_t, ip.PC_p, PC_n/2) # Antagonist torque contribution
			#Feedforward
			PC_ff["PC_"+str(module)+"_"+str(joint)+"_pos"] = sim.create(PC_t, ip.PC_p, PC_n/2) # Agonist torque contribution
			PC_ff["PC_"+str(module)+"_"+str(joint)+"_neg"] = sim.create(PC_t, ip.PC_p, PC_n/2) # Antagonist torque contribution
			
			#Create independent subgroups of DCNs for the +|- errors
			#Recurrent
			DCN_r["DCN_"+str(module)+"_"+str(joint)+"_pos"] = sim.create(DCN_t, ip.DCN_p, DCN_n/2) # Agonist torque contribution
			DCN_r["DCN_"+str(module)+"_"+str(joint)+"_neg"] = sim.create(DCN_t, ip.DCN_p, DCN_n/2) # Antagonist torque contribution
			#Feedforward
			DCN_ff["DCN_"+str(module)+"_"+str(joint)+"_pos"] = sim.create(DCN_t, ip.DCN_p, DCN_n/2) # Agonist torque contribution
			DCN_ff["DCN_"+str(module)+"_"+str(joint)+"_neg"] = sim.create(DCN_t, ip.DCN_p, DCN_n/2) # Antagonist torque contribution

			#Create the final IO population (combination of the subgroups)
			IO["IO_"+str(module)+"_"+str(joint)] = IO["IO_"+str(module)+"_"+str(joint)+"_pos"] + IO["IO_"+str(module)+"_"+str(joint)+"_neg"]
			
			#Create the final PC population (combination of the subgroups)
			#Recurrent
			PC_r["PC_"+str(module)+"_"+str(joint)] = PC_r["PC_"+str(module)+"_"+str(joint)+"_pos"] + PC_r["PC_"+str(module)+"_"+str(joint)+"_neg"]
			#Feedforward
			PC_ff["PC_"+str(module)+"_"+str(joint)] = PC_ff["PC_"+str(module)+"_"+str(joint)+"_pos"] + PC_ff["PC_"+str(module)+"_"+str(joint)+"_neg"]

			#Create the final DCN population (combination of the subgroups)
			#Recurrent
			DCN_r["DCN_"+str(module)+"_"+str(joint)] = DCN_r["DCN_"+str(module)+"_"+str(joint)+"_pos"] + DCN_r["DCN_"+str(module)+"_"+str(joint)+"_neg"]
			#Feedforward
			DCN_ff["DCN_"+str(module)+"_"+str(joint)] = DCN_ff["DCN_"+str(module)+"_"+str(joint)+"_pos"] + DCN_ff["DCN_"+str(module)+"_"+str(joint)+"_neg"]

	#################### SYNAPSES DEFINITIONS ########################################
	# Types of synapses
	# MF-GC: excitatory, each GC receives inputs from just a few MF
	# GC-PC: excitatory, each GC has potential to form few synapses with a large number of PC
	# IO-PC: teaching signal, each PC receives one input from IO    
	# PC-DCN: inhibitory,  
	# MF-DCN: excitatory,   
	

	# synapse parameters
	# PCReceptor = {'AMPA': 1, 'GABA': 2, 'COMPLEX_SPIKE' : 3}
	
	# connection definition
	MF_GC_c  = sim.FixedNumberPreConnector(4, allow_self_connections=False) #each GC receives inputs from just a few MF (select number, now it is 4)
	if ip.PLAST1_sinexp:
		GC_PC_c  = {'rule': 'fixed_indegree', 'indegree': int(0.8*GC_n), "multapses": False}
	else:
		GC_PC_c  = sim.FixedNumberPreConnector(int(0.8*GC_n), allow_self_connections=False) #Each PC receives around 80% of GC
	IO_PC_c  = sim.OneToOneConnector()
	PC_DCN_c = sim.OneToOneConnector() #when doing the projection, 2 PC mapped to 1 DCN, select one to one for every half
	MF_DCN_c = sim.AllToAllConnector()
	

	# synapses definition
	MF_GC_r  = sim.StaticSynapse(**ip.MF_GC_p_r)
	MF_GC_ff  = sim.StaticSynapse(**ip.MF_GC_p_ff)

	
	if ip.PLAST1_sinexp:
		#GC_PC  = GC_PC_p #the type of synapse already define in the parameters init file
		
		GC_PC_r = {}
		GC_PC_ff = {}
		for module in range(ip.n_modules):
			for joint in range(ip.n_joints):
				#Recurrent
				sim.nest.CopyModel('stdp_synapse_sinexp',"stdp_synapse_sinexp"+str(module)+"_"+str(joint)+"_pos_r")
				sim.nest.CopyModel('stdp_synapse_sinexp',"stdp_synapse_sinexp"+str(module)+"_"+str(joint)+"_neg_r")
				#Feedforward
				sim.nest.CopyModel('stdp_synapse_sinexp',"stdp_synapse_sinexp"+str(module)+"_"+str(joint)+"_pos_ff")
				sim.nest.CopyModel('stdp_synapse_sinexp',"stdp_synapse_sinexp"+str(module)+"_"+str(joint)+"_neg_ff")

				#Recurrent
				sim.nest.SetDefaults("stdp_synapse_sinexp"+str(module)+"_"+str(joint)+"_pos_r",{"A_minus":   ip.GC_PC_p_r["A_minus"+"_"+str(module)+"_"+str(joint)],   # double - Amplitude of weight change for depression
																"A_plus":    ip.GC_PC_p_r["A_plus"+"_"+str(module)+"_"+str(joint)],   # double - Amplitude of weight change for facilitation 
																"Wmin":      ip.GC_PC_p_r["Wmin"+"_"+str(module)+"_"+str(joint)],    # double - Minimal synaptic weight 
																"Wmax":      ip.GC_PC_p_r["Wmax"+"_"+str(module)+"_"+str(joint)],    # double - Maximal synaptic weight
																"stdp_delay": ip.GC_PC_p_r['stdp_delay'+"_"+str(module)+"_"+str(joint)],
																"vt":        vt_r["vt_"+str(module)+"_"+str(joint)+"_pos"][0]})

				sim.nest.SetDefaults("stdp_synapse_sinexp"+str(module)+"_"+str(joint)+"_neg_r",{"A_minus":   ip.GC_PC_p_r["A_minus"+"_"+str(module)+"_"+str(joint)],   # double - Amplitude of weight change for depression
																"A_plus":    ip.GC_PC_p_r["A_plus"+"_"+str(module)+"_"+str(joint)],   # double - Amplitude of weight change for facilitation 
																"Wmin":      ip.GC_PC_p_r["Wmin"+"_"+str(module)+"_"+str(joint)],    # double - Minimal synaptic weight 
																"Wmax":      ip.GC_PC_p_r["Wmax"+"_"+str(module)+"_"+str(joint)],    # double - Maximal synaptic weight
																"stdp_delay": ip.GC_PC_p_r['stdp_delay'+"_"+str(module)+"_"+str(joint)],
																"vt":        vt_r["vt_"+str(module)+"_"+str(joint)+"_neg"][0]})

				#Feedforward
				sim.nest.SetDefaults("stdp_synapse_sinexp"+str(module)+"_"+str(joint)+"_pos_ff",{"A_minus":   ip.GC_PC_p_ff["A_minus"+"_"+str(module)+"_"+str(joint)],   # double - Amplitude of weight change for depression
																"A_plus":    ip.GC_PC_p_ff["A_plus"+"_"+str(module)+"_"+str(joint)],   # double - Amplitude of weight change for facilitation 
																"Wmin":      ip.GC_PC_p_ff["Wmin"+"_"+str(module)+"_"+str(joint)],    # double - Minimal synaptic weight 
																"Wmax":      ip.GC_PC_p_ff["Wmax"+"_"+str(module)+"_"+str(joint)],    # double - Maximal synaptic weight
																"stdp_delay": ip.GC_PC_p_ff['stdp_delay'+"_"+str(module)+"_"+str(joint)],
																"vt":        vt_ff["vt_"+str(module)+"_"+str(joint)+"_pos"][0]})

				sim.nest.SetDefaults("stdp_synapse_sinexp"+str(module)+"_"+str(joint)+"_neg_ff",{"A_minus":   ip.GC_PC_p_ff["A_minus"+"_"+str(module)+"_"+str(joint)],   # double - Amplitude of weight change for depression
																"A_plus":    ip.GC_PC_p_ff["A_plus"+"_"+str(module)+"_"+str(joint)],   # double - Amplitude of weight change for facilitation 
																"Wmin":      ip.GC_PC_p_ff["Wmin"+"_"+str(module)+"_"+str(joint)],    # double - Minimal synaptic weight 
																"Wmax":      ip.GC_PC_p_ff["Wmax"+"_"+str(module)+"_"+str(joint)],    # double - Maximal synaptic weight
																"stdp_delay": ip.GC_PC_p_ff['stdp_delay'+"_"+str(module)+"_"+str(joint)],
																"vt":        vt_ff["vt_"+str(module)+"_"+str(joint)+"_neg"][0]})


				#Recurrent
				GC_PC_r[str(module)+"_"+str(joint)+"_pos"] = {"model":    "stdp_synapse_sinexp"+str(module)+"_"+str(joint)+"_pos_r",
														"weight":   ip.GC_PC_p_r["weight"+"_"+str(module)+"_"+str(joint)],
														"delay":    ip.GC_PC_p_r["delay"+"_"+str(module)+"_"+str(joint)]}

				GC_PC_r[str(module)+"_"+str(joint)+"_neg"] = {"model":    "stdp_synapse_sinexp"+str(module)+"_"+str(joint)+"_neg_r",
														"weight":   ip.GC_PC_p_r["weight"+"_"+str(module)+"_"+str(joint)],
														"delay":    ip.GC_PC_p_r["delay"+"_"+str(module)+"_"+str(joint)]}

				#Feedforward
				GC_PC_ff[str(module)+"_"+str(joint)+"_pos"] = {"model":    "stdp_synapse_sinexp"+str(module)+"_"+str(joint)+"_pos_ff",
														"weight":   ip.GC_PC_p_ff["weight"+"_"+str(module)+"_"+str(joint)],
														"delay":    ip.GC_PC_p_ff["delay"+"_"+str(module)+"_"+str(joint)]}

				GC_PC_ff[str(module)+"_"+str(joint)+"_neg"] = {"model":    "stdp_synapse_sinexp"+str(module)+"_"+str(joint)+"_neg_ff",
														"weight":   ip.GC_PC_p_ff["weight"+"_"+str(module)+"_"+str(joint)],
														"delay":    ip.GC_PC_p_ff["delay"+"_"+str(module)+"_"+str(joint)]}

		


	else:
		GC_PC_r  = sim.StaticSynapse(**ip.GC_PC_p_r)
		GC_PC_ff  = sim.StaticSynapse(**ip.GC_PC_p_ff)
	
	IO_PC_r  = sim.StaticSynapse(**ip.IO_PC_p_r)
	IO_PC_ff  = sim.StaticSynapse(**ip.IO_PC_p_ff)


	MF_DCN_r = {}
	MF_DCN_ff = {}
	for module in range(ip.n_modules):
		for joint in range(ip.n_joints):
			MF_DCN_r[str(module)+"_"+str(joint)] = sim.StaticSynapse(**ip.MF_DCN_p_r["MF_DCN_p"+"_"+str(module)+"_"+str(joint)])
			MF_DCN_ff[str(module)+"_"+str(joint)] = sim.StaticSynapse(**ip.MF_DCN_p_ff["MF_DCN_p"+"_"+str(module)+"_"+str(joint)])
	
	PC_DCN_r = {}
	PC_DCN_ff = {}
	for module in range(ip.n_modules):
		for joint in range(ip.n_joints):
			PC_DCN_r[str(module)+"_"+str(joint)] = sim.StaticSynapse(**ip.PC_DCN_p_r["PC_DCN_p"+"_"+str(module)+"_"+str(joint)])
			PC_DCN_ff[str(module)+"_"+str(joint)] = sim.StaticSynapse(**ip.PC_DCN_p_ff["PC_DCN_p"+"_"+str(module)+"_"+str(joint)])
	
	
	#################### PROJECTIONS DEFINITIONS ########################################

	for module in range(ip.n_modules):
		for joint in range(ip.n_joints):
			# MF-GC: excitatory, each GC receives inputs from just a few MF
			#Recurrent
			sim.Projection(presynaptic_population=MF_r["MF_"+str(module)+"_"+str(joint)], postsynaptic_population=GC_r["GC"], 
				connector=MF_GC_c, synapse_type=MF_GC_r, receptor_type='excitatory')
			#Feedforward
			sim.Projection(presynaptic_population=MF_ff["MF_"+str(module)+"_"+str(joint)], postsynaptic_population=GC_ff["GC"], 
				connector=MF_GC_c, synapse_type=MF_GC_ff, receptor_type='excitatory') 

			# GC-PC: excitatory, each GC has potential to form few synapses with a large number of PC
			if ip.PLAST1_sinexp:
				PCi_r = {}
				PCi_ff = {}
				vti_r = {}
				vti_ff = {}
				for P in range(PC_n/2):
					#Recurrent
					PCi_r["PCi_"+str(module)+"_"+str(joint)+"_pos"] = sim.PopulationView(PC_r["PC_"+str(module)+"_"+str(joint)+"_pos"], np.array([P]))
					sim.nest.Connect(tuple(GC_r["GC"].all_cells), tuple(PCi_r["PCi_"+str(module)+"_"+str(joint)+"_pos"].all_cells), GC_PC_c, GC_PC_r[str(module)+"_"+str(joint)+"_pos"])
					
					A = sim.nest.GetConnections(tuple(GC_r["GC"].all_cells), tuple(PCi_r["PCi_"+str(module)+"_"+str(joint)+"_pos"].all_cells))
					sim.nest.SetStatus(A, {"vt_num":P})
					
					
					PCi_r["PCi_"+str(module)+"_"+str(joint)+"_neg"] = sim.PopulationView(PC_r["PC_"+str(module)+"_"+str(joint)+"_neg"], np.array([P]))
					sim.nest.Connect(tuple(GC_r["GC"].all_cells), tuple(PCi_r["PCi_"+str(module)+"_"+str(joint)+"_neg"].all_cells), GC_PC_c, GC_PC_r[str(module)+"_"+str(joint)+"_neg"])
					
					A = sim.nest.GetConnections(tuple(GC_r["GC"].all_cells), tuple(PCi_r["PCi_"+str(module)+"_"+str(joint)+"_neg"].all_cells))
					sim.nest.SetStatus(A, {"vt_num":P})
					
					#Feedforward
					PCi_ff["PCi_"+str(module)+"_"+str(joint)+"_pos"] = sim.PopulationView(PC_ff["PC_"+str(module)+"_"+str(joint)+"_pos"], np.array([P]))
					sim.nest.Connect(tuple(GC_ff["GC"].all_cells), tuple(PCi_ff["PCi_"+str(module)+"_"+str(joint)+"_pos"].all_cells), GC_PC_c, GC_PC_ff[str(module)+"_"+str(joint)+"_pos"])
					
					A = sim.nest.GetConnections(tuple(GC_ff["GC"].all_cells), tuple(PCi_ff["PCi_"+str(module)+"_"+str(joint)+"_pos"].all_cells))
					sim.nest.SetStatus(A, {"vt_num":P})
					
					
					PCi_ff["PCi_"+str(module)+"_"+str(joint)+"_neg"] = sim.PopulationView(PC_ff["PC_"+str(module)+"_"+str(joint)+"_neg"], np.array([P]))
					sim.nest.Connect(tuple(GC_ff["GC"].all_cells), tuple(PCi_ff["PCi_"+str(module)+"_"+str(joint)+"_neg"].all_cells), GC_PC_c, GC_PC_ff[str(module)+"_"+str(joint)+"_neg"])
					
					A = sim.nest.GetConnections(tuple(GC_ff["GC"].all_cells), tuple(PCi_ff["PCi_"+str(module)+"_"+str(joint)+"_neg"].all_cells))
					sim.nest.SetStatus(A, {"vt_num":P})
			else:
				sim.Projection(presynaptic_population= GC_r["GC"], postsynaptic_population=PC_r["PC_"+str(module)+"_"+str(joint)+"_pos"], 
						connector=GC_PC_c, synapse_type=GC_PC_r, receptor_type='excitatory')
				sim.Projection(presynaptic_population= GC_ff["GC"], postsynaptic_population=PC_ff["PC_"+str(module)+"_"+str(joint)+"_neg"], 
						connector=GC_PC_c, synapse_type=GC_PC_ff, receptor_type='excitatory')

			# IO-PC: teaching signal, each PC receives one input from IO        
			if ip.PLAST1_sinexp:
				#Recurrent
				sim.Projection(presynaptic_population=IO["IO_"+str(module)+"_"+str(joint)+"_pos"], postsynaptic_population=vt_r["vt_"+str(module)+"_"+str(joint)+"_pos"],
					connector=IO_PC_c, synapse_type=IO_PC_r, receptor_type='excitatory')#'TEACHING_SIGNAL')
				sim.Projection(presynaptic_population=IO["IO_"+str(module)+"_"+str(joint)+"_neg"], postsynaptic_population=vt_r["vt_"+str(module)+"_"+str(joint)+"_neg"], 
					connector=IO_PC_c, synapse_type=IO_PC_r, receptor_type='excitatory')#'TEACHING_SIGNAL')

				#Feedforward
				sim.Projection(presynaptic_population=IO["IO_"+str(module)+"_"+str(joint)+"_pos"], postsynaptic_population=vt_ff["vt_"+str(module)+"_"+str(joint)+"_pos"],
					connector=IO_PC_c, synapse_type=IO_PC_ff, receptor_type='excitatory')#'TEACHING_SIGNAL')
				sim.Projection(presynaptic_population=IO["IO_"+str(module)+"_"+str(joint)+"_neg"], postsynaptic_population=vt_ff["vt_"+str(module)+"_"+str(joint)+"_neg"], 
					connector=IO_PC_c, synapse_type=IO_PC_ff, receptor_type='excitatory')#'TEACHING_SIGNAL')


			# PC-DCN: inhibitory, 2 PC mapped to 1 DCN, select one to one for every half
			PCi_r = {}
			PCi_ff = {}
			count_DCN=0
			for P in range(PC_n/2):
				#Recurrent
				PCi_r["PCi_"+str(module)+"_"+str(joint)+"_pos"] = sim.PopulationView(PC_r["PC_"+str(module)+"_"+str(joint)+"_pos"], np.array([P]))
				sim.Projection(presynaptic_population= PCi_r["PCi_"+str(module)+"_"+str(joint)+"_pos"], 
					postsynaptic_population=sim.PopulationView(DCN_r["DCN_"+str(module)+"_"+str(joint)+"_pos"], np.array([count_DCN])), 
					connector=PC_DCN_c, synapse_type=PC_DCN_r[str(module)+"_"+str(joint)], receptor_type='inhibitory')
				
				PCi_r["PCi_"+str(module)+"_"+str(joint)+"_neg"] = sim.PopulationView(PC_r["PC_"+str(module)+"_"+str(joint)+"_neg"], np.array([P]))
				sim.Projection(presynaptic_population= PCi_r["PCi_"+str(module)+"_"+str(joint)+"_neg"], 
					postsynaptic_population=sim.PopulationView(DCN_r["DCN_"+str(module)+"_"+str(joint)+"_neg"], np.array([count_DCN])), 
					connector=PC_DCN_c, synapse_type=PC_DCN_r[str(module)+"_"+str(joint)], receptor_type='inhibitory')
				
				#Feedforward
				PCi_ff["PCi_"+str(module)+"_"+str(joint)+"_pos"] = sim.PopulationView(PC_ff["PC_"+str(module)+"_"+str(joint)+"_pos"], np.array([P]))
				sim.Projection(presynaptic_population= PCi_ff["PCi_"+str(module)+"_"+str(joint)+"_pos"], 
					postsynaptic_population=sim.PopulationView(DCN_ff["DCN_"+str(module)+"_"+str(joint)+"_pos"], np.array([count_DCN])), 
					connector=PC_DCN_c, synapse_type=PC_DCN_ff[str(module)+"_"+str(joint)], receptor_type='inhibitory')
				
				PCi_ff["PCi_"+str(module)+"_"+str(joint)+"_neg"] = sim.PopulationView(PC_ff["PC_"+str(module)+"_"+str(joint)+"_neg"], np.array([P]))
				sim.Projection(presynaptic_population= PCi_ff["PCi_"+str(module)+"_"+str(joint)+"_neg"], 
					postsynaptic_population=sim.PopulationView(DCN_ff["DCN_"+str(module)+"_"+str(joint)+"_neg"], np.array([count_DCN])), 
					connector=PC_DCN_c, synapse_type=PC_DCN_ff[str(module)+"_"+str(joint)], receptor_type='inhibitory')

				if P%2 == 1:
					count_DCN += 1 

			# MF-DCN: excitatory, all MF go to each DCN (all to all)
			#Recurrent
			sim.Projection(presynaptic_population=MF_r["MF_"+str(module)+"_"+str(joint)], postsynaptic_population=DCN_r["DCN_"+str(module)+"_"+str(joint)+"_pos"], 
				connector=MF_DCN_c, synapse_type=MF_DCN_r[str(module)+"_"+str(joint)], receptor_type='excitatory')
			sim.Projection(presynaptic_population=MF_r["MF_"+str(module)+"_"+str(joint)], postsynaptic_population=DCN_r["DCN_"+str(module)+"_"+str(joint)+"_neg"], 
				connector=MF_DCN_c, synapse_type=MF_DCN_r[str(module)+"_"+str(joint)], receptor_type='excitatory')

			#Feedforward
			sim.Projection(presynaptic_population=MF_ff["MF_"+str(module)+"_"+str(joint)], postsynaptic_population=DCN_ff["DCN_"+str(module)+"_"+str(joint)+"_pos"], 
				connector=MF_DCN_c, synapse_type=MF_DCN_ff[str(module)+"_"+str(joint)], receptor_type='excitatory')
			sim.Projection(presynaptic_population=MF_ff["MF_"+str(module)+"_"+str(joint)], postsynaptic_population=DCN_ff["DCN_"+str(module)+"_"+str(joint)+"_neg"], 
				connector=MF_DCN_c, synapse_type=MF_DCN_ff[str(module)+"_"+str(joint)], receptor_type='excitatory')

	# Recording the spikes
	
	if ip.RECORDING_CELLS:    
	# Create Auxiliary tools
		#Recurrent
		recdict_r = [{"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_MF_0_0_recurrent"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_MF_0_0_cur_q_recurrent"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_MF_0_0_des_q_recurrent"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_MF_0_0_des_torque_recurrent"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_GR_recurrent"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_PC_0_0_recurrent"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_PC_0_0_pos_recurrent"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_PC_0_0_neg_recurrent"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_IO_0_0_recurrent"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_IO_0_0_pos_recurrent"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_IO_0_0_neg_recurrent"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_DCN_0_0_recurrent"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_DCN_0_0_pos_recurrent"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_DCN_0_0_neg_recurrent"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_MF_0_1_recurrent"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_MF_0_1_cur_q_recurrent"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_MF_0_1_des_q_recurrent"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_MF_0_1_des_torque_recurrent"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_PC_0_1_recurrent"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_PC_0_1_pos_recurrent"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_PC_0_1_neg_recurrent"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_IO_0_1_recurrent"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_IO_0_1_pos_recurrent"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_IO_0_1_neg_recurrent"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_DCN_0_1_recurrent"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_DCN_0_1_pos_recurrent"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_DCN_0_1_neg_recurrent"}]

		spikedetector_r = sim.nest.Create("spike_detector",len(recdict_r), params=recdict_r)

		sim.nest.Connect(tuple(MF_r["MF_0_0"].all_cells),    [spikedetector_r[0]])
		sim.nest.Connect(tuple(MF_r["MF_0_0_cur_q"].all_cells),    [spikedetector_r[1]])
		sim.nest.Connect(tuple(MF_r["MF_0_0_des_q"].all_cells),    [spikedetector_r[2]])
		sim.nest.Connect(tuple(MF_r["MF_0_0_des_torque"].all_cells),    [spikedetector_r[3]])
		sim.nest.Connect(tuple(GC_r["GC"].all_cells),        [spikedetector_r[4]])
		sim.nest.Connect(tuple(PC_r["PC_0_0"].all_cells),    [spikedetector_r[5]])
		sim.nest.Connect(tuple(PC_r["PC_0_0_pos"].all_cells),    [spikedetector_r[6]])
		sim.nest.Connect(tuple(PC_r["PC_0_0_neg"].all_cells),    [spikedetector_r[7]])
		sim.nest.Connect(tuple(IO["IO_0_0"].all_cells),    [spikedetector_r[8]])
		sim.nest.Connect(tuple(IO["IO_0_0_pos"].all_cells),    [spikedetector_r[9]])
		sim.nest.Connect(tuple(IO["IO_0_0_neg"].all_cells),    [spikedetector_r[10]])
		sim.nest.Connect(tuple(DCN_r["DCN_0_0"].all_cells),  [spikedetector_r[11]])
		sim.nest.Connect(tuple(DCN_r["DCN_0_0_pos"].all_cells),  [spikedetector_r[12]])
		sim.nest.Connect(tuple(DCN_r["DCN_0_0_neg"].all_cells),  [spikedetector_r[13]])

		sim.nest.Connect(tuple(MF_r["MF_0_1"].all_cells),    [spikedetector_r[14]])
		sim.nest.Connect(tuple(MF_r["MF_0_1_cur_q"].all_cells),    [spikedetector_r[15]])
		sim.nest.Connect(tuple(MF_r["MF_0_1_des_q"].all_cells),    [spikedetector_r[16]])
		sim.nest.Connect(tuple(MF_r["MF_0_1_des_torque"].all_cells),    [spikedetector_r[17]])
		sim.nest.Connect(tuple(PC_r["PC_0_1"].all_cells),    [spikedetector_r[18]])
		sim.nest.Connect(tuple(PC_r["PC_0_1_pos"].all_cells),    [spikedetector_r[19]])
		sim.nest.Connect(tuple(PC_r["PC_0_1_neg"].all_cells),    [spikedetector_r[20]])
		sim.nest.Connect(tuple(IO["IO_0_1"].all_cells),    [spikedetector_r[21]])
		sim.nest.Connect(tuple(IO["IO_0_1_pos"].all_cells),    [spikedetector_r[22]])
		sim.nest.Connect(tuple(IO["IO_0_1_neg"].all_cells),    [spikedetector_r[23]])
		sim.nest.Connect(tuple(DCN_r["DCN_0_1"].all_cells),  [spikedetector_r[24]])
		sim.nest.Connect(tuple(DCN_r["DCN_0_1_pos"].all_cells),  [spikedetector_r[25]])
		sim.nest.Connect(tuple(DCN_r["DCN_0_1_neg"].all_cells),  [spikedetector_r[26]])
		
		#Feedforward
		recdict_ff = [{"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_MF_0_0_feedforward"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_MF_0_0_des_q_feedforward"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_MF_0_0_des_qd_feedforward"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_MF_0_0_cur_q_feedforward"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_GR_feedforward"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_PC_0_0_feedforward"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_PC_0_0_pos_feedforward"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_PC_0_0_neg_feedforward"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_IO_0_0_feedforward"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_IO_0_0_pos_feedforward"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_IO_0_0_neg_feedforward"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_DCN_0_0_feedforward"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_DCN_0_0_pos_feedforward"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_DCN_0_0_neg_feedforward"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_MF_0_1_feedforward"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_MF_0_1_des_q_feedforward"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_MF_0_1_des_qd_feedforward"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_MF_0_1_cur_q_feedforward"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_PC_0_1_feedforward"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_PC_0_1_pos_feedforward"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_PC_0_1_neg_feedforward"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_IO_0_1_feedforward"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_IO_0_1_pos_feedforward"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_IO_0_1_neg_feedforward"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_DCN_0_1_feedforward"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_DCN_0_1_pos_feedforward"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_DCN_0_1_neg_feedforward"}]

		spikedetector_ff = sim.nest.Create("spike_detector",len(recdict_ff), params=recdict_ff)

		sim.nest.Connect(tuple(MF_ff["MF_0_0"].all_cells),    [spikedetector_ff[0]])
		sim.nest.Connect(tuple(MF_ff["MF_0_0_des_q"].all_cells),    [spikedetector_ff[1]])
		sim.nest.Connect(tuple(MF_ff["MF_0_0_des_qd"].all_cells),    [spikedetector_ff[2]])
		sim.nest.Connect(tuple(MF_ff["MF_0_0_cur_q"].all_cells),    [spikedetector_ff[3]])
		sim.nest.Connect(tuple(GC_ff["GC"].all_cells),        [spikedetector_ff[4]])
		sim.nest.Connect(tuple(PC_ff["PC_0_0"].all_cells),    [spikedetector_ff[5]])
		sim.nest.Connect(tuple(PC_ff["PC_0_0_pos"].all_cells),    [spikedetector_ff[6]])
		sim.nest.Connect(tuple(PC_ff["PC_0_0_neg"].all_cells),    [spikedetector_ff[7]])
		sim.nest.Connect(tuple(IO["IO_0_0"].all_cells),    [spikedetector_ff[8]])
		sim.nest.Connect(tuple(IO["IO_0_0_pos"].all_cells),    [spikedetector_ff[9]])
		sim.nest.Connect(tuple(IO["IO_0_0_neg"].all_cells),    [spikedetector_ff[10]])
		sim.nest.Connect(tuple(DCN_ff["DCN_0_0"].all_cells),  [spikedetector_ff[11]])
		sim.nest.Connect(tuple(DCN_ff["DCN_0_0_pos"].all_cells),  [spikedetector_ff[12]])
		sim.nest.Connect(tuple(DCN_ff["DCN_0_0_neg"].all_cells),  [spikedetector_ff[13]])

		sim.nest.Connect(tuple(MF_ff["MF_0_1"].all_cells),    [spikedetector_ff[14]])
		sim.nest.Connect(tuple(MF_ff["MF_0_1_des_q"].all_cells),    [spikedetector_ff[15]])
		sim.nest.Connect(tuple(MF_ff["MF_0_1_des_qd"].all_cells),    [spikedetector_ff[16]])
		sim.nest.Connect(tuple(MF_ff["MF_0_1_cur_q"].all_cells),    [spikedetector_ff[17]])
		sim.nest.Connect(tuple(PC_ff["PC_0_1"].all_cells),    [spikedetector_ff[18]])
		sim.nest.Connect(tuple(PC_ff["PC_0_1_pos"].all_cells),    [spikedetector_ff[19]])
		sim.nest.Connect(tuple(PC_ff["PC_0_1_neg"].all_cells),    [spikedetector_ff[20]])
		sim.nest.Connect(tuple(IO["IO_0_1"].all_cells),    [spikedetector_ff[21]])
		sim.nest.Connect(tuple(IO["IO_0_1_pos"].all_cells),    [spikedetector_ff[22]])
		sim.nest.Connect(tuple(IO["IO_0_1_neg"].all_cells),    [spikedetector_ff[23]])
		sim.nest.Connect(tuple(DCN_ff["DCN_0_1"].all_cells),  [spikedetector_ff[24]])
		sim.nest.Connect(tuple(DCN_ff["DCN_0_1_pos"].all_cells),  [spikedetector_ff[25]])
		sim.nest.Connect(tuple(DCN_ff["DCN_0_1_neg"].all_cells),  [spikedetector_ff[26]])
		

	return (MF_r["MF_0_0"], MF_r["MF_0_1"], MF_r["MF_0_0_cur_q"], MF_r["MF_0_1_cur_q"], 
		MF_r["MF_0_0_des_q"], MF_r["MF_0_1_des_q"], MF_r["MF_0_0_des_torque"], MF_r["MF_0_1_des_torque"], 
		GC_r["GC"], PC_r["PC_0_0"], PC_r["PC_0_1"], PC_r["PC_0_0_pos"], PC_r["PC_0_0_neg"], PC_r["PC_0_1_pos"], PC_r["PC_0_1_neg"], 
		IO["IO_0_0"], IO["IO_0_1"], IO["IO_0_0_pos"], IO["IO_0_0_neg"], IO["IO_0_1_pos"], IO["IO_0_1_neg"], 
		DCN_r["DCN_0_0"], DCN_r["DCN_0_1"], DCN_r["DCN_0_0_pos"], DCN_r["DCN_0_0_neg"], DCN_r["DCN_0_1_pos"], DCN_r["DCN_0_1_neg"],
		MF_ff["MF_0_0"], MF_ff["MF_0_1"], MF_ff["MF_0_0_des_q"], MF_ff["MF_0_0_des_qd"], MF_ff["MF_0_1_des_q"], MF_ff["MF_0_1_des_qd"], MF_ff["MF_0_0_cur_q"], MF_ff["MF_0_1_cur_q"], 
		GC_ff["GC"], PC_ff["PC_0_0"], PC_ff["PC_0_1"], PC_ff["PC_0_0_pos"], PC_ff["PC_0_0_neg"], PC_ff["PC_0_1_pos"], PC_ff["PC_0_1_neg"], 
		DCN_ff["DCN_0_0"], DCN_ff["DCN_0_1"], DCN_ff["DCN_0_0_pos"], DCN_ff["DCN_0_0_neg"], DCN_ff["DCN_0_1_pos"], DCN_ff["DCN_0_1_neg"])

# -----------------------------------------------------------------------#
(MF_0_0_recurrent, MF_0_1_recurrent, MF_0_0_cur_q_recurrent, MF_0_1_cur_q_recurrent, MF_0_0_des_q_recurrent, 
	MF_0_1_des_q_recurrent, MF_0_0_des_torque_recurrent, MF_0_1_des_torque_recurrent, 
	GC_recurrent, PC_0_0_recurrent, PC_0_1_recurrent, PC_0_0_pos_recurrent, PC_0_0_neg_recurrent, PC_0_1_pos_recurrent, PC_0_1_neg_recurrent, 
	IO_0_0, IO_0_1, IO_0_0_pos, IO_0_0_neg, IO_0_1_pos, IO_0_1_neg, 
	DCN_0_0_recurrent, DCN_0_1_recurrent, DCN_0_0_pos_recurrent, DCN_0_0_neg_recurrent, 
	DCN_0_1_pos_recurrent, DCN_0_1_neg_recurrent, 
	MF_0_0_feedforward, MF_0_1_feedforward, MF_0_0_des_q_feedforward, MF_0_0_des_qd_feedforward, 
	MF_0_1_des_q_feedforward, MF_0_1_des_qd_feedforward, MF_0_0_cur_q_feedforward, MF_0_1_cur_q_feedforward, 
	GC_feedforward, PC_0_0_feedforward, PC_0_1_feedforward, PC_0_0_pos_feedforward, PC_0_0_neg_feedforward, PC_0_1_pos_feedforward, PC_0_1_neg_feedforward, 
	DCN_0_0_feedforward, DCN_0_1_feedforward, DCN_0_0_pos_feedforward, DCN_0_0_neg_feedforward, 
	DCN_0_1_pos_feedforward, DCN_0_1_neg_feedforward) = create_cerebellum()