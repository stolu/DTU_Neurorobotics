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
sys.path.append("/home/dtu-neurorobotics/Documents/NRP/Models/brain_model/DTU_spiking_cereb_recurrent")
#from init_parameters_dtu_pavia_cereb import*

sim.setup(timestep=0.1, min_delay=0.1, max_delay=1001.0, threads=8, rng_seeds=[1234])
# 8 threads: recomended number. Reference: http://www.nest-simulator.org/wp-content/uploads/2015/02/NEST_by_Example.pdf

# Remapping variables to allow reloading of module by importing it whole (Easier for development)
import init_parameters_dtu_pavia_cereb_recurrent_torque as ip
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

	MF   = {}
	GC   = {}
	IO   = {}
	PC   = {}
	DCN  = {}
	vt   = {}

	################
	###### MF ######
	################
	for module in range(ip.n_modules):
		for joint in range(ip.n_joints):
			#Create independent subgroups of Volume Transmitter
			vt["vt_"+str(module)+"_"+str(joint)+"_pos"] = sim.create(vt_t, {}, PC_n/2)
			vt["vt_"+str(module)+"_"+str(joint)+"_neg"] = sim.create(vt_t, {}, PC_n/2)

			vti = {}
			for P in range(PC_n/2):
				vti["vti_"+str(module)+"_"+str(joint)+"_pos"] = sim.PopulationView(vt["vt_"+str(module)+"_"+str(joint)+"_pos"], np.array([P]))
				vti["vti_"+str(module)+"_"+str(joint)+"_neg"] = sim.PopulationView(vt["vt_"+str(module)+"_"+str(joint)+"_neg"], np.array([P]))

				sim.nest.SetStatus(tuple(vti["vti_"+str(module)+"_"+str(joint)+"_pos"].all_cells), {"vt_num": P})
				sim.nest.SetStatus(tuple(vti["vti_"+str(module)+"_"+str(joint)+"_neg"].all_cells), {"vt_num": P})

			#Create independent subgroups of MFs for the different types of inputs
			#MF["MF_"+str(module)+"_"+str(joint)+"_cur_q"]  = sim.create(MF_t, ip.MF_p, ip.MF_n/n_inputs_MF) # Mossy Fiber module "module" joint "joint", current position 
			#MF["MF_"+str(module)+"_"+str(joint)+"_des_q"]  = sim.create(MF_t, ip.MF_p, ip.MF_n/n_inputs_MF) # Mossy Fiber module "module" joint "joint", desired position
			MF["MF_"+str(module)+"_"+str(joint)+"_des_torque"]  = sim.create(MF_t, ip.MF_p, ip.MF_n/n_inputs_MF) # Mossy Fiber module "module" joint "joint", desired torque
			
			
			#Create the final MF population (combination the subgroups)
			MF["MF_"+str(module)+"_"+str(joint)] = MF["MF_"+str(module)+"_"+str(joint)+"_des_torque"]
	

	################
	###### GC ######
	################
	GC["GC"] = sim.create(GC_t, ip.GC_p, GC_n)

	################   ################   #################
	###### IO ######   ###### PC ######   ###### DCN ######
	################   ################   #################

	for module in range(ip.n_modules):
		for joint in range(ip.n_joints):
			#Create independent subgroups of IOs for the +|- errors
			IO["IO_"+str(module)+"_"+str(joint)+"_pos"] = sim.create(IO_t, ip.IO_p, IO_n/2) # Agonist torque contribution
			IO["IO_"+str(module)+"_"+str(joint)+"_neg"] = sim.create(IO_t, ip.IO_p, IO_n/2) # Antagonist torque contribution
			#Create independent subgroups of PCs for the +|- errors
			PC["PC_"+str(module)+"_"+str(joint)+"_pos"] = sim.create(PC_t, ip.PC_p, PC_n/2) # Agonist torque contribution
			PC["PC_"+str(module)+"_"+str(joint)+"_neg"] = sim.create(PC_t, ip.PC_p, PC_n/2) # Antagonist torque contribution
			#Create independent subgroups of DCNs for the +|- errors
			DCN["DCN_"+str(module)+"_"+str(joint)+"_pos"] = sim.create(DCN_t, ip.DCN_p, DCN_n/2) # Agonist torque contribution
			DCN["DCN_"+str(module)+"_"+str(joint)+"_neg"] = sim.create(DCN_t, ip.DCN_p, DCN_n/2) # Antagonist torque contribution

			#Create the final IO population (combination of the subgroups)
			IO["IO_"+str(module)+"_"+str(joint)] = IO["IO_"+str(module)+"_"+str(joint)+"_pos"] + IO["IO_"+str(module)+"_"+str(joint)+"_neg"]
			#Create the final PC population (combination of the subgroups)
			PC["PC_"+str(module)+"_"+str(joint)] = PC["PC_"+str(module)+"_"+str(joint)+"_pos"] + PC["PC_"+str(module)+"_"+str(joint)+"_neg"]
			#Create the final DCN population (combination of the subgroups)
			DCN["DCN_"+str(module)+"_"+str(joint)] = DCN["DCN_"+str(module)+"_"+str(joint)+"_pos"] + DCN["DCN_"+str(module)+"_"+str(joint)+"_neg"]
			
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
	MF_GC  = sim.StaticSynapse(**ip.MF_GC_p)


	
	if ip.PLAST1_sinexp:
		#GC_PC  = GC_PC_p #the type of synapse already define in the parameters init file
		
		GC_PC = {}
		for module in range(ip.n_modules):
			for joint in range(ip.n_joints):
						
				sim.nest.CopyModel('stdp_synapse_sinexp',"stdp_synapse_sinexp"+str(module)+"_"+str(joint)+"_pos")
				sim.nest.CopyModel('stdp_synapse_sinexp',"stdp_synapse_sinexp"+str(module)+"_"+str(joint)+"_neg")

				
				sim.nest.SetDefaults("stdp_synapse_sinexp"+str(module)+"_"+str(joint)+"_pos",{"A_minus":   ip.GC_PC_p["A_minus"+"_"+str(module)+"_"+str(joint)],   # double - Amplitude of weight change for depression
																"A_plus":    ip.GC_PC_p["A_plus"+"_"+str(module)+"_"+str(joint)],   # double - Amplitude of weight change for facilitation 
																"Wmin":      ip.GC_PC_p["Wmin"+"_"+str(module)+"_"+str(joint)],    # double - Minimal synaptic weight 
																"Wmax":      ip.GC_PC_p["Wmax"+"_"+str(module)+"_"+str(joint)],    # double - Maximal synaptic weight
																"stdp_delay": ip.GC_PC_p['stdp_delay'+"_"+str(module)+"_"+str(joint)],
																"vt":        vt["vt_"+str(module)+"_"+str(joint)+"_pos"][0]})

				sim.nest.SetDefaults("stdp_synapse_sinexp"+str(module)+"_"+str(joint)+"_neg",{"A_minus":   ip.GC_PC_p["A_minus"+"_"+str(module)+"_"+str(joint)],   # double - Amplitude of weight change for depression
																"A_plus":    ip.GC_PC_p["A_plus"+"_"+str(module)+"_"+str(joint)],   # double - Amplitude of weight change for facilitation 
																"Wmin":      ip.GC_PC_p["Wmin"+"_"+str(module)+"_"+str(joint)],    # double - Minimal synaptic weight 
																"Wmax":      ip.GC_PC_p["Wmax"+"_"+str(module)+"_"+str(joint)],    # double - Maximal synaptic weight
																"stdp_delay": ip.GC_PC_p['stdp_delay'+"_"+str(module)+"_"+str(joint)],
																"vt":        vt["vt_"+str(module)+"_"+str(joint)+"_neg"][0]})


				GC_PC[str(module)+"_"+str(joint)+"_pos"] = {"model":    "stdp_synapse_sinexp"+str(module)+"_"+str(joint)+"_pos",
														"weight":   ip.GC_PC_p["weight"+"_"+str(module)+"_"+str(joint)],
														"delay":    ip.GC_PC_p["delay"+"_"+str(module)+"_"+str(joint)]}

				GC_PC[str(module)+"_"+str(joint)+"_neg"] = {"model":    "stdp_synapse_sinexp"+str(module)+"_"+str(joint)+"_neg",
														"weight":   ip.GC_PC_p["weight"+"_"+str(module)+"_"+str(joint)],
														"delay":    ip.GC_PC_p["delay"+"_"+str(module)+"_"+str(joint)]}

	else:
		GC_PC  = sim.StaticSynapse(**ip.GC_PC_p)
	
	IO_PC  = sim.StaticSynapse(**ip.IO_PC_p)


	MF_DCN = {}
	for module in range(ip.n_modules):
		for joint in range(ip.n_joints):
			MF_DCN[str(module)+"_"+str(joint)] = sim.StaticSynapse(**ip.MF_DCN_p["MF_DCN_p"+"_"+str(module)+"_"+str(joint)])
	
	PC_DCN = {}
	for module in range(ip.n_modules):
		for joint in range(ip.n_joints):
			PC_DCN[str(module)+"_"+str(joint)] = sim.StaticSynapse(**ip.PC_DCN_p["PC_DCN_p"+"_"+str(module)+"_"+str(joint)])
	
	
	#################### PROJECTIONS DEFINITIONS ########################################

	for module in range(ip.n_modules):
		for joint in range(ip.n_joints):
			# MF-GC: excitatory, each GC receives inputs from just a few MF
			sim.Projection(presynaptic_population=MF["MF_"+str(module)+"_"+str(joint)], postsynaptic_population=GC["GC"], 
				connector=MF_GC_c, synapse_type=MF_GC, receptor_type='excitatory') 

			# GC-PC: excitatory, each GC has potential to form few synapses with a large number of PC
			if ip.PLAST1_sinexp:
				PCi = {}
				vti = {}
				for P in range(PC_n/2):
					PCi["PCi_"+str(module)+"_"+str(joint)+"_pos"] = sim.PopulationView(PC["PC_"+str(module)+"_"+str(joint)+"_pos"], np.array([P]))
					sim.nest.Connect(tuple(GC["GC"].all_cells), tuple(PCi["PCi_"+str(module)+"_"+str(joint)+"_pos"].all_cells), GC_PC_c, GC_PC[str(module)+"_"+str(joint)+"_pos"])
					
					A = sim.nest.GetConnections(tuple(GC["GC"].all_cells), tuple(PCi["PCi_"+str(module)+"_"+str(joint)+"_pos"].all_cells))
					#vti["vti_"+str(module)+"_"+str(joint)+"_pos"] = sim.PopulationView(vt["vt_"+str(module)+"_"+str(joint)+"_pos"], np.array([P]))
					#sim.nest.SetStatus(A, {"vt":tuple(vti["vti_"+str(module)+"_"+str(joint)+"_pos"].all_cells)[0]})
					sim.nest.SetStatus(A, {"vt_num":P})
					
					
					PCi["PCi_"+str(module)+"_"+str(joint)+"_neg"] = sim.PopulationView(PC["PC_"+str(module)+"_"+str(joint)+"_neg"], np.array([P]))
					sim.nest.Connect(tuple(GC["GC"].all_cells), tuple(PCi["PCi_"+str(module)+"_"+str(joint)+"_neg"].all_cells), GC_PC_c, GC_PC[str(module)+"_"+str(joint)+"_neg"])
					
					A = sim.nest.GetConnections(tuple(GC["GC"].all_cells), tuple(PCi["PCi_"+str(module)+"_"+str(joint)+"_neg"].all_cells))
					#vti["vti_"+str(module)+"_"+str(joint)+"_neg"] = sim.PopulationView(vt["vt_"+str(module)+"_"+str(joint)+"_neg"], np.array([P]))
					#sim.nest.SetStatus(A, {"vt":tuple(vti["vti_"+str(module)+"_"+str(joint)+"_neg"].all_cells)[0]})
					sim.nest.SetStatus(A, {"vt_num":P})
			else:
				sim.Projection(presynaptic_population= GC["GC"], postsynaptic_population=PC["PC_"+str(module)+"_"+str(joint)+"_pos"], 
						connector=GC_PC_c, synapse_type=GC_PC, receptor_type='excitatory')
				sim.Projection(presynaptic_population= GC["GC"], postsynaptic_population=PC["PC_"+str(module)+"_"+str(joint)+"_neg"], 
						connector=GC_PC_c, synapse_type=GC_PC, receptor_type='excitatory')

			# IO-PC: teaching signal, each PC receives one input from IO        
			if ip.PLAST1_sinexp:
				sim.Projection(presynaptic_population=IO["IO_"+str(module)+"_"+str(joint)+"_pos"], postsynaptic_population=vt["vt_"+str(module)+"_"+str(joint)+"_pos"],
					connector=IO_PC_c, synapse_type=IO_PC, receptor_type='excitatory')#'TEACHING_SIGNAL')
				sim.Projection(presynaptic_population=IO["IO_"+str(module)+"_"+str(joint)+"_neg"], postsynaptic_population=vt["vt_"+str(module)+"_"+str(joint)+"_neg"], 
					connector=IO_PC_c, synapse_type=IO_PC, receptor_type='excitatory')#'TEACHING_SIGNAL')


			# PC-DCN: inhibitory, 2 PC mapped to 1 DCN, select one to one for every half
			PCi = {}
			count_DCN=0
			for P in range(PC_n/2):
				PCi["PCi_"+str(module)+"_"+str(joint)+"_pos"] = sim.PopulationView(PC["PC_"+str(module)+"_"+str(joint)+"_pos"], np.array([P]))
				sim.Projection(presynaptic_population= PCi["PCi_"+str(module)+"_"+str(joint)+"_pos"], 
					postsynaptic_population=sim.PopulationView(DCN["DCN_"+str(module)+"_"+str(joint)+"_pos"], np.array([count_DCN])), 
					connector=PC_DCN_c, synapse_type=PC_DCN[str(module)+"_"+str(joint)], receptor_type='inhibitory')
				
				PCi["PCi_"+str(module)+"_"+str(joint)+"_neg"] = sim.PopulationView(PC["PC_"+str(module)+"_"+str(joint)+"_neg"], np.array([P]))
				sim.Projection(presynaptic_population= PCi["PCi_"+str(module)+"_"+str(joint)+"_neg"], 
					postsynaptic_population=sim.PopulationView(DCN["DCN_"+str(module)+"_"+str(joint)+"_neg"], np.array([count_DCN])), 
					connector=PC_DCN_c, synapse_type=PC_DCN[str(module)+"_"+str(joint)], receptor_type='inhibitory')
						   
				if P%2 == 1:
					count_DCN += 1 

			# MF-DCN: excitatory, all MF go to each DCN (all to all)
			sim.Projection(presynaptic_population=MF["MF_"+str(module)+"_"+str(joint)], postsynaptic_population=DCN["DCN_"+str(module)+"_"+str(joint)+"_pos"], 
				connector=MF_DCN_c, synapse_type=MF_DCN[str(module)+"_"+str(joint)], receptor_type='excitatory')
			sim.Projection(presynaptic_population=MF["MF_"+str(module)+"_"+str(joint)], postsynaptic_population=DCN["DCN_"+str(module)+"_"+str(joint)+"_neg"], 
				connector=MF_DCN_c, synapse_type=MF_DCN[str(module)+"_"+str(joint)], receptor_type='excitatory')

	# Recording the spikes
	
	if ip.RECORDING_CELLS:    
	# Create Auxiliary tools
		recdict = [{"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_MF_0_0"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_MF_0_0_des_torque"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_GR"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_PC_0_0"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_PC_0_0_pos"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_PC_0_0_neg"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_IO_0_0"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_IO_0_0_pos"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_IO_0_0_neg"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_DCN_0_0"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_DCN_0_0_pos"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_DCN_0_0_neg"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_MF_0_1"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_MF_0_1_des_torque"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_PC_0_1"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_PC_0_1_pos"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_PC_0_1_neg"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_IO_0_1"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_IO_0_1_pos"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_IO_0_1_neg"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_DCN_0_1"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_DCN_0_1_pos"},
				   {"to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_DCN_0_1_neg"}]

		spikedetector = sim.nest.Create("spike_detector",len(recdict), params=recdict)

		sim.nest.Connect(tuple(MF["MF_0_0"].all_cells),    [spikedetector[0]])
		sim.nest.Connect(tuple(MF["MF_0_0_des_torque"].all_cells),    [spikedetector[1]])
		sim.nest.Connect(tuple(GC["GC"].all_cells),        [spikedetector[2]])
		sim.nest.Connect(tuple(PC["PC_0_0"].all_cells),    [spikedetector[3]])
		sim.nest.Connect(tuple(PC["PC_0_0_pos"].all_cells),    [spikedetector[4]])
		sim.nest.Connect(tuple(PC["PC_0_0_neg"].all_cells),    [spikedetector[5]])
		sim.nest.Connect(tuple(IO["IO_0_0"].all_cells),    [spikedetector[6]])
		sim.nest.Connect(tuple(IO["IO_0_0_pos"].all_cells),    [spikedetector[7]])
		sim.nest.Connect(tuple(IO["IO_0_0_neg"].all_cells),    [spikedetector[8]])
		sim.nest.Connect(tuple(DCN["DCN_0_0"].all_cells),  [spikedetector[9]])
		sim.nest.Connect(tuple(DCN["DCN_0_0_pos"].all_cells),  [spikedetector[10]])
		sim.nest.Connect(tuple(DCN["DCN_0_0_neg"].all_cells),  [spikedetector[11]])

		sim.nest.Connect(tuple(MF["MF_0_1"].all_cells),    [spikedetector[12]])
		sim.nest.Connect(tuple(MF["MF_0_1_des_torque"].all_cells),    [spikedetector[13]])
		sim.nest.Connect(tuple(PC["PC_0_1"].all_cells),    [spikedetector[14]])
		sim.nest.Connect(tuple(PC["PC_0_1_pos"].all_cells),    [spikedetector[15]])
		sim.nest.Connect(tuple(PC["PC_0_1_neg"].all_cells),    [spikedetector[16]])
		sim.nest.Connect(tuple(IO["IO_0_1"].all_cells),    [spikedetector[17]])
		sim.nest.Connect(tuple(IO["IO_0_1_pos"].all_cells),    [spikedetector[18]])
		sim.nest.Connect(tuple(IO["IO_0_1_neg"].all_cells),    [spikedetector[19]])
		sim.nest.Connect(tuple(DCN["DCN_0_1"].all_cells),  [spikedetector[20]])
		sim.nest.Connect(tuple(DCN["DCN_0_1_pos"].all_cells),  [spikedetector[21]])
		sim.nest.Connect(tuple(DCN["DCN_0_1_neg"].all_cells),  [spikedetector[22]])
		

	return MF["MF_0_0"], MF["MF_0_1"], MF["MF_0_0_des_torque"], MF["MF_0_1_des_torque"], GC["GC"], PC["PC_0_0"], PC["PC_0_1"], PC["PC_0_0_pos"], PC["PC_0_0_neg"], PC["PC_0_1_pos"], PC["PC_0_1_neg"], IO["IO_0_0"], IO["IO_0_1"], IO["IO_0_0_pos"], IO["IO_0_0_neg"], IO["IO_0_1_pos"], IO["IO_0_1_neg"], DCN["DCN_0_0"], DCN["DCN_0_1"], DCN["DCN_0_0_pos"], DCN["DCN_0_0_neg"], DCN["DCN_0_1_pos"], DCN["DCN_0_1_neg"]

# -----------------------------------------------------------------------#
MF_0_0, MF_0_1, MF_0_0_des_torque, MF_0_1_des_torque, GC, PC_0_0, PC_0_1, PC_0_0_pos, PC_0_0_neg, PC_0_1_pos, PC_0_1_neg, IO_0_0, IO_0_1, IO_0_0_pos, IO_0_0_neg, IO_0_1_pos, IO_0_1_neg, DCN_0_0, DCN_0_1, DCN_0_0_pos, DCN_0_0_neg, DCN_0_1_pos, DCN_0_1_neg = create_cerebellum()