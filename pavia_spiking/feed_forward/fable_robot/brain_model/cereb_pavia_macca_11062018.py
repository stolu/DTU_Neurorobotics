# -*- coding: utf-8 -*-
"""

"""
# pragma: no cover

__author__ = 'Marie Claire Capolei'

from hbp_nrp_cle.brainsim import simulator as sim
#from sim import native_cell_type, native_synapse_type
import numpy as np

#from pyNN import nest as nst

import logging
#import pyNN.nest 
#from pyNN.nest  import native_cell_type, native_synapse_type
global MF_number, GR_number, PC_number, DCN_number, IO_number
logger = logging.getLogger(__name__)

#import nest
#nest.Install("albertomodule")

def create_brain():
    global MF_number, GR_number, PC_number, DCN_number, IO_number
    debug_cereb = 0
    """
    Initializes PyNN with the minimal neuronal network
    """

    sim.setup(timestep=0.1, min_delay=0.1, max_delay=1.0, threads=6, rng_seeds=[0])

    # Following parameters were taken from the husky braitenberg brain experiment (braitenberg.py)

    #Physical quantity	Units
    #time	ms
    #voltage	mV
    #current	nA
    #conductance	µS
    #capacitance	nF
    #firing rate	/s
    #phase/angle	deg

    # ................................................................................................
    # --------------------------------------- CEREBELLUM----------------------------------------------
    # ................................................................................................
    # Cell numbers
    MF_number  = 200#300
    GR_number  = MF_number*20
    PC_number  = 40#72
    DCN_number = PC_number/2
    IO_number  = PC_number
    
    # .........................................................................
    # SYNAPSES PARAMETERS

    # define how many synaptic plasticity you want to use
    PLAST1 = False # PF-PC plasticity
    PLAST2 = False # MF - DCN plasticity
    PLAST3 = False # PC - DCN plasticity



    # - long-term potentiation (LTP) is a persistent strengthening of synapses based on recent patterns of activity. These are patterns of synaptic activity that produce a long-lasting increase in signal transmission between two neurons.
    # - Long-term depression (LTD), in neurophysiology, is an activity-dependent reduction in the efficacy of neuronal synapses lasting hours or longer following a long patterned stimulus. LTD occurs in many areas of the CNS with varying mechanisms depending upon brain region and developmental progress.
    # double - Amplitude of weight change for facilitation

    # ParellelFiber - PurkinjeCell 
    LTP1 =  1.5e-2
    LTD1 = -1e-1
    # MossyFiber - DeepCerebellarNuclei
    LTP2 =  1e-5
    LTD2 = -1e-6
    # PurkinjeCell - DeepCerebellarNuclei
    LTP3 =  1e-7
    LTD3 =  1e-6

    # initial synaptic weight
    Init_PFPC  = 2.0
    Init_MFDCN = 0.05
    Init_PCDCN =-0.25

    syn_delay = 0.1
    # ................................................................................................
    # -----------------------------------------------------------------------> DEFINE NEW NEURONS : 

    #       - granular cells neuron, 
    #       - purkinje neuron, 
    #       - inferior olive neurons, 
    #       - deep cerebellar nuclei neurons

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

    
    granular_neuron_param = {'tau_refrac' : 1.0,
                              'cm' : 2.0e-3, #nf
                              'v_thresh' : -40.0,
                              'v_reset' : -70.0,
                              'tau_m' : 2.0/0.2,#'g_L' : 0.2, C_m/g_L
                             #'tau_m' : 10.0,
                              'tau_syn_E' : 0.5,
                              'tau_syn_I' : 10.0}

    purkinje_neuron_param = {'tau_refrac' : 2.0,
                            'cm' : 400.0*10e-3,
                            'v_thresh' : -52.0,
                            'v_reset' : -70.0,
                            'tau_m' : 400.0/16.0,#'g_L' : 16.0, C_m/g_L
                            'tau_syn_E' : 0.5,
                           #'tau_m' : 25.0,
                            'tau_syn_I' : 1.6}
    nuclear_neuron_param = {'tau_refrac' : 1.0,
                            'cm' : 2.0*10e-3,
                            'v_thresh' : -40.0,
                            'v_reset' : -70.0,
                            'tau_m' : 2.0/0.2,#'g_L' : 0.2,
                           #'tau_m' : 10.0,
                            'tau_syn_E' : 0.5,
                            'tau_syn_I' : 10.0}

    if PLAST3:
        nuclear_neuron_param = {#'tau_minus': 30.0,
                                'tau_refrac' : 1.0,
                                'cm' : 2.0e-3,
                                'v_thresh' : -40.0,
                                'v_reset' : -70.0,
                                'tau_m' : 2.0/0.2,#'g_L' : 0.2, C_m/g_L
                               #'tau_m' : 10.0,
                                'tau_syn_E' : 0.5,
                                'tau_syn_I' : 10.0}


    olivary_neuron_param = {'tau_refrac' : 1.0,
                            'cm' : 2.0e-3,
                            'v_thresh' : -40.0,
                            'v_reset' : -70.0,
                            'tau_m' : 2.0/0.2,#'g_L' : 0.2,
                           #'tau_m' : 10.0,
                            'tau_syn_E' : 0.5,
                            'tau_syn_I' : 10.0}



    # iaf_cond_exp 
    # is an implementation of a spiking neuron using IAF dynamics with conductance-based synapses. Incoming spike events induce a post-synaptic change of conductance modelled by an exponential function. The exponential function is normalised such that an event of weight 1.0 results in a peak conductance of 1 nS. 
    granular_neuron = sim.IF_cond_exp(**granular_neuron_param)
    purkinje_neuron = sim.IF_cond_exp(**purkinje_neuron_param)
    nuclear_neuron  = sim.IF_cond_exp(**nuclear_neuron_param)
    olivary_neuron  = sim.IF_cond_exp(**olivary_neuron_param)
    
    volume_transmitter_neuron = sim.native_cell_type("volume_transmitter")(**{})
    volume_transmitter_neuron2 = sim.native_cell_type("volume_transmitter")(**{})
    
    
    
    # ................................................................................................
    # -----------------------------------------------------------------------> CREATE THE POPULATION : 
    # - MF
    # - GR
    # - PC
    # - IO
    # - DCN

#    MF = sim.Population(size = int(MF_number/2), cellclass = granular_neuron)
#    
#    GR = sim.Population(size = GR_number, cellclass = granular_neuron)
#    
#    PC = sim.Population(size = PC_number, cellclass = purkinje_neuron)
#    
#    DCN = sim.Population(size = DCN_number, cellclass = nuclear_neuron)
#    
#    IO = sim.Population(size = IO_number, cellclass = olivary_neuron)
    

    MF_j1  = sim.Population( size = MF_number,  cellclass = granular_neuron)
    MF_j2  = sim.Population( size = MF_number,  cellclass = granular_neuron)
    
    GR     = sim.Population( size = GR_number,  cellclass = granular_neuron)
    
    PC_j1  = sim.Population( size = PC_number,  cellclass = purkinje_neuron)
    PC_j2  = sim.Population( size = PC_number,  cellclass = purkinje_neuron)
    
#    DCN_j1 = sim.Population( size = DCN_number, cellclass = nuclear_neuron)
#    DCN_j2 = sim.Population( size = DCN_number, cellclass = nuclear_neuron)
#    
#    IO_j1  = sim.Population( size = IO_number,  cellclass = olivary_neuron)
#    IO_j2  = sim.Population( size = IO_number,  cellclass = olivary_neuron)    
    DCN_j1_pos = sim.Population( size = DCN_number, cellclass = nuclear_neuron)
    DCN_j1_neg = sim.Population( size = DCN_number, cellclass = nuclear_neuron)
    DCN_j2_pos = sim.Population( size = DCN_number, cellclass = nuclear_neuron)
    DCN_j2_neg = sim.Population( size = DCN_number, cellclass = nuclear_neuron)
    
    IO_j1_pos  = sim.Population( size = IO_number,  cellclass = olivary_neuron)
    IO_j1_neg  = sim.Population( size = IO_number,  cellclass = olivary_neuron)
    IO_j2_pos  = sim.Population( size = IO_number,  cellclass = olivary_neuron) 
    IO_j2_neg  = sim.Population( size = IO_number,  cellclass = olivary_neuron)     
    # Name: volume_transmitter_alberto - Node used in combination with neuromodulated synaptic plasticity. It collects all spikes emitted by the population of neurons connected to the volume transmitter and transmits the signal to a user-specific subset of synapses.
    if PLAST1:
      vt  = sim.Population( size = PC_number, cellclass = volume_transmitter_neuron)
        #vt=nest.Create("volume_transmitter_alberto",PC_number)
    if PLAST2:
      vt2 = sim.Population( size = DCN_number, cellclass = volume_transmitter_neuron2)
      #vt2=nest.Create("volume_transmitter_alberto",DCN_number)


    
    # .........................................................................
    # WEIGHTS RECORDER from GranularCell to PurkinjeCell
    # recdict2 = {"to_memory": False,
    #            "to_file":    False,
    #            "label":     "PFPC",
    #            "senders":    GR,
    #            "targets":    PC
    #            }
    #WeightPFPC = nest.Create('weight_recorder',params=recdict2)
    #WeightPFPC = WeightRecorder(sampling_interval=1.0, projection=connections)



    # .........................................................................
    # DEFINE NEW KINDS OF SYNAPSES 


    #     - MF-GR (static)
    #     - PC-DCN (static or spike time dependent)
    #     - PF-PC (static or sinusoidail spike time dependent)
    #     - IO-PC (static)
    #     - MF-DCN (static or cosinus spike time dependent)

    # stdp_synapse is a connector to create synapses with spike time dependent plasticity (as defined in [1]). Here the weight dependence exponent can be set separately for potentiation and depression. 
    
    #..........................................................................
    # -- MossyFiber - GranularCell synapses

    
    MFGR_conn_param = { "weight": 0.625,#{'distribution' : 'uniform', 'low': 0.55, 'high': 0.7}, # -> 0.75 GR fire at 7 Hz
                        "delay": syn_delay } #1.


    
    MFGR_conn_synapse_type = sim.StaticSynapse(**MFGR_conn_param)
    if debug_cereb == 1:
        print("\n \n  \n  ******************** START CREATING SYNAPSE ************ \n \n  \n ")
    # .........................................................................
    # -- ParallelFiber - PurkinjeCell synapses
    #A_minus, A_plus, Wmax, Wmin, delay, exponent, peak, tau_minus, weight
    try:
        if debug_cereb == 1:
            print("\n \n  \n  ******************** START CREATING ParallelFiber - PurkinjeCell synapses ************ \n \n  \n ")
        if PLAST1:
            PFPC_conn_sin_param = {#"weight": Init_PFPC,
                                   #"delay":  1.0,
                                   "A_minus":   LTD1,   # double - Amplitude of weight change for depression
                                   "A_plus":    LTP1,   # double - Amplitude of weight change for facilitation 
                                   "Wmin":      0.0,    # double - Minimal synaptic weight 
                                   "Wmax":      4.0e-3,    # double - Maximal synaptic weight
                                   'exponent': 20.0,
                                   'peak': 100.0}
    
            #PFPC_conn_synapse_sin_type = sim.native_synapse_type("stdp_sin_synapse")(**PFPC_conn_sin_param)
            PFPC_conn_synapse_sin_type = sim.STDPMechanism(**{})
    
        else:
            PFPC_conn_param = {"weight": Init_PFPC,
                               "delay":  syn_delay}
            PFPC_conn_synapse_type = sim.StaticSynapse(**PFPC_conn_param)  
            #PFPC_conn_synapse_type = sim.native_synapse_type("static_synapse")(**PFPC_conn_param)
            if debug_cereb == 1:
                print("\n \n  \n  ******************** ParallelFiber - PurkinjeCell synapses CREATED  ************ \n \n  \n ")
    
        # .........................................................................
        # -- PurkinjeCell - DeepCerebellarNuclei synapses
        
        if PLAST3:
            PCDCN_conn_param = {"weight": Init_PCDCN,
                                "tau_plus": 30.0,
                                 "lambda": LTP3,
                                 "alpha": LTD3/LTP3,
                                 "mu_plus": 0.0,  # Additive STDP
                                 "mu_minus": 0.0, # Additive STDP
                                 "Wmax": -0.5,
                                 "delay": syn_delay}
            PCDCN_conn_synapse_type = sim.native_synapse_type("stdp_synapse")(**PCDCN_conn_param)
    
        else:
            PCDCN_conn_param = {"weight": Init_PCDCN,
                                "delay": syn_delay}
            PCDCN_conn_synapse_type = sim.StaticSynapse(**PCDCN_conn_param)
            if debug_cereb == 1:
                print(" \n \n ******* I create PCDCN_conn_synapse_type \n "+str(PCDCN_conn_synapse_type)+" \n")
            
                print("\n \n  \n  ********************  PurkinjeCell synapses -DCN SYNAPSE ************ \n \n  \n ")

    except Exception as e:
        print("\n \n  \n  Failing in ParallelFiber - PurkinjeCell synapses"+str(e))
    # .........................................................................
    # -- MossyFiber - DeepCerebellarNuclei synapses
    try:
        print("\n \n  \n  ******************** START CREATING MossyFiber - DeepCerebellarNuclei synapses ************ \n \n  \n ")
        if PLAST2:
            # MF-DCN excitatory plastic connections - every MF is connected with every DCN
            MFDCN_conn_param = {"weight": Init_MFDCN,
                                "delay": syn_delay,
                                "mu_minus":   LTD2,   # double - Amplitude of weight change for depression
                                "mu_plus":    LTP2,   # double - Amplitude of weight change for facilitation 
                                #"Wmin":      0.0,    # double - Minimal synaptic weight 
                                "Wmax":      0.25}     # double - Maximal synaptic weight 
            MFDCN_conn_synapse_type = sim.native_synapse_type("iaf_cond_exp_cos")(**{})#MFDCN_conn_param) #stdp_synapse_cosexp
            MFDCN_conn_synapse_type = sim.STDPMechanism(**{})
        else:
            MFDCN_conn_param = {"weight": Init_MFDCN,
                                "delay":  syn_delay}
            MFDCN_conn_synapse_type = sim.StaticSynapse(**MFDCN_conn_param)
    
        # .........................................................................
        # -- InferiorOlive - PurkinjeCell static synapses 
        IOPC_conn_param = {"weight": 1.0,
                               "delay": syn_delay}
        IOPC_conn_synapse_type = sim.StaticSynapse(**IOPC_conn_param)
        
        
        # .........................................................................
        # -- InferiorOlive - DeepCerebellarNuclei static synapses 
        IODCN_conn_param = {"weight": 1.0,
                               "delay": syn_delay}
        IODCN_conn_synapse_type = sim.StaticSynapse(**IODCN_conn_param)
        if debug_cereb == 1:
            print("\n \n  \n  ********************  MossyFiber - DeepCerebellarNuclei synapses created ************ \n \n  \n ")
    except Exception as e:
        print("\n \n  \n  Failing in  MossyFiber - DeepCerebellarNuclei synapses"+str(e))

    
    
    # ................................................................................................
    # -----------------------------------------------------------------------> CREATE THE PROJECTIONS: 
    
    # .........................................................................
    try:
        if debug_cereb == 1:
            print("\n \n  \n  ******************** START CREATING  MossyFiber - GranularCell PROJECTION ************ \n \n  \n ")
        # -- MossyFiber - GranularCell 
        # MF-GR excitatory fixed connections - each GR receives 4 connections from 4 random granule cells
        #nest.Connect(MF,GR,{'rule': 'fixed_indegree', 'indegree': 4, "multapses": False},MFGR_conn_param)
        #connector = sim.FixedProbabilityConnector(p_connect=0.04)
        # FixedNumberPreConnector = Each post-synaptic neuron is connected to exactly n pre-synaptic neurons chosen at random.
        #FixedNumberPostConnector = Each pre-synaptic neuron is connected to exactly n post-synaptic neurons chosen at random
        MFGR_conn_j1 = sim.Projection(  presynaptic_population = MF_j1,
                                        postsynaptic_population  = GR,
                                        connector                = sim.FixedNumberPostConnector(4, allow_self_connections=False),#sim.FixedNumberPostConnector(int(0.04*MF_number), allow_self_connections=False),  #each GC receives inputs from just a few MF (select number, now it is 4)
                                        synapse_type             = MFGR_conn_synapse_type,
                                        receptor_type            = 'excitatory')
                                    
        MFGR_conn_j2 = sim.Projection(  presynaptic_population = MF_j2,
                                        postsynaptic_population  = GR,
                                        connector                = sim.FixedNumberPostConnector(4, allow_self_connections=False),#sim.FixedNumberPostConnector(int(0.04*MF_number), allow_self_connections=False), #each GC receives inputs from just a few MF (select number, now it is 4)
                                        synapse_type             = MFGR_conn_synapse_type,
                                        receptor_type            = 'excitatory')
        #MFGR_conn.get()
        if debug_cereb == 1:
            print("\n \n  \n  ******************** MossyFiber - GranularCell PROJECTION CREATED ************ \n \n  \n ")
    except Exception as e:
        print("\n \n  \n  Failing in projecting mossy with granular"+str(e))
    # .........................................................................                          
    # PC-DCN inhibitory plastic connections - each DCN receives 2 connections from 2 contiguous PC
    #import ipdb 
    #ipdb.set_trace()
    try:
        if debug_cereb == 1:
            print("\n \n  \n  ******************** START CREATING  PC DCN PROJECTION ************ \n \n  \n ")
        count_DCN=0
        for P in range(PC_number):
            PCi_j1 = sim.PopulationView( PC_j1, np.array([P]))
            PCiDCi_cnni_j1_neg = sim.Projection( presynaptic_population  = PCi_j1,
                                             postsynaptic_population = sim.PopulationView(DCN_j1_neg, np.array([count_DCN])),
                                             connector               = sim.OneToOneConnector(),
                                             synapse_type             = PCDCN_conn_synapse_type,
                                             receptor_type            = 'excitatory')
            PCiDCi_cnni_j1_pos = sim.Projection( presynaptic_population  = PCi_j1,
                                             postsynaptic_population = sim.PopulationView(DCN_j1_pos, np.array([count_DCN])),
                                             connector               = sim.OneToOneConnector(),
                                             synapse_type             = PCDCN_conn_synapse_type,
                                             receptor_type            = 'excitatory')
            if debug_cereb == 1:
                print("After projection")#+str(PCiDCi_cnni_j1))
            
            PCi_j2 = sim.PopulationView( PC_j2, np.array([P]))
            if debug_cereb == 1:            
                print("After pci_j2")
            PCiDCi_cnni_j2_neg = sim.Projection( presynaptic_population  = PCi_j2,
                                             postsynaptic_population = sim.PopulationView(DCN_j2_neg, np.array([count_DCN])),
                                             connector               = sim.OneToOneConnector(),
                                             synapse_type             = PCDCN_conn_synapse_type,
                                             receptor_type            = 'excitatory')
            PCiDCi_cnni_j2_pos = sim.Projection( presynaptic_population  = PCi_j2,
                                             postsynaptic_population = sim.PopulationView(DCN_j2_pos, np.array([count_DCN])),
                                             connector               = sim.OneToOneConnector(),
                                             synapse_type             = PCDCN_conn_synapse_type,
                                             receptor_type            = 'excitatory')
            if debug_cereb == 1:
                print("After projection")#+str(PCiDCi_cnni_j2))            
            #PCiDCi_cnni.get()
            #nest.SetStatus(PCiDCi_cnni,{'vt': [vt[count_DCN]]})
            
            if PLAST2:
                PCVT2_conn_param = {"weight": 1.0,
                                    "delay": syn_delay}
                PCVT2_conn_synapse_type = sim.native_synapse_type("static_synapse")(**PCVT2_conn_param)
                
                PCiDCNi_cnni_j1 = sim.Projection( presynaptic_population  = PCi_j1,
                                                  postsynaptic_population = sim.PopulationView(vt2, np.array([count_DCN])),
                                                  connector               = sim.OneToOneConnector(),
                                                  synapse_type            = PCVT2_conn_synapse_type,
                                                  receptor_type           = 'excitatory')
                
                PCiDCNi_cnni_j2 = sim.Projection(presynaptic_population  = PCi_j2,
                                                 postsynaptic_population = sim.PopulationView(vt2, np.array([count_DCN])),
                                                 connector               = sim.OneToOneConnector(),
                                                 synapse_type            = PCVT2_conn_synapse_type,
                                                 receptor_type           = 'excitatory')
        
            if P%2 == 1:
                count_DCN += 1
        if debug_cereb == 1:
            print("\n \n  \n  ******************** PC DCN PROJECTION CREATED ************ \n \n  \n ") 
        #PCDCN_conn = nest.GetConnections(PC,DCN)       
    
        #nest.SetStatus(PCDCN_conn,{'vt': vt})
    except Exception as e:
        print("\n \n  \n  Failing in projecting purkinje with dcn"+str(e))
    
    # .........................................................................
    # -- ParallelFiber - PurkinjeCell projections
    try:
        if debug_cereb == 1:
            print("\n \n  \n  ******************** START CREATING  PF PC PROJECTION ************ \n \n  \n ")
        if PLAST1:
            # PF-PC excitatory plastic connections - each PC receives the random 80% of the GR
            for i in range(PC_number):
                PCi_j1 = sim.PopulationView(PC_j1, np.array([i]))
                
                GRPCi_conn = sim.Projection(presynaptic_population  = GR,
                                            postsynaptic_population = PCi_j1,
                                            connector               = sim.FixedProbabilityConnector(p_connect=int(0.8*GR_number)),
                                            synapse_type            = PFPC_conn_synapse_type,
                                            receptor_type           = 'excitatory'
                                            )
                PCi_j2 = sim.PopulationView(PC_j2, np.array([i]))
                
                GRPCi_conn = sim.Projection(presynaptic_population  = GR,
                                            postsynaptic_population = PCi_j2,
                                            connector               = sim.FixedProbabilityConnector(p_connect=int(0.8*GR_number)),
                                            synapse_type            = PFPC_conn_synapse_type,
                                            receptor_type           = 'excitatory'
                                            )
                #GRPCi_conn.set(U= vt[i]) # Wmax, alpha, delay, lambda, mu_minus, mu_plus, tau_minus, tau_plus, weight
        else:
            GRPC_conn_j1 = sim.Projection(  presynaptic_population  = GR,
                                            postsynaptic_population = PC_j1,
                                            connector               = sim.FixedProbabilityConnector(p_connect=int(0.8*GR_number)),
                                            synapse_type            = PFPC_conn_synapse_type,
                                            receptor_type           = 'excitatory')
            GRPC_conn_j2 = sim.Projection(  presynaptic_population  = GR,
                                            postsynaptic_population = PC_j2,
                                            connector               = sim.FixedProbabilityConnector(p_connect=int(0.8*GR_number)),
                                            synapse_type            = PFPC_conn_synapse_type,
                                            receptor_type           = 'excitatory')
        #PFPC_conn = nest.GetConnections(GR,PC)
        if debug_cereb == 1:
            print("\n \n  \n  ******************** PF PC PROJECTION created ************ \n \n  \n ")
    except Exception as e:
        print("\n \n  \n  Failing in projecting parallel fiber to purkinje"+str(e))
    
    # .........................................................................
    # -- MossyFiber - DeepCerebellarNuclei projections
    try:
        if debug_cereb == 1:
            print("\n \n  \n  ******************** START CREATING  MOSSY DCN PROJECTION ************ \n \n  \n ")
        if PLAST2:
            # MF-DCN excitatory plastic connections - every MF is connected with every DCN
            
            for i in range(DCN_number):
                DCNi_j1_pos = sim.PopulationView(DCN_j1_pos, np.array([i]))
                MFDCNi_cnni_j1_pos = sim.Projection(presynaptic_population  = MF_j1,
                                                    postsynaptic_population = DCNi_j1_pos,
                                                    connector               = sim.AllToAllConnector(),
                                                    synapse_type            = MFDCN_conn_synapse_type,
                                                    receptor_type           = 'excitatory')
                DCNi_j1_neg = sim.PopulationView(DCN_j1_neg, np.array([i]))
                MFDCNi_cnni_j1_neg = sim.Projection(presynaptic_population  = MF_j1,
                                                    postsynaptic_population = DCNi_j1_neg,
                                                    connector               = sim.AllToAllConnector(),
                                                    synapse_type            = MFDCN_conn_synapse_type,
                                                    receptor_type           = 'excitatory')
                DCNi_j2_pos = sim.PopulationView(DCN_j2_pos, np.array([i]))
                MFDCNi_cnni_j2_pos = sim.Projection(presynaptic_population  = MF_j2,
                                                    postsynaptic_population = DCNi_j2_pos,
                                                    connector               = sim.AllToAllConnector(),
                                                    synapse_type            = MFDCN_conn_synapse_type,
                                                    receptor_type           = 'excitatory')
                DCNi_j2_neg = sim.PopulationView(DCN_j2_neg, np.array([i]))
                MFDCNi_cnni_j2_neg = sim.Projection(presynaptic_population  = MF_j2,
                                                    postsynaptic_population = DCNi_j2_neg,
                                                    connector               = sim.AllToAllConnector(),
                                                    synapse_type            = MFDCN_conn_synapse_type,
                                                    receptor_type           = 'excitatory')
                #MFDCNi_cnni.set({U=vt2[i])
        else:
            MFDCN_cnn_j1_pos = sim.Projection(presynaptic_population = MF_j1,
                                                postsynaptic_population = DCN_j1_pos,
                                                connector = sim.AllToAllConnector(),
                                                synapse_type = MFDCN_conn_synapse_type,
                                                receptor_type='excitatory')
            MFDCN_cnn_j1_neg = sim.Projection(presynaptic_population = MF_j1,
                                                postsynaptic_population = DCN_j1_neg,
                                                connector = sim.AllToAllConnector(),
                                                synapse_type = MFDCN_conn_synapse_type,
                                                receptor_type='excitatory')
            MFDCN_cnn_j2_pos = sim.Projection( presynaptic_population   = MF_j2,
                                                postsynaptic_population = DCN_j2_pos,
                                                connector               = sim.AllToAllConnector(),
                                                synapse_type            = MFDCN_conn_synapse_type,
                                                receptor_type           = 'excitatory')                        
            MFDCN_cnn_j2_neg = sim.Projection( presynaptic_population   = MF_j2,
                                                postsynaptic_population = DCN_j2_neg,
                                                connector               = sim.AllToAllConnector(),
                                                synapse_type            = MFDCN_conn_synapse_type,
                                                receptor_type           = 'excitatory')
    
        if debug_cereb == 1:
            print("\n \n  \n  ********************  MOSSY DCN PROJECTION CREATED************ \n \n  \n ")
    except Exception as e:
        print("\n \n  \n  Failing in projecting mossy with DCN"+str(e))
    
    # .........................................................................
    # -- InferiorOlive - PurkinjeCell projections
    try:
        print("\n \n  \n  ******************** START CREATING  IO PC PROJECTION ************ \n \n  \n ")
        # IO-PC teaching connections - Each IO is one-to-one connected with each PC
        if PLAST1:
            IOPC_cnn_j1_pos = sim.Projection(   presynaptic_population  = IO_j1_pos,
                                                postsynaptic_population = vt,
                                                connector               = sim.OneToOneConnector(),
                                                synapse_type            = IOPC_conn_synapse_type,
                                                receptor_type           = 'excitatory')
            IOPC_cnn_j1_neg = sim.Projection(   presynaptic_population  = IO_j1_neg,
                                                postsynaptic_population = vt,
                                                connector               = sim.OneToOneConnector(),
                                                synapse_type            = IOPC_conn_synapse_type,
                                                receptor_type           = 'excitatory')
            IOPC_cnn_j2_pos = sim.Projection(  presynaptic_population   = IO_j2_pos,
                                                postsynaptic_population = vt,
                                                connector               = sim.OneToOneConnector(),
                                                synapse_type            = IOPC_conn_synapse_type,
                                                receptor_type           = 'excitatory')                                
            IOPC_cnn_j2_neg = sim.Projection(  presynaptic_population   = IO_j2_neg,
                                                postsynaptic_population = vt,
                                                connector               = sim.OneToOneConnector(),
                                                synapse_type            = IOPC_conn_synapse_type,
                                                receptor_type           = 'excitatory')
        
        # .........................................................................
        # -- InferiorOlive - DeepCerebellarNuclei projections                                                
        IODCN_cnn_j1_pos = sim.Projection(  presynaptic_population  = IO_j1_pos,
                                            postsynaptic_population = DCN_j1_pos,
                                            connector               = sim.AllToAllConnector(),
                                            synapse_type            = IODCN_conn_synapse_type,
                                            receptor_type           = 'excitatory') 
        IODCN_cnn_j1_neg = sim.Projection(  presynaptic_population  = IO_j1_neg,
                                            postsynaptic_population = DCN_j1_neg,
                                            connector               = sim.AllToAllConnector(),
                                            synapse_type            = IODCN_conn_synapse_type,
                                            receptor_type           = 'excitatory')   
                                        
        IODCN_cnn_j2_pos = sim.Projection(  presynaptic_population  = IO_j2_pos,
                                            postsynaptic_population = DCN_j2_pos,
                                            connector               = sim.AllToAllConnector(),
                                            synapse_type            = IODCN_conn_synapse_type,
                                            receptor_type           = 'excitatory') 
        IODCN_cnn_j2_neg = sim.Projection(  presynaptic_population  = IO_j2_neg,
                                            postsynaptic_population = DCN_j2_neg,
                                            connector               = sim.AllToAllConnector(),
                                            synapse_type            = IODCN_conn_synapse_type,
                                            receptor_type           = 'excitatory')  
        if debug_cereb == 1:
            print("\n \n  \n  ********************   IO PC PROJECTION CREATED ************ \n \n  \n ")
    except Exception as e:
        print("\n \n  \n Failing in projecting IO with purkinje"+str(e))
    
    '''
    sim.initialize(MF, v = MF.get('v_rest'))
    sim.initialize(GR, v = GR.get('v_rest'))
    sim.initialize(PC, v = PC.get('v_rest'))
    sim.initialize(DCN, v = DCN.get('v_rest'))
    sim.initialize(IO, v = IO.get('v_rest'))
    '''
    
    return MF_j1, MF_j2, GR, PC_j1, PC_j2, IO_j1_pos, IO_j1_neg, IO_j2_pos, IO_j2_neg, DCN_j1_pos, DCN_j1_neg, DCN_j2_pos, DCN_j2_neg

MOSSY_j1, MOSSY_j2, GRANULAR, PURKINJE_j1, PURKINJE_j2, INFOLIVE_j1_pos, INFOLIVE_j1_neg , INFOLIVE_j2_pos, INFOLIVE_j2_neg,  DCN_j1_pos, DCN_j1_neg, DCN_j2_pos, DCN_j2_neg = create_brain()
