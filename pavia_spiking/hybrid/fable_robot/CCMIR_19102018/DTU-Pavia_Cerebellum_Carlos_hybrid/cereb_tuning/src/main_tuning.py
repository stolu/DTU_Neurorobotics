
import os
import sys
from CerebTestTune_test import CerebTestTune

dir_path = os.path.dirname(os.path.realpath(__file__))

OUTPUT_PATH = dir_path.replace("/src", "/results")

#You may change the values below if needed

TEST = int(sys.argv[1:][0])
JOINTS = ["0", "1"]
MODULES = 1
MODEL = {"model":"dtu_pavia_simple_cereb", 
         "MF_number": 120, # Must be multiple of n_inputs
         "GR_number": 750,
         "PC_number": 24,  # Must be EVEN
         "IO_number": 24,  # Must be EVEN
		 "DCN_number": 12} # Must be EVEN}, 

#Recurrent
mf0_r = "MF_0_0_recurrent"
pc0_pos_r = "PC_0_0_pos_recurrent"
pc0_neg_r = "PC_0_0_neg_recurrent"
pc0_r = "PC_0_0_recurrent"
gr_r = "GR_recurrent"
io0_r = "IO_0_0"
dcn0_pos_r = "DCN_0_0_pos_recurrent"
dcn0_neg_r = "DCN_0_0_neg_recurrent"
dcn0_r = "DCN_0_0_recurrent"

mf1_r = "MF_0_1_recurrent"
pc1_pos_r = "PC_0_1_pos_recurrent"
pc1_neg_r = "PC_0_1_neg_recurrent"
pc1_r = "PC_0_1_recurrent"
io1_r = "IO_0_1_recurrent"
dcn1_pos_r = "DCN_0_1_pos_recurrent"
dcn1_neg_r = "DCN_0_1_neg_recurrent"
dcn1_r = "DCN_0_1_recurrent"

#Feedforward
mf0_ff = "MF_0_0_feedforward"
pc0_pos_ff = "PC_0_0_pos_feedforward"
pc0_neg_ff = "PC_0_0_neg_feedforward"
pc0_ff = "PC_0_0_feedforward"
gr_ff = "GR_feedforward"
io0_ff = "IO_0_0_feedforward"
dcn0_pos_ff = "DCN_0_0_pos_feedforward"
dcn0_neg_ff = "DCN_0_0_neg_feedforward"
dcn0_ff = "DCN_0_0_feedforward"

mf1_ff = "MF_0_1_feedforward"
pc1_pos_ff = "PC_0_1_pos_feedforward"
pc1_neg_ff = "PC_0_1_neg_feedforward"
pc1_ff = "PC_0_1_feedforward"
io1_ff = "IO_0_1_feedforward"
dcn1_pos_ff = "DCN_0_1_pos_feedforward"
dcn1_neg_ff = "DCN_0_1_neg_feedforward"
dcn1_ff = "DCN_0_1_feedforward"



test_results = CerebTestTune(str(TEST), OUTPUT_PATH, MODULES, JOINTS, MODEL)
