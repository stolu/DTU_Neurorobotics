
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


mf0 = "MF_0_0"
pc0_pos_q = "PC_0_0_pos_q"
pc0_neg_q = "PC_0_0_neg_q"
pc0_pos_qd = "PC_0_0_pos_qd"
pc0_neg_qd = "PC_0_0_neg_qd"
pc0 = "PC_0_0"
gr = "GR"
io0 = "IO_0_0"
dcn0_pos_q = "DCN_0_0_pos_q"
dcn0_neg_q = "DCN_0_0_neg_q"
dcn0_pos_qd = "DCN_0_0_pos_qd"
dcn0_neg_qd = "DCN_0_0_neg_qd"
dcn0 = "DCN_0_0"

mf1 = "MF_0_1"
pc1_pos_q = "PC_0_1_pos_q"
pc1_neg_q = "PC_0_1_neg_q"
pc1_pos_qd = "PC_0_1_pos_qd"
pc1_neg_qd = "PC_0_1_neg_qd"
pc1 = "PC_0_1"
io1 = "IO_0_1"
dcn1_pos_q = "DCN_0_1_pos_q"
dcn1_neg_q = "DCN_0_1_neg_q"
dcn1_pos_qd = "DCN_0_1_pos_qd"
dcn1_neg_qd = "DCN_0_1_neg_qd"
dcn1 = "DCN_0_1"

test_results = CerebTestTune(str(TEST), OUTPUT_PATH, MODULES, JOINTS, MODEL)
