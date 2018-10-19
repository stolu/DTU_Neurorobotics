
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
pc0_pos = "PC_0_0_pos"
pc0_neg = "PC_0_0_neg"
pc0 = "PC_0_0"
gr = "GR"
io0 = "IO_0_0"
dcn0_pos = "DCN_0_0_pos"
dcn0_neg = "DCN_0_0_neg"
dcn0 = "DCN_0_0"

mf1 = "MF_0_1"
pc1_pos = "PC_0_1_pos"
pc1_neg = "PC_0_1_neg"
pc1 = "PC_0_1"
io1 = "IO_0_1"
dcn1_pos = "DCN_0_1_pos"
dcn1_neg = "DCN_0_1_neg"
dcn1 = "DCN_0_1"

test_results = CerebTestTune(str(TEST), OUTPUT_PATH, MODULES, JOINTS, MODEL)
