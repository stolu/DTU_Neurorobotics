
import os 
from CerebTestTune_test import CerebTestTune

dir_path = os.path.dirname(os.path.realpath(__file__))

OUTPUT_PATH = dir_path.replace("/src", "/results")

#You may change the values below if needed
TEST = 5
JOINTS = ["0"]
MODULES = 1
MODEL = {"model":"dtu_pavia_simple_cereb", 
         "MF_number": 120, # Must be multiple of n_inputs
         "GR_number": 750,
         "PC_number": 24,  # Must be EVEN
         "IO_number": 24,  # Must be EVEN
		 "DCN_number": 12} # Must be EVEN}, 


mf = "MF_0_0"
pc_pos = "PC_0_0_pos"
pc_neg = "PC_0_0_neg"
gr = "GR"
io = "IO_0_0"
dcn_pos = "DCN_0_0_pos"
dcn_neg = "DCN_0_0_neg"

test_results = CerebTestTune(str(TEST), OUTPUT_PATH, MODULES, JOINTS, MODEL)
