# This specifies that the neurons of the motor population
# should be monitored. You can see them in the spike train widget
#@nrp.NeuronMonitor(nrp.brain.MF_0_0_des_q +nrp.brain.MF_0_1_des_q+ nrp.brain.IO_0_0_pos+ nrp.brain.IO_0_0_neg + nrp.brain.IO_0_1_pos + nrp.brain.IO_0_1_neg, nrp.spike_recorder)
#@nrp.NeuronMonitor(nrp.brain.MF_0_0_des_q+ nrp.brain.MF_0_0_des_qd+ nrp.brain.MF_0_1_des_q+ nrp.brain.MF_0_1_des_qd+ nrp.brain.MF_0_0_cur_q+ nrp.brain.MF_0_1_cur_q+ nrp.brain.GC+ nrp.brain.PC_0_0_pos+ nrp.brain.PC_0_0_neg+ nrp.brain.PC_0_1_pos+ nrp.brain.PC_0_1_neg+ nrp.brain.IO_0_0_pos+ nrp.brain.IO_0_0_neg+ nrp.brain.IO_0_1_pos+ nrp.brain.IO_0_1_neg+ nrp.brain.DCN_0_0_pos+ nrp.brain.DCN_0_0_neg+ nrp.brain.DCN_0_1_pos+ nrp.brain.DCN_0_1_neg, nrp.spike_recorder)
@nrp.NeuronMonitor(nrp.brain.MF_0_0_cur_q_plus+ nrp.brain.MF_0_0_cur_q_minus+nrp.brain.MF_0_0_des_q+nrp.brain.MF_0_0_des_torque +nrp.brain.MF_0_1_cur_q_plus+nrp.brain.MF_0_1_cur_q_minus+nrp.brain.MF_0_1_des_q+nrp.brain.MF_0_1_des_torque+nrp.brain.IO_0_0_pos+ nrp.brain.IO_0_0_neg+ nrp.brain.IO_0_1_pos+ nrp.brain.IO_0_1_neg+ nrp.brain.DCN_0_0_pos+ nrp.brain.DCN_0_0_neg+ nrp.brain.DCN_0_1_pos+ nrp.brain.DCN_0_1_neg, nrp.spike_recorder)
def monitor_population_mossy1(t):
    # Uncomment to log into the 'log-console' visible in the simulation
    # clientLogger.info("Time: ", t)
    return True
