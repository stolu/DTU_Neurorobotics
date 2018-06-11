# This specifies that the neurons of the motor population
# should be monitored. You can see them in the spike train widget
@nrp.NeuronMonitor(nrp.brain.MOSSY_j1 +nrp.brain.MOSSY_j2+ nrp.brain.INFOLIVE_j1_pos+ nrp.brain.INFOLIVE_j1_neg + nrp.brain.INFOLIVE_j2_pos + nrp.brain.INFOLIVE_j2_neg, nrp.spike_recorder)
def monitor_population_mossy1(t):
    # Uncomment to log into the 'log-console' visible in the simulation
    # clientLogger.info("Time: ", t)
    return True
