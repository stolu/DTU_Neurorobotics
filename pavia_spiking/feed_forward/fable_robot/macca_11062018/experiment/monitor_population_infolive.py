# This specifies that the neurons of the motor population
# should be monitored. You can see them in the spike train widget
@nrp.NeuronMonitor(nrp.brain.INFOLIVE_j1 + nrp.brain.INFOLIVE_j2, nrp.spike_recorder)
def monitor_population_io1pos(t):
    # Uncomment to log into the 'log-console' visible in the simulation
    # clientLogger.info("Time: ", t)
    return True
