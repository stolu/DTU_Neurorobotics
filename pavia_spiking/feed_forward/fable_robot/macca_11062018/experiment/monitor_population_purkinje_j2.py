# This specifies that the neurons of the motor population
# should be monitored. You can see them in the spike train widget
@nrp.NeuronMonitor(nrp.brain.PURKINJE_j2, nrp.spike_recorder)
def monitor_population_purkinje2(t):
    # Uncomment to log into the 'log-console' visible in the simulation
    # clientLogger.info("Time: ", t)
    return True
