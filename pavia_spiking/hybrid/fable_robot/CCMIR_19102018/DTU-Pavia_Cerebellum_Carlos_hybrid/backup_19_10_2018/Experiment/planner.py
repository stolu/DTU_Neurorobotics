# Imported Python Transfer Function
import numpy as np
from std_msgs.msg    import Float64,Float64MultiArray, Bool
from sensor_msgs.msg import JointState
from numpy           import pi as pi
import sys
sys.path.append("/home/dtu-neurorobotics/Documents/NRP/Experiments/fable_manipulation_dtu_spiking_cereb_hybrid")
from rad_bas_f import RBF
j_des  = sensor_msgs.msg.JointState()
ee_des = std_msgs.msg.Float64MultiArray()


# ** Encoding Mossy input **
# encoding = RBF(total_n_neurons, min_in, max_in, min_out, max_out)


rbf_mossy_q_out_max = 0.25
rbf_mossy_q_out_min = 0.00001
rbf_mossy_q_min_input_range    = - np.pi*0.5
rbf_mossy_q_max_input_range    = + np.pi*0.5

rbf_mossy_qd_out_max = 0.25
rbf_mossy_qd_out_min = 0.00001
rbf_mossy_qd_min_input_range    = - 8.753#np.pi*0.5/0.0085
rbf_mossy_qd_max_input_range    = + 8.7520#np.pi*0.5/0.0085

rbf_mossy_torque_out_max         = 0.25
rbf_mossy_torque_out_min         = 0.0001
rbf_mossy_torque_min_input_range = -1.56
rbf_mossy_torque_max_input_range =  1.56

#[4.3838718833537405, -8.752036251879566] speed max values
# initialization of the RBF object for IO (it does not matter which joint, this will be specified when calling the function)
rbf_mossy_q  = RBF(nrp.config.brain_root.MF_n/nrp.config.brain_root.n_inputs_MF, rbf_mossy_q_min_input_range,  rbf_mossy_q_max_input_range,  rbf_mossy_q_out_min,  rbf_mossy_q_out_max) 
rbf_mossy_torque = RBF(nrp.config.brain_root.MF_n/nrp.config.brain_root.n_inputs_MF, rbf_mossy_torque_min_input_range, rbf_mossy_torque_max_input_range, rbf_mossy_torque_out_min, rbf_mossy_torque_out_max) 

rbf_mossy_qd = RBF(nrp.config.brain_root.MF_n/nrp.config.brain_root.n_inputs_MF, rbf_mossy_qd_min_input_range, rbf_mossy_qd_max_input_range, rbf_mossy_qd_out_min, rbf_mossy_qd_out_max) 
#rbf_mossy_qd = RBF(nrp.config.brain_root.MF_n/nrp.config.brain_root.n_inputs_MF, rbf_mossy_qd_min_input_range, rbf_mossy_qd_max_input_range, rbf_mossy_qd_out_min, rbf_mossy_qd_out_max) 
# -----------------------------------------------------------------------#
# **  map **
@nrp.MapVariable("debug_traj", 			 initial_value = 0)
@nrp.MapVariable("desired_joints", 		 initial_value = j_des)
@nrp.MapVariable("desired_end_effector",    initial_value = ee_des)
@nrp.MapVariable("can_plan", 			 initial_value = True)
@nrp.MapVariable("time", 			      initial_value = 0.)
@nrp.MapVariable("n_joints",                initial_value = 2)
# ** Input the trajectory type: 1) infinity loop; 2) circle; 3) fixed
@nrp.MapVariable("trajectory_type",   initial_value = 1)
# ** Input the fixed reference
@nrp.MapVariable("fixed_ref_x",       initial_value = np.deg2rad(45.))
@nrp.MapVariable("fixed_ref_y",       initial_value = np.deg2rad(45.))

# data encoding Recurrent
@nrp.MapVariable("radial_func_q_r",     initial_value = rbf_mossy_q )
@nrp.MapVariable("radial_func_torque_r",  initial_value = rbf_mossy_torque) #same mapping for PLUS and MINUS current positions
# data encoding Feedforward
@nrp.MapVariable("radial_func_q_ff",     initial_value = rbf_mossy_q )
@nrp.MapVariable("radial_func_qd_ff",    initial_value = rbf_mossy_qd )

#@nrp.MapVariable("radial_func_qd",    initial_value = rbf_mossy_qd )

# -----------------------------------------------------------------------#
# **  MOSSY FIBERS INPUTS  **
#Recurrent
@nrp.MapSpikeSource("MF_0_0_des_q_recurrent",  nrp.map_neurons(range(0, nrp.config.brain_root.MF_n/nrp.config.brain_root.n_inputs_MF), lambda i: nrp.brain.MF_0_0_des_q_recurrent[i]),  nrp.ac_source)
@nrp.MapSpikeSource("MF_0_1_des_q_recurrent",  nrp.map_neurons(range(0, nrp.config.brain_root.MF_n/nrp.config.brain_root.n_inputs_MF), lambda i: nrp.brain.MF_0_1_des_q_recurrent[i]),  nrp.ac_source)
@nrp.MapSpikeSource("MF_0_0_des_torque_recurrent",  nrp.map_neurons(range(0, nrp.config.brain_root.MF_n/nrp.config.brain_root.n_inputs_MF), lambda i: nrp.brain.MF_0_0_des_torque_recurrent[i]),  nrp.ac_source)
@nrp.MapSpikeSource("MF_0_1_des_torque_recurrent",  nrp.map_neurons(range(0, nrp.config.brain_root.MF_n/nrp.config.brain_root.n_inputs_MF), lambda i: nrp.brain.MF_0_1_des_torque_recurrent[i]),  nrp.ac_source)

#Feedforward
@nrp.MapSpikeSource("MF_0_0_des_q_feedforward",  nrp.map_neurons(range(0, nrp.config.brain_root.MF_n/nrp.config.brain_root.n_inputs_MF), lambda i: nrp.brain.MF_0_0_des_q_feedforward[i]),  nrp.ac_source)
@nrp.MapSpikeSource("MF_0_0_des_qd_feedforward", nrp.map_neurons(range(0, nrp.config.brain_root.MF_n/nrp.config.brain_root.n_inputs_MF), lambda i: nrp.brain.MF_0_0_des_qd_feedforward[i]),  nrp.ac_source)
@nrp.MapSpikeSource("MF_0_1_des_q_feedforward",  nrp.map_neurons(range(0, nrp.config.brain_root.MF_n/nrp.config.brain_root.n_inputs_MF), lambda i: nrp.brain.MF_0_1_des_q_feedforward[i]),  nrp.ac_source)
@nrp.MapSpikeSource("MF_0_1_des_qd_feedforward", nrp.map_neurons(range(0, nrp.config.brain_root.MF_n/nrp.config.brain_root.n_inputs_MF), lambda i: nrp.brain.MF_0_1_des_qd_feedforward[i]),  nrp.ac_source)

# -----------------------------------------------------------------------#
# ** publish on ros topics **

# ** Desired end-effector trajectory **
@nrp.MapRobotPublisher( "desired_ee_trajectory_pub"     , Topic('/robot/desired_trajectory/end_effector', std_msgs.msg.Float64MultiArray))

# ** Desired Joints trajectory **
@nrp.MapRobotPublisher( "desired_joints_trajectory_pub" , Topic('/robot/desired_trajectory/joints',       sensor_msgs.msg.JointState))
@nrp.MapRobotPublisher( "plan_pub",                       Topic('/robot/plan',                            std_msgs.msg.Bool))
@nrp.MapRobotSubscriber("plan_sub",                       Topic('/robot/plan' ,                           std_msgs.msg.Bool))

@nrp.MapRobotSubscriber("desired_LF_torque", Topic('/robot/desired_LF_torque/joints',       std_msgs.msg.Float64MultiArray))

@nrp.Robot2Neuron()

def planner(t, debug_traj, trajectory_type, n_joints,
			desired_end_effector, desired_joints, 
			fixed_ref_x, fixed_ref_y, 
			plan_sub, plan_pub, can_plan, time,
			MF_0_0_des_q_recurrent, MF_0_1_des_q_recurrent, 
			MF_0_0_des_torque_recurrent, MF_0_1_des_torque_recurrent,
			MF_0_0_des_q_feedforward, MF_0_0_des_qd_feedforward, 
			MF_0_1_des_q_feedforward, MF_0_1_des_qd_feedforward,
			radial_func_q_r, radial_func_torque_r, radial_func_q_ff, radial_func_qd_ff , 
			desired_ee_trajectory_pub, desired_joints_trajectory_pub, desired_LF_torque):	
	# ** oscillator parameters **
	amplitude = np.deg2rad(40)
	period    = 10
	frequency = 2*np.pi/period
	dt = 0.0085#0.02#0.04#.009#0.01
	desired_joints.value.position = []
	desired_joints.value.velocity = []
	desired_joints.value.effort   = []
	desired_end_effector.value.data = []
	
	MF_joint_q_des_recurrent  = [MF_0_0_des_q_recurrent, MF_0_1_des_q_recurrent]
	MF_joint_des_torque_recurrent = [MF_0_0_des_torque_recurrent,  MF_0_1_des_torque_recurrent]

	MF_joint_q_des_feedforward = [MF_0_0_des_q_feedforward, MF_0_1_des_q_feedforward]
	MF_joint_qd_des_feedforward = [MF_0_0_des_qd_feedforward, MF_0_1_des_qd_feedforward]
	
	try:
		can_plan.value = plan_sub.value.data
		#clientLogger.info(" ------> "+ str(can_plan.value))
	except Exception as e:
		clientLogger.info(" --> Planner publisher: "+str(e))
	if can_plan.value == True:
		if trajectory_type.value == 1:
			if t < 0.01:
				 clientLogger.info("trajectory : infinity loop with amplitude "+str(np.rad2deg(amplitude))+" deg" )
			phase = np.rad2deg(0.)
			scale = 2 / (3 - np.cos( 2* (frequency*time.value + phase) ))
			desired_end_effector.value.data.append(scale * np.cos(np.deg2rad(frequency*time.value + phase)))
			desired_end_effector.value.data.append(scale * np.sin(np.deg2rad(frequency*time.value + phase))/ 2)
			
			phase_j2 = np.deg2rad(90)
			
			desired_joints.value.effort.append(   -4*np.power(np.pi, 2) *amplitude * np.sin(2 * np.pi *time.value)) #"accelleration"
			desired_joints.value.velocity.append(  2*np.pi*amplitude * np.cos(2 * np.pi *time.value) )
			desired_joints.value.position.append(  amplitude * np.sin(2 * np.pi *time.value) )
			
			desired_joints.value.effort.append(   -16*np.power(np.pi, 2) *amplitude * np.cos(4* np.pi *time.value + phase_j2)) #"accelleration"
			desired_joints.value.velocity.append( -4*np.pi*amplitude * np.sin(4 * np.pi *time.value + phase_j2) )
			desired_joints.value.position.append( amplitude * np.cos( 4* np.pi *time.value + phase_j2))
			
			#clientLogger.info("desired_joints: "+str(desired_joints.value.position))
		elif trajectory_type.value == 2:
			if t < 0.01:
				clientLogger.info("trajectory : circle with amplitude "+str(np.rad2deg(amplitude))+" deg")
			phase_j2 = np.deg2rad(90)
			desired_end_effector.value.data.append( amplitude*np.cos(np.deg2rad(frequency*time.value )) )
			desired_end_effector.value.data.append( amplitude*np.cos(np.deg2rad(frequency*time.value + phase_j2)) )
			desired_joints.value.effort.append(    -4*np.power(np.pi, 2) *amplitude * np.cos(2 * np.pi *time.value)) #"accelleration"
			desired_joints.value.velocity.append(  -2*np.pi*amplitude * np.sin(2 * np.pi *time.value) )
			desired_joints.value.position.append(  amplitude * np.cos(2 * np.pi *time.value) )
			desired_joints.value.effort.append(	  -4*np.power(np.pi, 2) *amplitude * np.cos(2 * np.pi *time.value + phase_j2)) #"accelleration"
			desired_joints.value.velocity.append(  -2*np.pi*amplitude * np.sin(2 * np.pi *time.value + phase_j2) )
			desired_joints.value.position.append(  amplitude * np.cos(2 * np.pi *time.value + phase_j2))    

		elif trajectory_type.value == 3:
			if t < 0.01:
				clientLogger.info("trajectory :  steady reference with amplitude ("+str(fixed_ref_x.value)+","+str(fixed_ref_x.value)+")")
			desired_end_effector.value.data.append( amplitude*np.cos(fixed_ref_x.value) )
			desired_end_effector.value.data.append( amplitude*np.sin(fixed_ref_y.value) )
			desired_joints.value.effort.append(0.)
			desired_joints.value.velocity.append(0.)
			desired_joints.value.position.append(fixed_ref_x.value)
			desired_joints.value.effort.append(0.)
			desired_joints.value.velocity.append(0. )
			desired_joints.value.position.append(fixed_ref_y.value )

		else:
			clientLogger.info("** No trajectory type selected! **")

		if debug_traj.value == 1:
			clientLogger.info("desired end-effector: "+str(desired_end_effector.value.data))
			clientLogger.info("desired joints: "+str(desired_joints.value.position))
		try:
			# send the information as amplitude of a current to the mossy fibers
			for k in range(0, n_joints.value):
				des_q_mossy_r  = radial_func_q_r.value.function(desired_joints.value.position[k])
				des_q_mossy_ff  = radial_func_q_ff.value.function(desired_joints.value.position[k]) 
				des_qd_mossy_ff = radial_func_qd_ff.value.function(desired_joints.value.velocity[k])

				if t==0:
					des_torque_mossy_r  = radial_func_torque_r.value.function(0.0)
				else:
					des_torque_mossy_r  = radial_func_torque_r.value.function(desired_LF_torque.value.data[k])
				
				#Recurrent inputs
				m = 0
				for neuron_MFqdes  in MF_joint_q_des_recurrent[k]:
					neuron_MFqdes.amplitude  = des_q_mossy_r[m] 
					m = m + 1

				m = 0
				for neuron_MFqdes_torque  in MF_joint_des_torque_recurrent[k]:
					neuron_MFqdes_torque.amplitude  = des_torque_mossy_r[m] 
					m = m + 1

				#Feedforward inputs
				m = 0
				for neuron_MFqdes  in MF_joint_q_des_feedforward[k]:
					neuron_MFqdes.amplitude  = des_q_mossy_ff[m] 
					m = m + 1

				m = 0
				for neuron_MFqddes in MF_joint_qd_des_feedforward[k]:
					neuron_MFqddes.amplitude = des_qd_mossy_ff[m]
					m = m + 1

			# ** Publish desired Joints value **
			desired_joints_trajectory_pub.send_message(desired_joints.value)
			# ** Publish desired End-effector position **
			desired_ee_trajectory_pub.send_message(desired_end_effector.value)
			can_plan.value = False
			plan_pub.send_message(False)
			time.value = time.value + dt

		except Exception as e:
			clientLogger.info(" --> Planner Exception: "+str(e))
