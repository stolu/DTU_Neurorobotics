# Imported Python Transfer Function
import numpy as np
from numpy import pi as pi
from std_msgs.msg    import Float64, Float64MultiArray, Bool
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import LinkStates
from gazebo_msgs.srv import ApplyJointEffort
from rospy import ServiceProxy, wait_for_service, Duration
import sys
sys.path.append("/home/dtu-neurorobotics/Documents/robotics-toolbox-python")

import robot
global robot
#from norma import*

# -----------------------------------------------------------------------#
# ** Create robot model for inverse dynamics **
fable_a      = [ 0.,  0.06    ]
fable_alpha  = [ 0., -0.5 * pi]
fable_theta  = [ 0.,  0.      ]
fable_d 	 = [ 0.04, 0.,   0.4]
fable_m 	 = [ 1000, 0.01, 0.1]
fable_sigma  = [ 0, 0, 0] # 0 for a revolute joint, non-zero for prismatic
n_links      = 2
n_joints     = 2
jm           = [200e-3, 200e-3]
fable_links  = []
for i in range(0,n_links):
	fable_links.append( robot.Link(alpha=fable_alpha[i], A=fable_a[i], D=fable_d[i], theta=fable_theta[i], sigma=fable_sigma[i] ))
	# Denhavit convention
	fable_links[i].convention = robot.Link.LINK_DH
	# link mass
	fable_links[i].m = fable_m[i+1]
	#link COG wrt link coordinate frame 3x1
	fable_links[i].r = np.mat([ fable_a[i]/2, 0, fable_d[i]/2])
	# Inertia 
	fable_links[i].I = np.mat([0., 0.,  0.,  0.0177187,  0.0177187,  0])
	# Inertia (motor referred)
	fable_links[i].Jm =  jm[i] 
	# gear ratio
	fable_links[i].G =  1.
	# viscous friction (motor referenced)
	fable_links[i].B =   1.#0.01
	# Coulomb friction (motor referenced) 1 element if symmetric, else 2
	fable_links[i].Tc = np.mat([ 0., 0.])

fable       = robot.Robot( fable_links, name='fable', manuf='Unimation', comment='00')
joints_name = ['joint_1', 'joint_2']
# -----------------------------------------------------------------------#
# ** LF Controller Ks **
Kp = [ 0.98,  1.9   ] #[ 0.68,   0.95   ]
Kd = [ 0.004,   0.004  ] # [ 0.004,   0.009  ]
Ki = [ 0.00001, 0.00001] # [ 0.00001, 0.00001]
prev_error = [ 0., 0.]

DCNcmd_pos = std_msgs.msg.Float64MultiArray()
DCNcmd_pos.data   = [0.,0.]
DCNcmd_neg = std_msgs.msg.Float64MultiArray()
DCNcmd_neg.data   = [0.,0.]
DCNcmd_total = std_msgs.msg.Float64MultiArray()
DCNcmd_total.data   = [0.,0.]
LFcmd = std_msgs.msg.Float64MultiArray()
LFcmd.data   = [0.,0.]
dcn_gain = 1./(nrp.config.brain_root.DCN_n) #total number of neurons per layer: DCN_n_pos(DCN_n/2) + DCN_n_neg(DCN_n/2)

# -----------------------------------------------------------------------#
# ** connecting to the ros service to control the robot in torque **
clientLogger.info('Waiting for ROS Service /gazebo/apply_joint_effort')
wait_for_service('/gazebo/apply_joint_effort')
clientLogger.info('Found ROS Service /gazebo/apply_joint_effort')
service_proxy = ServiceProxy('/gazebo/apply_joint_effort', gazebo_msgs.srv.ApplyJointEffort, persistent=True)
wrench_dt = Duration.from_sec(0.08)#0.2

# -----------------------------------------------------------------------#
# **  map **
@nrp.MapVariable("debug", 			initial_value = 0)
@nrp.MapVariable("LF", 			    initial_value = 1)
@nrp.MapVariable("previous_tau",  	initial_value = prev_error)
@nrp.MapVariable("K_p", 			initial_value = Kp)
@nrp.MapVariable("K_d", 			initial_value = Kd)
@nrp.MapVariable("K_i", 			initial_value = Ki)
@nrp.MapVariable("proxy", 	 		initial_value = service_proxy)
@nrp.MapVariable("duration", 		initial_value = wrench_dt)
@nrp.MapVariable("fable_robot", 	initial_value = fable)
@nrp.MapVariable("n_joints", 		initial_value = n_joints)
@nrp.MapVariable("joints_name", 	initial_value = joints_name)
@nrp.MapVariable("DCNcommand_pos", 	     initial_value = DCNcmd_pos)
@nrp.MapVariable("DCNcommand_neg", 	     initial_value = DCNcmd_neg)
@nrp.MapVariable("DCNcommand_total", 	     initial_value = DCNcmd_total)
@nrp.MapVariable("LFcommand", 	    initial_value = LFcmd)

@nrp.MapVariable("tau_gain_dcn", 	initial_value = dcn_gain)
# -----------------------------------------------------------------------#
# **  DEEP CEEBELLAR NUCLEI OUTPUTS  **
@nrp.MapSpikeSink("DCN_0_0_pos",nrp.map_neurons(range(0, nrp.config.brain_root.DCN_n/2),lambda i: nrp.brain.DCN_0_0_pos[i]), nrp.population_rate, tau_fall=150.0, tau_rise=100.0)
@nrp.MapSpikeSink("DCN_0_0_neg",nrp.map_neurons(range(0, nrp.config.brain_root.DCN_n/2),lambda i: nrp.brain.DCN_0_0_neg[i]), nrp.population_rate, tau_fall=150.0, tau_rise=100.0)

@nrp.MapSpikeSink("DCN_0_1_pos",nrp.map_neurons(range(0, nrp.config.brain_root.DCN_n/2),lambda i: nrp.brain.DCN_0_1_pos[i]), nrp.population_rate, tau_fall=150.0, tau_rise=100.0)
@nrp.MapSpikeSink("DCN_0_1_neg",nrp.map_neurons(range(0, nrp.config.brain_root.DCN_n/2),lambda i: nrp.brain.DCN_0_1_neg[i]), nrp.population_rate, tau_fall=150.0, tau_rise=100.0)

# -----------------------------------------------------------------------#
# ** subscribe to ros topics **
# ** Current Joint states **
@nrp.MapRobotSubscriber("joints_current",      Topic('/joint_states', sensor_msgs.msg.JointState))
# ** Desired Joints trajectory **
@nrp.MapRobotSubscriber("desired_joints_trajectory", Topic('/robot/desired_trajectory/joints',  sensor_msgs.msg.JointState))
# -----------------------------------------------------------------------#
# ** publish on ros topics **
# joint position and velocity error publishers
@nrp.MapRobotSubscriber("error_q",   Topic('/robot/joint_error/position',     std_msgs.msg.Float64MultiArray))
@nrp.MapRobotSubscriber("error_qd",  Topic('/robot/joint_error/velocity',     std_msgs.msg.Float64MultiArray))
@nrp.MapRobotSubscriber("error_qdd", Topic('/robot/joint_error/acceleration', std_msgs.msg.Float64MultiArray))

@nrp.MapRobotPublisher("plan_pub", Topic('/robot/plan', std_msgs.msg.Bool))
@nrp.MapRobotPublisher("motor1", Topic('/robot/joint_1/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("motor2", Topic('/robot/joint_2/cmd_pos', std_msgs.msg.Float64))

# ** Recording the control input **
@nrp.MapCSVRecorder("command_recorder", filename="command.csv", headers=["time", "tau_inv1", "tau_inv2","tau_lf1","tau_lf2","dcn1","dcn2","control_input1","control_input2"])
# -----------------------------------------------------------------------#
# ** "Brain" comunication **
@nrp.Neuron2Robot()
def controller( t, debug, proxy, duration, fable_robot, 
				desired_joints_trajectory, n_joints, joints_name, joints_current,
				error_q, error_qd, error_qdd, DCNcommand_pos, DCNcommand_neg, DCNcommand_total, LF, LFcommand, previous_tau, 
				K_p, K_d, K_i, command_recorder, plan_pub, 
				DCN_0_0_pos, DCN_0_1_pos, DCN_0_0_neg, DCN_0_1_neg, tau_gain_dcn, motor1, motor2):

	# =================================== Inverse Dynamics ===================================
	# ex.: tau = robot.dynamics.rne(robot, q, qd, qdd)
	if LF.value == 1:
		tau_inv = robot.dynamics.rne(fable_robot.value, desired_joints_trajectory.value.position, [ 0., 0.], [ 1., 1.]) # Inertia matrix
	else:
		tau_inv = robot.dynamics.rne(fable_robot.value, desired_joints_trajectory.value.position, desired_joints_trajectory.value.velocity, desired_joints_trajectory.value.effort)


	# =================================== LF Controller ===================================
	
	if LF.value == 1:
		for j in range(0, n_joints.value):
			LFcommand.value.data[j] = ( K_p.value[j]*error_q.value.data[j] + K_d.value[j]*error_qd.value.data[j]  + previous_tau.value[j] )*tau_inv[0,j]  #  + K_i.value[j]*error_qdd.value.data[j]	
			#silvia correct version
			#LFcommand.value.data[j] = ( K_p.value[j]*error_q.value.data[j] + K_d.value[j]*error_qd.value.data[j])*tau_inv[0,j]  #  + K_i.value[j]*error_qdd.value.data[j]
			previous_tau.value[j] = LFcommand.value.data[j]

		if debug.value == 1:
			clientLogger.info("\n tau_inv "+str(tau_inv)+"\n LFcommand.value.data "+str(LFcommand.value.data)+"\n previous_tau: "+str(previous_tau.value ))
			#clientLogger.info("Joint "+str(j)+"\n KP contribution "+str(K_p.value[j]*error_q.value.data[j])+"\n KD contribution "+str(K_d.value[j]*error_qd.value.data[j] )+"\n Ki contribution: "+str( K_i.value[j]*error_qdd.value.data[j] ))
	else:
		tau_LF = [ 1., 1.]
		LFcommand.value.data = [ 1.*tau_inv[0,0], 1.*tau_inv[0,1] ] 

	#silvia correct version
	#LFcommand.value.data = [ LFcommand.value.data[0]*tau_inv[0,0], LFcommand.value.data[1]*tau_inv[0,1] ] 
		
	try:
		# =================================== Cerebellum Predict ===================================
		
		# =================================== Control Input ===================================
		control_input = []
		total_DCN = []
		dcn_cmd_pos = [DCN_0_0_pos, DCN_0_1_pos]
		dcn_cmd_neg = [DCN_0_0_neg, DCN_0_1_neg]

		for j in range(0, n_joints.value):
			DCNcommand_pos.value.data[j] = 0. # just to be 100% sure that it starts from 0
			DCNcommand_neg.value.data[j] = 0. # just to be 100% sure that it starts from 0
			DCNcommand_total.value.data[j] = 0.
			# DCN_command = positive contribution + negative contibution
			for dcn_pop_pos in dcn_cmd_pos[j]:
				DCNcommand_pos.value.data[j]  = DCNcommand_pos.value.data[j] + dcn_pop_pos.rate
				if debug.value == 1:
					clientLogger.info("positive DCN_joint " + str(j) + " : " + str( dcn_pop_pos.rate) )

			for dcn_pop_neg in dcn_cmd_neg[j]:
				DCNcommand_neg.value.data[j]  = DCNcommand_neg.value.data[j] + (-1.)*dcn_pop_neg.rate
				if debug.value == 1:
					clientLogger.info("negative DCN_joint " + str(j) + " : " + str( dcn_pop_neg.rate) )

			
			DCNcommand_total.value.data[j] = (DCNcommand_pos.value.data[j]+ DCNcommand_neg.value.data[j])/110.0
			
			control_input.append(LFcommand.value.data[j] + DCNcommand_total.value.data[j])
			#control_input.append(DCNcommand_total.value.data[j])
			#control_input.append(LFcommand.value.data[j])

		if debug.value == 1:
			clientLogger.info("\n control_input: "+str(control_input))

		# =================================== Send the Control Input ===================================     
		# ** Sends the control input (effort) to the plant **
		for j in range(0, n_joints.value):
			proxy.value.call(joints_name.value[j], control_input[j],  None, duration.value)
			if debug.value == 1:
				clientLogger.info("control_input sent!")
		# =================================== Comunication with the planner ===================================       
		plan_pub.send_message(True)

		 # =================================== Record the Data ===================================
		 # ** record ** --- t,"tau_inv","tau_lf","dcn","control_input"
		command_recorder.record_entry(t, tau_inv[0,0],tau_inv[0,1], LFcommand.value.data[0], LFcommand.value.data[1], DCNcommand_total.value.data[0], DCNcommand_total.value.data[0] ,
									   control_input[0], control_input[1] )
	except Exception as e:
		clientLogger.info(" --> Controller Exception: "+str(e))
