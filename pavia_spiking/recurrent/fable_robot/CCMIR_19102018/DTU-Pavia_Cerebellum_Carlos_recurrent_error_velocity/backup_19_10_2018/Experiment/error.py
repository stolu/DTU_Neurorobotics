# Imported Python Transfer Function
import numpy as np
from std_msgs.msg    import Float64, Float64MultiArray
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import LinkStates
import sys, time
global time
sys.path.append("/home/dtu-neurorobotics/Documents/NRP/Experiments/fable_manipulation_dtu_spiking_cereb_recurrent_error_velocity")
from rad_bas_f import RBF

e_q           = std_msgs.msg.Float64MultiArray()
e_q.data      = [0.,0.]
e_qd          = std_msgs.msg.Float64MultiArray()
e_qd.data     = [0.,0.]
e_qdd         = std_msgs.msg.Float64MultiArray()
e_qdd.data    = [0.,0.]


prev_error = std_msgs.msg.Float64MultiArray()
prev_error.data = [ 0., 0.]

prev_errorss = [[0.],[0.]]
int_time = [[0.],[0.]]


prev_pos = std_msgs.msg.Float64MultiArray()
prev_pos.data   = [0.,0.]

prev_vel = std_msgs.msg.Float64MultiArray()
prev_vel.data   = [0.,0.]

vel = std_msgs.msg.Float64MultiArray()
vel.data   = [0.,0.]

acc = std_msgs.msg.Float64MultiArray()
acc.data   = [0.,0.]

#Initiating FIFO queue for calculating velocity by moving average
velocity_queue_list_init = [[],[]]
velocity_average_init = std_msgs.msg.Float64MultiArray()#[0.0, 0.0]
velocity_average_init.data = [0.0, 0.0]
velocity_moving_avg_window_size_init = 20

curr_acc      = std_msgs.msg.Float64MultiArray()
curr_acc.data = [0.,0.]
prev_vel      = std_msgs.msg.Float64MultiArray()
prev_vel.data = [0.,0.]
joints_name   = ['joint_1', 'joint_2']
n_links       = 2
n_joints      = 2


# ** Encoding Mossy input **
# encoding = RBF(total_n_neurons, min_in, max_in, min_out, max_out)
rbf_mossy_q_out_max          = 0.253
rbf_mossy_q_out_min          = 0.00001
rbf_mossy_q_min_input_range  = - np.pi*0.5
rbf_mossy_q_max_input_range  = + np.pi*0.5

#rbf_mossy_torque_out_max         = 0.25
#rbf_mossy_torque_out_min         = 0.0001
#rbf_mossy_torque_min_input_range = -1.56
#rbf_mossy_torque_max_input_range =  1.56
# initialization of the RBF object for MF (it does not matter which joint, this will be specified when calling the function)
rbf_mossy_q  = RBF(nrp.config.brain_root.MF_n/nrp.config.brain_root.n_inputs_MF, rbf_mossy_q_min_input_range,  rbf_mossy_q_max_input_range,  rbf_mossy_q_out_min,  rbf_mossy_q_out_max) 
#rbf_mossy_torque = RBF(nrp.config.brain_root.MF_n/nrp.config.brain_root.n_inputs_MF, rbf_mossy_torque_min_input_range, rbf_mossy_torque_max_input_range, rbf_mossy_torque_out_min, rbf_mossy_torque_out_max) 


# -----------------------------------------------------------------------#
# **  map **
@nrp.MapVariable("debug",                 initial_value = 0)
@nrp.MapVariable("n_joints",              initial_value = n_joints)
@nrp.MapVariable("joints_name",           initial_value = joints_name)

@nrp.MapVariable("error_q",               initial_value = e_q)
@nrp.MapVariable("error_qd",              initial_value = e_qd)
@nrp.MapVariable("error_qdd",             initial_value = e_qdd)

@nrp.MapVariable("previous_error", 	 initial_value = prev_error)
@nrp.MapVariable("previous_errors", 	 initial_value = prev_errorss)
@nrp.MapVariable("integral_time", 	 initial_value = int_time)

@nrp.MapVariable("previous_pos", 	 initial_value = prev_pos)
@nrp.MapVariable("previous_vel", 	 initial_value = prev_vel)
@nrp.MapVariable("veloc", 			 initial_value = vel)
@nrp.MapVariable("acceleration", 	 initial_value = acc)

@nrp.MapVariable("velocity_average_", initial_value = velocity_average_init)
@nrp.MapVariable("velocity_queue_list_", initial_value = velocity_queue_list_init)
@nrp.MapVariable("velocity_moving_avg_window_size", initial_value = velocity_moving_avg_window_size_init)

# data encoding
@nrp.MapVariable("radial_func_q",  initial_value = rbf_mossy_q) #same mapping for PLUS and MINUS current positions
#@nrp.MapVariable("radial_func_torque",  initial_value = rbf_mossy_torque) #same mapping for PLUS and MINUS current positions

# -----------------------------------------------------------------------#
# ** subscribe to ros topics **
@nrp.MapRobotSubscriber("joints_current",      Topic('/joint_states',       sensor_msgs.msg.JointState))
@nrp.MapRobotSubscriber("links_state_current", Topic('/gazebo/link_states', gazebo_msgs.msg.LinkStates))

# ** Desired end-effector trajectory **
@nrp.MapRobotSubscriber("desired_trajectory_ee",     Topic('/robot/desired_trajectory/end_effector', std_msgs.msg.Float64MultiArray))
# ** Desired Joints trajectory **
@nrp.MapRobotSubscriber("desired_joints_trajectory", Topic('/robot/desired_trajectory/joints',       sensor_msgs.msg.JointState))

#@nrp.MapRobotSubscriber("desired_LF_torque", Topic('/robot/desired_LF_torque/joints',       std_msgs.msg.Float64MultiArray))

# -----------------------------------------------------------------------#
# ** publish on ros topics **
# joint position and velocity error publishers
@nrp.MapRobotPublisher("error_q_pub",   Topic('/robot/joint_error/position',     std_msgs.msg.Float64MultiArray))
@nrp.MapRobotPublisher("error_qd_pub",  Topic('/robot/joint_error/velocity',     std_msgs.msg.Float64MultiArray))
@nrp.MapRobotPublisher("error_qdd_pub", Topic('/robot/joint_error/acceleration', std_msgs.msg.Float64MultiArray))

# -----------------------------------------------------------------------#
# **  INFERIOR OLIVE INPUTS  **
# each IO is splitted in 2 part: one dedicated to the error in position, one dedicated to the error in velocity
# each part is splitted in 2, negative and positive.
#           -INFOLIVE_j1_pos = [IO_joint1_q_pos , IO_joint1_qd_pos]
#           -INFOLIVE_j1_neg = [IO_joint1_q_neg , IO_joint1_qd_neg]
#           -INFOLIVE_j2_pos = [IO_joint2_q_pos , IO_joint2_qd_pos]
#           -INFOLIVE_j2_neg = [IO_joint2_q_neg , IO_joint2_qd_neg]

@nrp.MapSpikeSource("IO_0_0_q_pos",  nrp.map_neurons(range(0, nrp.config.brain_root.IO_n/4), lambda i: nrp.brain.IO_0_0_pos_q[i]), nrp.poisson, weight=0.1)#nrp.ac_source)
@nrp.MapSpikeSource("IO_0_0_q_neg",  nrp.map_neurons(range(0, nrp.config.brain_root.IO_n/4), lambda i: nrp.brain.IO_0_0_neg_q[i]), nrp.poisson, weight=0.1)#nrp.ac_source)
@nrp.MapSpikeSource("IO_0_0_qd_pos",  nrp.map_neurons(range(0, nrp.config.brain_root.IO_n/4), lambda i: nrp.brain.IO_0_0_pos_qd[i]), nrp.poisson, weight=0.1)#nrp.ac_source)
@nrp.MapSpikeSource("IO_0_0_qd_neg",  nrp.map_neurons(range(0, nrp.config.brain_root.IO_n/4), lambda i: nrp.brain.IO_0_0_neg_qd[i]), nrp.poisson, weight=0.1)#nrp.ac_source)



@nrp.MapSpikeSource("IO_0_1_q_pos",  nrp.map_neurons(range(0, nrp.config.brain_root.IO_n/4), lambda i: nrp.brain.IO_0_1_pos_q[i]),  nrp.poisson, weight=0.1)#nrp.ac_source)
@nrp.MapSpikeSource("IO_0_1_q_neg",  nrp.map_neurons(range(0, nrp.config.brain_root.IO_n/4), lambda i: nrp.brain.IO_0_1_neg_q[i]),  nrp.poisson, weight=0.1)#nrp.ac_source)
@nrp.MapSpikeSource("IO_0_1_qd_pos",  nrp.map_neurons(range(0, nrp.config.brain_root.IO_n/4), lambda i: nrp.brain.IO_0_1_pos_qd[i]),  nrp.poisson, weight=0.1)#nrp.ac_source)
@nrp.MapSpikeSource("IO_0_1_qd_neg",  nrp.map_neurons(range(0, nrp.config.brain_root.IO_n/4), lambda i: nrp.brain.IO_0_1_neg_qd[i]),  nrp.poisson, weight=0.1)#nrp.ac_source)



# **  MOSSY FIBERS INPUTS  **
# send the current joint values to the mossy
@nrp.MapSpikeSource("MF_0_0_cur_q",  nrp.map_neurons(range(0, nrp.config.brain_root.MF_n/nrp.config.brain_root.n_inputs_MF), lambda i: nrp.brain.MF_0_0_cur_q[i]),  nrp.ac_source)
@nrp.MapSpikeSource("MF_0_1_cur_q",  nrp.map_neurons(range(0, nrp.config.brain_root.MF_n/nrp.config.brain_root.n_inputs_MF), lambda i: nrp.brain.MF_0_1_cur_q[i]),  nrp.ac_source)


#@nrp.MapSpikeSource("MF_0_0_des_torque",  nrp.map_neurons(range(0, nrp.config.brain_root.MF_n/nrp.config.brain_root.n_inputs_MF), lambda i: nrp.brain.MF_0_0_des_torque[i]),  nrp.ac_source)
#@nrp.MapSpikeSource("MF_0_1_des_torque",  nrp.map_neurons(range(0, nrp.config.brain_root.MF_n/nrp.config.brain_root.n_inputs_MF), lambda i: nrp.brain.MF_0_1_des_torque[i]),  nrp.ac_source)



# -----------------------------------------------------------------------#
# ** Recording **
@nrp.MapCSVRecorder("recorder", filename="joint_info.csv", headers=["time", "q1","q2","qd1","qd2","qdd1","qdd2",  "q1_des","q2_des","qd1_des","qd2_des","qdd1_des","qdd2_des" ,"e_q1","e_q2","e_qd1","e_qd2","e_qdd1","e_qdd2"])


@nrp.Robot2Neuron()
def error(  t, debug, n_joints, joints_name, joints_current, links_state_current, 
			desired_trajectory_ee, desired_joints_trajectory,
			error_q, error_qd, error_q_pub, error_qd_pub,
			error_qdd, error_qdd_pub,
			recorder,
			IO_0_0_q_pos, IO_0_1_q_pos,
			IO_0_0_q_neg,  IO_0_1_q_neg,
			IO_0_0_qd_pos, IO_0_1_qd_pos,
			IO_0_0_qd_neg,  IO_0_1_qd_neg,
			MF_0_0_cur_q, MF_0_1_cur_q,
			radial_func_q, previous_error, previous_errors, integral_time, previous_pos, 
			previous_vel, veloc, acceleration, velocity_moving_avg_window_size, velocity_average_, velocity_queue_list_):


	try:
		
		IO_input_q_pos   = [IO_0_0_q_pos, IO_0_1_q_pos]  
		IO_input_q_neg   = [IO_0_0_q_neg, IO_0_1_q_neg]
		IO_input_qd_pos  = [IO_0_0_qd_pos, IO_0_1_qd_pos]  
		IO_input_qd_neg  = [IO_0_0_qd_neg, IO_0_1_qd_neg]
		
		MF_joint_q_cur  = [MF_0_0_cur_q,  MF_0_1_cur_q]
		
		#MF_joint_des_torque = [MF_0_0_des_torque,  MF_0_1_des_torque]
		# ****************************************** end-effector --> not used at the moment
		# ** end-effector name **
		topic_index = links_state_current.value.name.index('robot::top')
		
		# ** end-effector trajectory error ** 
		error_x = desired_trajectory_ee.value.data[0] - links_state_current.value.pose[topic_index].position.x
		error_y = desired_trajectory_ee.value.data[1] - links_state_current.value.pose[topic_index].position.y
		if debug.value == 1:
			clientLogger.info("error_x: "+str(error_x)+" error_y: "+str(error_y))

		# -----------------------------------------------------------------------#

		# ** joints trajectory error **
		delta_t = 0.02 # 0.08

		for k in range(0, n_joints.value):
			
			error_q.value.data[k]      = desired_joints_trajectory.value.position[k]  - joints_current.value.position[k]
			veloc.value.data[k]        = (joints_current.value.position[k] - previous_pos.value.data[k])/delta_t #0.0085
			
			previous_pos.value.data[k] = joints_current.value.position[k]
			previous_vel.value.data[k] = veloc.value.data[k]

			#velocity_frac = joints_current.value.velocity[k] / velocity_moving_avg_window_size.value
			#velocity_queue_list_.value[k].append(velocity_frac)
			#velocity_average_.value.data[k] = velocity_average_.value.data[k] + velocity_frac - velocity_queue_list_.value[k][0]
			#velocity_queue_list_.value[k].pop(0)

			#error_qd.value.data[k]   = desired_joints_trajectory.value.velocity[k] - velocity_average_.value.data[k]

                        teaching_signal_error_qd   = desired_joints_trajectory.value.velocity[k] - veloc.value.data[k]
			
			error_qd.value.data[k]  = (error_q.value.data[k] - previous_error.value.data[k])/delta_t

			previous_error.value.data[k] = error_q.value.data[k]
			
			if error_qd.value.data[k] > 10.:
				error_qd.value.data[k] = 10.
			elif error_qd.value.data[k] < -10.:
				error_qd.value.data[k] = -10.

			N = 120./8#40.
			
			if len(previous_errors.value[k]) < int(N):
				previous_errors.value[k].append(error_q.value.data[k])
				integral_time.value[k].append(t)
				dummy_N = len(previous_errors.value[k])
				error_qdd.value.data[k]  =( (integral_time.value[k][ dummy_N-1] - integral_time.value[k][0])/dummy_N)* (previous_errors.value[k][0]*0.5 + previous_errors.value[k][dummy_N-1]*0.5 + np.sum(previous_errors.value[k][1:-1]))

				
				if debug.value == 1:
					clientLogger.info(str(len(previous_errors.value[k]))+"  Joint "+str(j)+" integral error iteration : "+str(t/delta_t)+" "+str(error_qd.value.data[k])+" \n "+str(prev_errors.value[k]) )
			else:
				previous_errors.value[k].append(error_q.value.data[k])
				previous_errors.value[k].pop(0)
				integral_time.value[k].append(t)
				integral_time.value[k].pop(0)    				
				error_qdd.value.data[k]  =( (integral_time.value[k][int(N)-1] - integral_time.value[k][0])/N)* (previous_errors.value[k][0]*0.5 + previous_errors.value[k][int(N)-1]*0.5 + np.sum(previous_errors.value[k][1:-1]))

				#error_qdd.value.data[k]  = ( error_q.value.data[k] - previous_errors.value[k][0] ) #/(N)
				if debug.value == 1:
					clientLogger.info("Joint "+str(j)+"iteration " +str(t/delta_t)+" integral error : "+str(error_qd.value.data[k])+" \n "+str(previous_errors.value[k][:]) )

			# -----------------------------------------------------------------------#
			# ** Sending the input to the inferior olive **

			#e_io  = radial_func_e.value.function(abs(error_q.value.data[k])) # calculate the rbs of the absolute value, the sign will be taken in account later
			#ed_io = radial_func_ed.value.function(abs(error_qd.value.data[k]))

			#clientLogger.info("e_io: " + str(e_io))
			#clientLogger.info("ed_io: " + str(ed_io))


			#e_io = ((10.0 - 0.0)/(np.pi - 0.0))*(abs(error_q.value.data[k]) - np.pi) + 10.0
			max_io_rate = 12.0 #biologically 10hz but we round it up
			min_io_rate = 0.0
			max_error_q = abs(np.pi)
			min_error_q = 0.0
			max_error_qd = 10.0
			min_error_qd = 0.0

			e_io = ((max_io_rate - min_io_rate)/(max_error_q - min_error_q))*(abs(error_q.value.data[k]) - max_error_q) + max_io_rate
			ed_io = ((max_io_rate - min_io_rate)/(max_error_qd - min_error_qd))*(abs(teaching_signal_error_qd) - max_error_qd) + max_io_rate

			

			#io  = 0
			#iod = 0

			if error_q.value.data[k] >= 0.:
				for neuron_IOq_p in IO_input_q_pos[k]:
						neuron_IOq_p.rate  = e_io#[io] # this is to keep the sign		
			#			io = io + 1

			if error_q.value.data[k] < 0.:
				for neuron_IOq_n in IO_input_q_neg[k]: 
						neuron_IOq_n.rate  = 1.0*e_io#[io] # this is to keep the sign		
			#			io = io + 1
                        
                        if teaching_signal_error_qd >= 0.:
				for neuron_IOqd_p in IO_input_qd_pos[k]:
						neuron_IOqd_p.rate  = ed_io#[io] # this is to keep the sign		
			#			io = io + 1

			if teaching_signal_error_qd < 0.:
				for neuron_IOqd_n in IO_input_qd_neg[k]: 
						neuron_IOqd_n.rate  = 1.0*ed_io#[io] # this is to keep the sign		
			#			io = io + 1
			
			# -----------------------------------------------------------------------#
			# ** Sending the current joint states to the mossy fibers **
			cur_q_mossy  = radial_func_q.value.function(joints_current.value.position[k])
			#if t==0:
			#	des_torque_mossy  = radial_func_torque.value.function(0.0)
			#else:
			#	des_torque_mossy  = radial_func_torque.value.function(desired_LF_torque.value.data[k])
			


			#if the torque has been sent, the current position is the PLUS position
			m = 0
			p = 0
			for neuron_MFqcur  in MF_joint_q_cur[k]:
				neuron_MFqcur.amplitude  = cur_q_mossy[m] 
				m = m + 1
		
			#for neuron_MFqdes_torque  in MF_joint_des_torque[k]:
			#	neuron_MFqdes_torque.amplitude  = des_torque_mossy[p] 
			#	p = p + 1

			if debug.value == 1:
				clientLogger.info("postion error joint "+str(k)+" :"+str(error_q.value.data[k])+" \n  e_io[io] max: "+str(max( e_io)) +" \n  e_io[io] min: "+str(min( e_io)))
				clientLogger.info("velocity error joint "+str(k)+" :"+str(error_qd.value.data[k])+" \n  ed_io[io] max: "+str(max( ed_io)) +" \n  ed_io[io] min: "+str(min( ed_io)))
				clientLogger.info("\n IO_input_q_pos[k] "+str( IO_input_q_pos[k]))
				clientLogger.info("\n IO_input_qd_pos[k] "+str( IO_input_qd_pos[k]))
				clientLogger.info("\n IO_input_q_neg[k] "+str( IO_input_q_neg[k]))
				clientLogger.info("\n IO_input_qd_neg[k] "+str( IO_input_qd_neg[k]))



		if debug.value == 1:        
			#clientLogger.info("\n  desired_joints_trajectory.value.effort: "+str( desired_joints_trajectory.value.effort)+"\n joints_current.value.acceletion: "+str(current_acceleration.value.data)+"\n error_qdd.value.data"+str( error_qdd.value.data))
			clientLogger.info("\n  desired_joints_trajectory.value.velocity: "+str( desired_joints_trajectory.value.velocity)+"\n joints_current.value.velocity: "+str(joints_current.value.velocity)+"\n error_qd.value.data"+str( error_qd.value.data))
			clientLogger.info("\n  desired_joints_trajectory.value.position: "+str( desired_joints_trajectory.value.position)+"\n joints_current.value.position: "+str(joints_current.value.position)+"\n error_qd.value.data"+str( error_q.value.data))


		# -----------------------------------------------------------------------#
		# ** publish the error **
		error_q_pub.send_message(error_q.value)
		#error_qd_pub.send_message(veloc.value)
		error_qd_pub.send_message(error_qd.value)
		error_qdd_pub.send_message(error_qdd.value)

		# -----------------------------------------------------------------------#
		# ** record **
		recorder.record_entry(t, 
								joints_current.value.position[0], joints_current.value.position[1],
								joints_current.value.velocity[0],joints_current.value.velocity[1],
								0.0, 0.0,
								desired_joints_trajectory.value.position[0], desired_joints_trajectory.value.position[1],
								desired_joints_trajectory.value.velocity[0],desired_joints_trajectory.value.velocity[1],
								desired_joints_trajectory.value.effort[0],desired_joints_trajectory.value.effort[1],
								error_q.value.data[0],error_q.value.data[1],
								error_qd.value.data[0],error_qd.value.data[1],
								error_qdd.value.data[0],error_qdd.value.data[1])


	except Exception as e:
		clientLogger.info(" --> Error Exception: "+str(e))
