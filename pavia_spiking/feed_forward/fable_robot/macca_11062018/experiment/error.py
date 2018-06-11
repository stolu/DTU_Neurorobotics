# Imported Python Transfer Function
import numpy as np
from std_msgs.msg    import Float64, Float64MultiArray
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import LinkStates
import sys, time
global time
sys.path.append("/home/dtu-neurorobotics2/Documents/analog_cerebellum") # directory where the rbf is located
from rad_bas_f import RBF
e_q           = std_msgs.msg.Float64MultiArray()
e_q.data      = [0.,0.]
e_qd          = std_msgs.msg.Float64MultiArray()
e_qd.data     = [0.,0.]
e_qdd         = std_msgs.msg.Float64MultiArray()
e_qdd.data    = [0.,0.]
curr_acc      = std_msgs.msg.Float64MultiArray()
curr_acc.data = [0.,0.]
prev_vel      = std_msgs.msg.Float64MultiArray()
prev_vel.data = [0.,0.]
joints_name = ['joint_1', 'joint_2']
n_links  = 2
n_joints = 2


# ** Encoding Mossy input **
# encoding = RBF(total_n_neurons, min_in, max_in, min_out, max_out)
rbf_mossy_q_out_max = 0.1
rbf_mossy_q_out_min = 0.00001
rbf_mossy_q_min_input_range    = - np.pi*0.5
rbf_mossy_q_max_input_range    = + np.pi*0.5
rbf_mossy_qd_out_max = 0.1
rbf_mossy_qd_out_min = 0.0001
rbf_mossy_qd_min_input_range   = -10.
rbf_mossy_qd_max_input_range   =  10.

rbf_mossy_q  = RBF(nrp.config.brain_root.MF_number/4, rbf_mossy_q_min_input_range,  rbf_mossy_q_max_input_range,  rbf_mossy_q_out_min,  rbf_mossy_q_out_max) 
rbf_mossy_qd = RBF(nrp.config.brain_root.MF_number/4, rbf_mossy_qd_min_input_range, rbf_mossy_qd_max_input_range, rbf_mossy_qd_out_min, rbf_mossy_qd_out_max) 

# ** Encoding Inferior Olive input **

rbf_io_q_out_max = 0.1
rbf_io_q_out_min = 0.000001
rbf_io_q_min_input_range    = 0.#- np.pi
rbf_io_q_max_input_range    = + np.pi
rbf_io_qd_out_max = 0.1
rbf_io_qd_out_min = 0.0001
rbf_io_qd_min_input_range   = 0.
rbf_io_qd_max_input_range   = + 20.

rbf_io_q  = RBF(nrp.config.brain_root.IO_number/2,  rbf_io_q_min_input_range,  rbf_io_q_max_input_range,  rbf_io_q_out_min,  rbf_io_q_out_max)
rbf_io_qd = RBF(nrp.config.brain_root.IO_number/2,  rbf_io_qd_min_input_range,  rbf_io_qd_max_input_range,  rbf_io_qd_out_min,  rbf_io_qd_out_max) 

# -----------------------------------------------------------------------#
# **  map **
@nrp.MapVariable("debug",                 initial_value = 0)
@nrp.MapVariable("previous_time",         initial_value = 0.)
@nrp.MapVariable("n_joints",              initial_value = n_joints)
@nrp.MapVariable("joints_name",           initial_value = joints_name)
@nrp.MapVariable("previous_velocity",     initial_value = prev_vel)
@nrp.MapVariable("current_acceleration",  initial_value = curr_acc)

@nrp.MapVariable("error_q",               initial_value = e_q)
@nrp.MapVariable("error_qd",              initial_value = e_qd)
@nrp.MapVariable("error_qdd",             initial_value = e_qdd)

# data encoding
@nrp.MapVariable("radial_func_q",  initial_value = rbf_mossy_q  )
@nrp.MapVariable("radial_func_qd", initial_value = rbf_mossy_qd )
@nrp.MapVariable("radial_func_e",  initial_value = rbf_io_q     )
@nrp.MapVariable("radial_func_ed", initial_value = rbf_io_qd    )

# -----------------------------------------------------------------------#
# ** subscribe to ros topics **
@nrp.MapRobotSubscriber("joints_current",      Topic('/joint_states',       sensor_msgs.msg.JointState))
@nrp.MapRobotSubscriber("links_state_current", Topic('/gazebo/link_states', gazebo_msgs.msg.LinkStates))

# ** Desired end-effector trajectory **
@nrp.MapRobotSubscriber("desired_trajectory_ee",     Topic('/robot/desired_trajectory/end_effector', std_msgs.msg.Float64MultiArray))
# ** Desired Joints trajectory **
@nrp.MapRobotSubscriber("desired_joints_trajectory", Topic('/robot/desired_trajectory/joints',       sensor_msgs.msg.JointState))


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

@nrp.MapSpikeSource("IO_joint1_q_pos",  nrp.map_neurons(range(                                0, nrp.config.brain_root.IO_number/2), 	lambda i: nrp.brain.INFOLIVE_j1_pos[i]),  nrp.ac_source)
@nrp.MapSpikeSource("IO_joint1_q_neg",  nrp.map_neurons(range(                                0, nrp.config.brain_root.IO_number/2), 	lambda i: nrp.brain.INFOLIVE_j1_neg[i]),  nrp.ac_source)

@nrp.MapSpikeSource("IO_joint1_qd_pos", nrp.map_neurons(range(nrp.config.brain_root.IO_number/2, nrp.config.brain_root.IO_number  ), lambda i: nrp.brain.INFOLIVE_j1_pos[i]),  nrp.ac_source)
@nrp.MapSpikeSource("IO_joint1_qd_neg", nrp.map_neurons(range(nrp.config.brain_root.IO_number/2, nrp.config.brain_root.IO_number  ), lambda i: nrp.brain.INFOLIVE_j1_neg[i]),  nrp.ac_source)

@nrp.MapSpikeSource("IO_joint2_q_pos",  nrp.map_neurons(range(                                0, nrp.config.brain_root.IO_number/2), 	lambda i: nrp.brain.INFOLIVE_j2_pos[i]),  nrp.ac_source)
@nrp.MapSpikeSource("IO_joint2_q_neg",  nrp.map_neurons(range(                                0, nrp.config.brain_root.IO_number/2), 	lambda i: nrp.brain.INFOLIVE_j2_neg[i]),  nrp.ac_source)

@nrp.MapSpikeSource("IO_joint2_qd_pos", nrp.map_neurons(range(nrp.config.brain_root.IO_number/2, nrp.config.brain_root.IO_number  ),  lambda i: nrp.brain.INFOLIVE_j2_pos[i]), nrp.ac_source)
@nrp.MapSpikeSource("IO_joint2_qd_neg", nrp.map_neurons(range(nrp.config.brain_root.IO_number/2, nrp.config.brain_root.IO_number  ),  lambda i: nrp.brain.INFOLIVE_j2_neg[i]), nrp.ac_source)

# **  MOSSY FIBERS INPUTS  **
# send the current joint values to the mossy
@nrp.MapSpikeSource("MF_joint1_q_curr",  nrp.map_neurons(range(nrp.config.brain_root.MF_number/2,   nrp.config.brain_root.MF_number*3/4),   lambda i: nrp.brain.MOSSY_j1[i]),  nrp.ac_source)
@nrp.MapSpikeSource("MF_joint1_qd_curr", nrp.map_neurons(range(nrp.config.brain_root.MF_number*3/4, nrp.config.brain_root.MF_number    ),   lambda i: nrp.brain.MOSSY_j1[i]),  nrp.ac_source)
@nrp.MapSpikeSource("MF_joint2_q_curr",  nrp.map_neurons(range(nrp.config.brain_root.MF_number/2,   nrp.config.brain_root.MF_number*3/4),   lambda i: nrp.brain.MOSSY_j2[i]),  nrp.ac_source)
@nrp.MapSpikeSource("MF_joint2_qd_curr", nrp.map_neurons(range(nrp.config.brain_root.MF_number*3/4, nrp.config.brain_root.MF_number    ),   lambda i: nrp.brain.MOSSY_j2[i]),  nrp.ac_source)


# -----------------------------------------------------------------------#
# ** Recording **
@nrp.MapCSVRecorder("recorder", filename="joint_info.csv", headers=["time", "q1","q2","qd1","qd2","qdd1","qdd2",  "q1_des","q2_des","qd1_des","qd2_des","qdd1_des","qdd2_des" ,"e_q1","e_q2","e_qd1","e_qd2","e_qdd1","e_qdd2"])


@nrp.Robot2Neuron()
def error(  t, debug, n_joints, joints_name, joints_current, links_state_current, 
            desired_trajectory_ee, desired_joints_trajectory, 
            error_q, error_qd, error_q_pub, error_qd_pub,
            error_qdd, error_qdd_pub,
            current_acceleration,
            recorder,
            IO_joint1_q_pos, IO_joint1_qd_pos, IO_joint2_q_pos, IO_joint2_qd_pos,
            IO_joint1_q_neg, IO_joint1_qd_neg, IO_joint2_q_neg, IO_joint2_qd_neg,
            MF_joint1_q_curr, MF_joint1_qd_curr, MF_joint2_q_curr, MF_joint2_qd_curr,
            radial_func_q, radial_func_qd, radial_func_e, radial_func_ed,
            previous_time, previous_velocity
            ):
    try:
        
        IO_input_q_pos   = [ IO_joint1_q_pos,   IO_joint2_q_pos  ]  
        IO_input_q_neg   = [ IO_joint1_q_neg,   IO_joint2_q_neg  ]
        IO_input_qd_pos  = [ IO_joint1_qd_pos,  IO_joint2_qd_pos ]
        IO_input_qd_neg  = [ IO_joint1_qd_neg,  IO_joint2_qd_neg ]
        
        MF_joint_q_curr  = [ MF_joint1_q_curr,  MF_joint2_q_curr ]
        MF_joint_qd_curr = [ MF_joint1_qd_curr, MF_joint2_qd_curr]
        
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
        delta_t = time.clock()-previous_time.value
        previous_time.value = time.clock()
        for k in range(0, n_joints.value):
			current_acceleration.value.data[k] = ((joints_current.value.velocity[k] - previous_velocity.value.data[k])/ delta_t)
			error_q.value.data[k]    = desired_joints_trajectory.value.position[k]  - joints_current.value.position[k]
			error_qd.value.data[k]   = desired_joints_trajectory.value.velocity[k]  - joints_current.value.velocity[k]
			error_qdd.value.data[k]  = desired_joints_trajectory.value.effort[k]    - current_acceleration.value.data[k]
			previous_velocity.value.data[k] = joints_current.value.velocity[k]

			# -----------------------------------------------------------------------#
			# ** Sending the input to the inferior olive **
			e_io  = radial_func_e.value.function(  abs(error_q.value.data[k] ) ) # calculate the rbs of the absolute value, the sign will be taken in account later
			ed_io = radial_func_ed.value.function( abs(error_qd.value.data[k]) )
			io  = 0
			iod = 0
			
			if error_q.value.data[k] >= 0.:
				for neuron_IOq_p in IO_input_q_pos[k]:
        				neuron_IOq_p.amplitude  = e_io[io]*(abs(error_q.value.data[k]) / error_q.value.data[k]) # this is to keep the sign		
        				io = io + 1

			elif error_q.value.data[k] < 0.:
				for neuron_IOq_n in IO_input_q_neg[k]: 
        				neuron_IOq_n.amplitude  = e_io[io]*(abs(error_q.value.data[k]) / error_q.value.data[k]) # this is to keep the sign		
        				io = io + 1

			elif error_qd.vdalue.data[k] >= 0.:
				for neuron_IOqd_p in IO_input_qd_pos[k]:
        				neuron_IOqd_p.amplitude = ed_io[iod]*(abs(error_qd.value.data[k]) / error_qd.value.data[k])
        				iod = iod + 1

			elif error_qd.vdalue.data[k] < 0.:
				for neuron_IOqd_n in IO_input_qd_neg[k]:
        				neuron_IOqd_n.amplitude = ed_io[iod]*(abs(error_qd.value.data[k]) / error_qd.value.data[k])
        				iod = iod + 1
			
			if debug.value == 1:
				clientLogger.info("postion error joint "+str(k)+" :"+str(error_q.value.data[k])+" \n  e_io[io] max: "+str(max( e_io)) +" \n  e_io[io] min: "+str(min( e_io)))
				clientLogger.info("velocity error joint "+str(k)+" :"+str(error_qd.value.data[k])+" \n  ed_io[io] max: "+str(max( ed_io)) +" \n  ed_io[io] min: "+str(min( ed_io)))
				clientLogger.info("\n IO_input_q_pos[k] "+str( IO_input_q_pos[k]))
				clientLogger.info("\n IO_input_qd_pos[k] "+str( IO_input_qd_pos[k]))
				clientLogger.info("\n IO_input_q_neg[k] "+str( IO_input_q_neg[k]))
				clientLogger.info("\n IO_input_qd_neg[k] "+str( IO_input_qd_neg[k]))

			# -----------------------------------------------------------------------#
			# ** Sending the current joint states to the mossy fibers **
			curr_q_mossy  = radial_func_q.value.function( joints_current.value.position[k] ) 
			curr_qd_mossy = radial_func_qd.value.function( joints_current.value.velocity[k] )
			m = 0
			n = 0
			for neuron_MFqcurr  in MF_joint_q_curr[k]:
				neuron_MFqcurr.amplitude  = curr_q_mossy[m] 
				m = m + 1
			for neuron_MFqdcurr in MF_joint_qd_curr[k]:
				neuron_MFqdcurr.amplitude = curr_qd_mossy[n] 
				n = n + 1

        if debug.value == 1:        
            clientLogger.info("\n  desired_joints_trajectory.value.effort: "+str( desired_joints_trajectory.value.effort)+"\n joints_current.value.acceletion: "+str(current_acceleration.value.data)+"\n error_qdd.value.data"+str( error_qdd.value.data))
            clientLogger.info("\n  desired_joints_trajectory.value.velocity: "+str( desired_joints_trajectory.value.velocity)+"\n joints_current.value.velocity: "+str(joints_current.value.velocity)+"\n error_qd.value.data"+str( error_qd.value.data))
            clientLogger.info("\n  desired_joints_trajectory.value.position: "+str( desired_joints_trajectory.value.position)+"\n joints_current.value.position: "+str(joints_current.value.position)+"\n error_qd.value.data"+str( error_q.value.data))


        # -----------------------------------------------------------------------#
        # ** publish the error **
        error_q_pub.send_message(error_q.value)
        error_qd_pub.send_message(error_qd.value)
        error_qdd_pub.send_message(error_qdd.value)

        # -----------------------------------------------------------------------#
        # ** record **
        recorder.record_entry(t, joints_current.value.position[0], joints_current.value.position[1],joints_current.value.velocity[0],joints_current.value.velocity[1],current_acceleration.value.data[0],current_acceleration.value.data[1],
                              desired_joints_trajectory.value.position[0], desired_joints_trajectory.value.position[1],desired_joints_trajectory.value.velocity[0],desired_joints_trajectory.value.velocity[1],desired_joints_trajectory.value.effort[0],desired_joints_trajectory.value.effort[1],
                              error_q.value.data[0],error_q.value.data[1],error_qd.value.data[0],error_qd.value.data[1],error_qdd.value.data[0],error_qdd.value.data[1])


    except Exception as e:
        clientLogger.info(" --> Error Exception: "+str(e))
