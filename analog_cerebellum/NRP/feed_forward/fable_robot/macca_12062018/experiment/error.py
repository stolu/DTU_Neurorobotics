
import numpy as np
from std_msgs.msg    import Float64, Float64MultiArray
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import LinkStates
import sys, time
global time
e_q   = std_msgs.msg.Float64MultiArray()
e_qd  = std_msgs.msg.Float64MultiArray()
e_qdd = std_msgs.msg.Float64MultiArray()
e_q.data   = [0.,0.]
e_qd.data  = [0.,0.]
e_qdd.data = [0.,0.]

curr_acc = std_msgs.msg.Float64MultiArray()
curr_acc.data = [0.,0.]
prev_vel = std_msgs.msg.Float64MultiArray()
prev_vel.data = [0.,0.]

joints_name = ['joint_1', 'joint_2']
n_links  = 2
n_joints = 2

# -----------------------------------------------------------------------#
# **  map **

@nrp.MapVariable("debug",         initial_value = 0)

@nrp.MapVariable("previous_time", initial_value = 0.)

@nrp.MapVariable("n_joints",      initial_value = n_joints)
@nrp.MapVariable("joints_name",   initial_value = joints_name)


@nrp.MapVariable("previous_velocity",     initial_value = prev_vel)

@nrp.MapVariable("current_acceleration", initial_value = curr_acc)

@nrp.MapVariable("error_q",       initial_value = e_q)
@nrp.MapVariable("error_qd",      initial_value = e_qd)
@nrp.MapVariable("error_qdd",     initial_value = e_qdd)


# -----------------------------------------------------------------------#
# ** subscribe to ros topics **
@nrp.MapRobotSubscriber("joints_current",      Topic('/joint_states', sensor_msgs.msg.JointState))
@nrp.MapRobotSubscriber("links_state_current", Topic('/gazebo/link_states', gazebo_msgs.msg.LinkStates))

# ** Desired end-effector trajectory **
@nrp.MapRobotSubscriber("desired_trajectory_ee", Topic('/robot/desired_trajectory/end_effector', std_msgs.msg.Float64MultiArray))

# ** Desired Joints trajectory **
@nrp.MapRobotSubscriber("desired_joints_trajectory", Topic('/robot/desired_trajectory/joints', sensor_msgs.msg.JointState))

# -----------------------------------------------------------------------#
# ** publish on ros topics **

# joint position and velocity error publishers
@nrp.MapRobotPublisher("error_q_pub",   Topic('/robot/joint_error/position',     std_msgs.msg.Float64MultiArray))
@nrp.MapRobotPublisher("error_qd_pub",  Topic('/robot/joint_error/velocity',     std_msgs.msg.Float64MultiArray))
@nrp.MapRobotPublisher("error_qdd_pub", Topic('/robot/joint_error/acceleration', std_msgs.msg.Float64MultiArray))


#@nrp.MapCSVRecorder("error_recorder", filename="error.csv", headers=["time", "e_q1","e_q2","e_qd1","e_qd2","e_qdd1","e_qdd2"])
@nrp.MapCSVRecorder("recorder", filename="joint_info.csv", headers=["time", "q1","q2","qd1","qd2","qdd1","qdd2",  "q1_des","q2_des","qd1_des","qd2_des","qdd1_des","qdd2_des" ,"e_q1","e_q2","e_qd1","e_qd2","e_qdd1","e_qdd2"])
#@nrp.MapCSVRecorder("desired_joint_recorder", filename="joint_desired.csv", headers=["time", "q1_des","q2_des","qd1_des","qd2_des","qdd1_des","qdd2_des"])

@nrp.Robot2Neuron()



def error(  t, debug, n_joints, joints_name, joints_current, links_state_current, 
            desired_trajectory_ee, desired_joints_trajectory, 
            error_q, error_qd, error_q_pub, error_qd_pub,
            error_qdd, error_qdd_pub,
            current_acceleration,
            recorder,#error_recorder, joint_recorder, desired_joint_recorder,
            previous_time, previous_velocity
            ):
    
    try:


        # ** end-effector name **
        topic_index = links_state_current.value.name.index('robot::top')

        # ** end-effector trajectory error **
        error_x = desired_trajectory_ee.value.data[0] - links_state_current.value.pose[topic_index].position.x
        error_y = desired_trajectory_ee.value.data[1] - links_state_current.value.pose[topic_index].position.y
        
        if debug.value == 1:
            clientLogger.info("error_x: "+str(error_x)+" error_y: "+str(error_y))
        
        # ** joints trajectory error **

        #current_acceleration = [0.,0.]
        delta_t = time.clock()-previous_time.value
        previous_time.value = time.clock()
           
        for k in range(0, n_joints.value):
            current_acceleration.value.data[k] = ((joints_current.value.velocity[k] - previous_velocity.value.data[k])/ delta_t)
            
            error_q.value.data[k]    = desired_joints_trajectory.value.position[k] - joints_current.value.position[k]
            error_qd.value.data[k]   = desired_joints_trajectory.value.velocity[k] - joints_current.value.velocity[k]
            error_qdd.value.data[k]  = desired_joints_trajectory.value.effort[k]   - current_acceleration.value.data[k]
            
            previous_velocity.value.data[k] = joints_current.value.velocity[k]
        
            
        if debug.value == 1:        
            clientLogger.info("\n  desired_joints_trajectory.value.effort: "+str( desired_joints_trajectory.value.effort)+"\n joints_current.value.acceletion: "+str(current_acceleration.value.data)+"\n error_qdd.value.data"+str( error_qdd.value.data))
            clientLogger.info("\n  desired_joints_trajectory.value.velocity: "+str( desired_joints_trajectory.value.velocity)+"\n joints_current.value.velocity: "+str(joints_current.value.velocity)+"\n error_qd.value.data"+str( error_qd.value.data))
            clientLogger.info("\n  desired_joints_trajectory.value.position: "+str( desired_joints_trajectory.value.position)+"\n joints_current.value.position: "+str(joints_current.value.position)+"\n error_qd.value.data"+str( error_q.value.data))

        # ** publish the error **
        error_q_pub.send_message(error_q.value)
        
        error_qd_pub.send_message(error_qd.value)

        error_qdd_pub.send_message(error_qdd.value)
        
        # ** record **
        recorder.record_entry(t, joints_current.value.position[0], joints_current.value.position[1],joints_current.value.velocity[0],joints_current.value.velocity[1],current_acceleration.value.data[0],current_acceleration.value.data[1],
                              desired_joints_trajectory.value.position[0], desired_joints_trajectory.value.position[1],desired_joints_trajectory.value.velocity[0],desired_joints_trajectory.value.velocity[1],desired_joints_trajectory.value.effort[0],desired_joints_trajectory.value.effort[1],
                              error_q.value.data[0],error_q.value.data[1],error_qd.value.data[0],error_qd.value.data[1],error_qdd.value.data[0],error_qdd.value.data[1])
        #error_recorder.record_entry(t, error_q.value.data[0],error_q.value.data[1],error_qd.value.data[0],error_qd.value.data[1],error_qdd.value.data[0],error_qdd.value.data[1])
        #joint_recorder.record_entry(t, joints_current.value.position[0], joints_current.value.position[1],joints_current.value.velocity[0],joints_current.value.velocity[1],current_acceleration.value.data[0],current_acceleration.value.data[1])
        #desired_joint_recorder.record_entry(t, desired_joints_trajectory.value.position[0], desired_joints_trajectory.value.position[1],desired_joints_trajectory.value.velocity[0],desired_joints_trajectory.value.velocity[1],desired_joints_trajectory.value.effort[0],desired_joints_trajectory.value.effort[1])
        
        if debug.value == 1:
            clientLogger.info("\n SENT ----> error_q: "+str(error_q.value.data)+"\n error_qd: "+str(error_qd.value.data)+"\n error_qdd: "+str(error_qdd.value.data))
        

    except Exception as e:
        clientLogger.info(" --> Error Exception: "+str(e))
    

