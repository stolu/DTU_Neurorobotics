# Imported Python Transfer Function
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 11:31:10 2018
@author: Marie Claire Capolei macca@elektro.dtu.dk
"""
import sys, math
from std_msgs.msg import Float64,Float64MultiArray, Bool
from sensor_msgs.msg import JointState
import numpy as np
from rospy import ServiceProxy, wait_for_service, Duration
sys.path.append("/usr/local/lib")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("~/Documents/LWPR")
sys.path.append("~/Documents/NRP/Experiments/icub_ball_balancing")
sys.path.append("~/Documents/analog_cerebellum")


from pid_controller import*
n_iter  = 500000
nin     = 8                     # ball_pos table_pos 2 * pos and vel joints
njoints = 1                    # Wrist roll and pitch

# Robot information

joints_index = ["r_wrist_prosup","r_wrist_yaw", "r_wrist_pitch","r_elbow" ]
curr_joints = sensor_msgs.msg.JointState()
curr_joints.position = [0. for i in range( 0, njoints)]
curr_joints.velocity = [0. for i in range( 0, njoints)]
curr_joints.effort = [0. for i in range( 0, njoints)]


# Control parameters
Kp_init = [0.63, 5.41, 10.41]#[0.09, 5.41, 10.41]
Kd_init = [0.05 , 0.01, 0.01]#[0.3 , 0.01, 0.01]
Ki_init = [0.04 , 0.01, 0.01]
K_in = std_msgs.msg.Float64MultiArray()
K_in.data = [ Kp_init, Kd_init, Ki_init]
Kp_cereb = [0.5, .4, .8]
Kd_cereb = [0.01 , 0.1, 0.1]
K_cereb = std_msgs.msg.Float64MultiArray()
K_cereb.data = [ Kp_cereb , Kd_cereb ]
LWPRcmd = std_msgs.msg.Float64MultiArray()
LWPRcmd.data   = [0. for i in range( 0, njoints)]
DCNcmd = std_msgs.msg.Float64MultiArray()
DCNcmd.data   = [0. for i in range( 0, njoints)]
LFcmd = std_msgs.msg.Float64MultiArray()
LFcmd.data   = [0. for i in range( 0, njoints)]
cntr_in = std_msgs.msg.Float64MultiArray()
cntr_in.data   = [0. for i in range( 0, njoints)]

des_init = sensor_msgs.msg.JointState()
des_init.position = [0. for i in range( 0, njoints)]
des_init.velocity = [0. for i in range( 0, njoints)]
#des_init = std_msgs.msg.Float64MultiArray()
#des_init.data =[ -0.09, -0.14, 0.] #[0. , 0.09, 0.78]
# -0.000105763160133 prosup , 0.436335761307 yaw, 0.799917785189 elbow
# w yaw - 0.14, w pitch 0.
pitch_range = [-1.13446, 0.174533] # effort 0.65
prosup_range  = [ -0.872665, 0.872665] # effort 0.45
yaw_range   = [-0.436332, 0.436332]
joint_rng = [ 0.872665 - 0.1, 0.436332 - 0.1, 0.13446 - 0.1]
traj_phase = [ 0. , np.pi*0.5 , 0.]


error = sensor_msgs.msg.JointState()
error.position = [0. for i in range( 0, njoints)]
error.velocity = [0. for i in range( 0, njoints)]
error.effort = [0. for i in range( 0, njoints)]

prev_error = sensor_msgs.msg.JointState()
prev_error.position = [0. for i in range( 0, njoints)]
prev_error.velocity = [0. for i in range( 0, njoints)]
prev_error.effort = [0. for i in range( 0, njoints)]
prev_errorss = [ [0.] for i in range( 0, njoints)]
int_time = [ [0.] for i in range( 0, njoints)]
# Start time for the vision-based controller
start_time = 5


# **** STATIC Controller  ****
global static_control
static_control = static_controller( njoints, Kp_init, Kd_init, Ki_init, derivation_step = 0.6, integration_step = 10. )

clientLogger.info('Waiting for ROS Service /gazebo/apply_joint_effort')
wait_for_service('/gazebo/apply_joint_effort')
clientLogger.info('Found ROS Service /gazebo/apply_joint_effort')
service_proxy = ServiceProxy('/gazebo/apply_joint_effort', gazebo_msgs.srv.ApplyJointEffort, persistent=True)
wrench_dt = Duration.from_sec(0.6)#0.08

# robot parameters
@nrp.MapVariable("n_joints", 	 initial_value = njoints)
@nrp.MapVariable("joints_idx", 	 initial_value = joints_index)
@nrp.MapVariable("current_joints",    initial_value = curr_joints)
@nrp.MapVariable("joints_range",    initial_value = joint_rng)
@nrp.MapVariable("trajectory_phase",    initial_value = traj_phase)
# variable related to the control commands
@nrp.MapVariable("K_init",    initial_value = K_in)
@nrp.MapVariable("K_cerebellum",    initial_value = K_cereb)
@nrp.MapVariable("LWPRcommand",       initial_value = LWPRcmd)
@nrp.MapVariable("DCNcommand", 	      initial_value = DCNcmd)
@nrp.MapVariable("LFcommand", 	      initial_value = LFcmd)
@nrp.MapVariable("controlcommand",    initial_value = cntr_in)
@nrp.MapVariable("error",    initial_value = error)
@nrp.MapVariable("previous_error",    initial_value = prev_error)
@nrp.MapVariable("previous_errors", 	 initial_value = prev_errorss)
@nrp.MapVariable("integral_time", 	 initial_value = int_time)
@nrp.MapVariable("desired_init",    initial_value = des_init)

# time parameters
@nrp.MapVariable("starting_time",    initial_value = start_time)
@nrp.MapVariable("LFdebug",    initial_value = True)
@nrp.MapVariable("init_pose",    initial_value = True)
# ** Map subscribers **
@nrp.MapRobotSubscriber("joints", Topic("/robot/joints", sensor_msgs.msg.JointState))

# variable related to the effort service
@nrp.MapVariable("proxy", 	 		 initial_value = service_proxy)
@nrp.MapVariable("duration", 		 initial_value = wrench_dt)


# ** Map publisher **

# -----------------------------------------------------------------------#
# ** Recording the control input **
#@nrp.MapCSVRecorder("command_recorder", filename="command.csv", headers=["time", "tau_inv1", "tau_inv2","tau_lf1","tau_lf2",
#                                                                         "dcn1","dcn2","lwpr1", "lwpr2",
#                                                                         "control_input1","control_input2"])
@nrp.MapCSVRecorder("command_recorder", filename="command.csv", headers=["time",
                                                                         "$ \tau_{PID,1}$","$ \tau_{DCN,1}$", "$ \tau_{Lwpr,1}$", "$ \tau_{tot,1}$"
                                                                         #"$ \tau_{PID,2}$","$ \tau_{DCN,2}$", "$ \tau_{Lwpr,2}$", "$ \tau_{tot,2}$",
                                                                         #"$ \tau_{PID,3}$","$ \tau_{DCN,3}$", "$ \tau_{Lwpr,3}$", "$ \tau_{tot,3}$"
                                                                         ])
@nrp.MapCSVRecorder("joint_recorder", filename="joint_info.csv", headers=["time", 
                                                                          "$ \vartheta_{r,1}$", "$ \vartheta_{1}^*$", "$ \dot \vartheta_{r,1}$", "$ \dot \vartheta_{1}^*$", "$ \dot \dot \vartheta_{1}^*$",
                                                                          #"$ \vartheta_{r,2}$", "$ \vartheta_{2}^*$", "$ \dot \vartheta_{r,2}$", "$ \dot \vartheta_{2}^*$",  
                                                                          #"$ \vartheta_{r,3}$", "$ \vartheta_{3}^*$", "$ \dot \vartheta_{r,3}$", "$ \dot \vartheta_{3}^*$", 
                                                                          "$ e_1 $", "$ \dot e_1 $", "$ \dotdot e_1 $"
                                                                          #"$ e_2 $", "$ \dot e_2 $",
                                                                          #"$ e_3 $", "$ \dot e_3 $"
                                                                          ])                                                                         
def balance_control(t, starting_time, LFdebug, init_pose,
                    proxy, duration,
                    n_joints,joints_idx, current_joints, joints, 
                    K_init, K_cerebellum,
                    LWPRcommand, DCNcommand, LFcommand, controlcommand,

                    error, previous_error, previous_errors, integral_time,
                    desired_init, joints_range, trajectory_phase,
                    command_recorder, joint_recorder
                    ):
    try:
        
        # Read encoders
        for idx in range(0,n_joints.value):#enumerate(joints_idx.value):
            current_joints.value.position[idx] = joints.value.position[joints.value.name.index(joints_idx.value[idx]) ]
            current_joints.value.velocity[idx] = joints.value.velocity[joints.value.name.index(joints_idx.value[idx]) ]
            current_joints.value.effort[idx] = joints.value.effort[joints.value.name.index(joints_idx.value[idx]) ]
        
        
        if t >1.:


            for idx in range(0, n_joints.value):
                
                desired_init.value.position[idx] = joints_range.value[idx]*np.sin(2.*np.pi*(1./4.)*t + trajectory_phase.value[idx] )      
                desired_init.value.velocity[idx] = 2.*np.pi*(1./4.)*joints_range.value[idx]*np.cos(2.*np.pi*(1./4.)*t + trajectory_phase.value[idx] )    
                
                # ** Error ** ---> reference target velocity
                error.value.position[idx] = desired_init.value.velocity[idx] - current_joints.value.velocity[idx] 
                error.value.velocity[idx] = (error.value.position[idx] - previous_error.value.position[idx] )/0.6 #(error.value.velocity[idx]-previous_error.value.velocity[idx])/0.02#0. - current_joints.value.velocity[idx]
                
 
                # ** Integral error **
                N = 10.
                if len(previous_errors.value[idx]) < int(N):
                    previous_errors.value[idx].append(error.value.position[idx])
                    integral_time.value[idx].append(t)
                    dummy_N = len(previous_errors.value[idx])
                    error.value.effort[idx]  =( (integral_time.value[idx][ dummy_N-1] - integral_time.value[idx][0])/dummy_N)* (previous_errors.value[idx][0]*0.5 + previous_errors.value[idx][dummy_N-1]*0.5 + np.sum(previous_errors.value[idx][1:-1]))

                    if LFdebug.value == True:
                        clientLogger.info("N>10 error integral "+str(error.value.effort[idx] ))
                    
                else:
                    previous_errors.value[idx].append(error.value.position[idx] )
                    previous_errors.value[idx].pop(0)
                    integral_time.value[idx].append(t)
                    integral_time.value[idx].pop(0)
                    error.value.effort[idx]  =( (integral_time.value[idx][int(N)-1] - integral_time.value[idx][0])/N)* (previous_errors.value[idx][0]*0.5 + previous_errors.value[idx][int(N)-1]*0.5 + np.sum(previous_errors.value[idx][1:-1]))
                    if LFdebug.value == True:
                        clientLogger.info("N>10 error integral "+str(error.value.effort[idx] ))
                
                if LFdebug.value == True:
                        clientLogger.info("erro joint "+str(idx)+"\n position "+str(error.value.position[idx])+"\n velocity "+str(error.value.velocity[idx])+"\n effort "+str(error.value.effort[idx]))
                previous_error.value.position[idx] = error.value.position[idx]
                
                
                LFcommand.value.data[idx] = controlcommand.value.data[idx] = K_init.value.data[0][idx]*error.value.position[idx] + K_init.value.data[1][idx]*error.value.velocity[idx] + K_init.value.data[2][idx]*error.value.effort[idx]
    
            for idx in range(0, n_joints.value):
                proxy.value.call( joints_idx.value[idx] , controlcommand.value.data[idx],  None, duration.value)
                if LFdebug.value == True:
                    clientLogger.info("\n sending command to joint "+str(joints_idx.value[idx])+" "+str(controlcommand.value.data[idx]))


            joint_recorder.record_entry(t,     desired_init.value.position[0], current_joints.value.position[0], desired_init.value.velocity[0], current_joints.value.velocity[0], current_joints.value.effort[0],
                                               #desired_init.value.position[1], current_joints.value.position[1], desired_init.value.velocity[1], current_joints.value.velocity[1], current_joints.value.effort[1],
                                               #desired_init.value.position[2], current_joints.value.position[2], desired_init.value.velocity[2], current_joints.value.velocity[2], current_joints.value.effort[2],
                                               error.value.position[0], error.value.velocity[0], error.value.effort[0]
                                               #error.value.position[1], error.value.velocity[1], error.value.effort[1],
                                               #error.value.position[2], error.value.velocity[2], error.value.effort[2]
                                          )
            command_recorder.record_entry(t,
                                               LFcommand.value.data[0], DCNcommand.value.data[0], LWPRcommand.value.data[0], controlcommand.value.data[0])
                                               #LFcommand.value.data[1], DCNcommand.value.data[1], LWPRcommand.value.data[1], controlcommand.value.data[1],
                                               #LFcommand.value.data[2], DCNcommand.value.data[2], LWPRcommand.value.data[2], controlcommand.value.data[2])

    except Exception as e:
        clientLogger.info(" --> Controller Exception: "+str(e))
        