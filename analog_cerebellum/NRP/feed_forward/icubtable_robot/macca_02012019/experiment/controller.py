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
sys.path.append("/home/silvia-neurorobotics/Documents/LWPR")
sys.path.append("/home/silvia-neurorobotics/Documents/NRP/Experiments/icub_ball_balancing")
sys.path.append("/home/silvia-neurorobotics/Documents/analog_cerebellum") # ~/ doesnt work

from pid_controller import*
from cerebellum_class import*
from norma import*



n_iter  = 500000
nin     = 8                     # ball_pos table_pos 2 * pos and vel joints
njoints = 3                    # Wrist roll and pitch

# Robot information

joints_index = ["r_wrist_prosup","r_wrist_yaw", "r_wrist_pitch","r_elbow" ]
curr_joints = sensor_msgs.msg.JointState()
curr_joints.position = [0. for i in range( 0, njoints)]
curr_joints.velocity = [0. for i in range( 0, njoints)]
curr_joints.effort = [0. for i in range( 0, njoints)]
#curr_joints.velocity = [0. for i in range( 0, njoints)]
#curr_joints.effort = [0. for i in range( 0, njoints)]


# Control parameters
# **** STATIC Controller  ****

#Kp_init = [0.3, 0.3, 0.35]#[0.09, 5.41, 10.41]
#Kd_init = [0.005 , 0.001, 0.004]#[0.3 , 0.01, 0.01]
#Ki_init = [0.4 , 0.3, 0.99]
#duration 0.2
#Kp_init = [1.9, 1.93, 2.935]#[0.09, 5.41, 10.41]
#Kd_init = [0.005 , 0.001, 0.004]#[0.3 , 0.01, 0.01]
#Ki_init = [0.94 , 0.9, 0.9]
# duration 0.5
Kp_init = [2.9, 2.3, 2.35]#[0.09, 5.41, 10.41]
Kd_init = [0.005 , 0.0001, 0.0004]#[0.3 , 0.01, 0.01]
Ki_init = [1.94 , 1.9, 1.9]
global static_control
static_control = static_controller( njoints, Kp_init, Kd_init, Ki_init, derivation_step = 0.02, integration_step = 10. )


DCNcmd = std_msgs.msg.Float64MultiArray()
DCNcmd.data   = [0. for i in range( 0, njoints)]

cntr_in = std_msgs.msg.Float64MultiArray()
cntr_in.data   = [0. for i in range( 0, njoints)]



des_init = sensor_msgs.msg.JointState()
des_init.position = [0. for i in range( 0, njoints)]
des_init.velocity = [0. for i in range( 0, njoints)]
des_init.effort = [0. for i in range( 0, njoints)]

des_joint = sensor_msgs.msg.JointState()
des_joint.position = [0. for i in range( 0, njoints)]
des_joint.velocity = [0. for i in range( 0, njoints)]
des_joint.effort = [0. for i in range( 0, njoints)]

prev_joints = sensor_msgs.msg.JointState()
prev_joints.position = [0. for i in range( 0, njoints)]
prev_joints.velocity = [0. for i in range( 0, njoints)]
prev_joints.effort = [0. for i in range( 0, njoints)]


error = sensor_msgs.msg.JointState()
error.position = [0. for i in range( 0, njoints)]

delta_err        = std_msgs.msg.Float64MultiArray()
delta_err.data   = [0. for i in range( 0, njoints)]
#des_init = std_msgs.msg.Float64MultiArray()
#des_init.data =[ -0.09, -0.14, 0.] #[0. , 0.09, 0.78]
# -0.000105763160133 prosup , 0.436335761307 yaw, 0.799917785189 elbow
# w yaw - 0.14, w pitch 0.
pitch_range = [-1.13446, 0.174533] # effort 0.65
prosup_range  = [ -0.7, 0.7] # effort 0.45
yaw_range   = [-0.436332, 0.436332]
joint_rng = [ 0.872665 - 0.7, 0.436332 - 0.3, 0.13446 - 0.1]
pose_max_range = [(prosup_range[0] - prosup_range[1]) , (yaw_range[0] -yaw_range[1]) , (pitch_range[0] -pitch_range[1])]
clientLogger.info("\n pose_max_range "+str(pose_max_range))

error_max_range = [0.4,0.3,0.1]#[(prosup_range[1]+ joint_rng[0]) , (joint_rng[1] + yaw_range[1]) , (pitch_range[1] + joint_rng[2])]
error_vel_max_range = [0.4,0.4,0.1]

velocity_max_range_des = [2.*np.pi*(1./4.)*joint_rng[0], 2.*np.pi*(1./4.)*joint_rng[1], 2.*np.pi*(1./4.)*joint_rng[2]]
clientLogger.info("\n velocity_max_range "+str(velocity_max_range_des))
traj_phase = [ 0.5*np.pi ,0.5*np.pi , 0.]

global joint_1_pose_norm, joint_2_pose_norm, joint_3_pose_norm
global joint_1_vel_norm, joint_2_vel_norm, joint_3_vel_norm
global joint_1_des_vel_norm, joint_2_des_vel_norm, joint_3_des_vel_norm

joint_1_pose_norm = norm_input(x_min_des= -1., x_max_des = 1., x_min_in = prosup_range[0] , x_max_in = prosup_range[1] )
joint_2_pose_norm = norm_input(x_min_des= -1., x_max_des = 1., x_min_in = yaw_range[0], x_max_in = yaw_range[1])
joint_3_pose_norm = norm_input(x_min_des= -1., x_max_des = 1., x_min_in = pitch_range[0], x_max_in = pitch_range[1] ) #-1.13446</lower>  <upper>0.174533

#joint_1_vel_norm = norm_input(x_min_des= -1., x_max_des = 1., x_min_in = -(2.*0.7)/0.02, x_max_in = (2.*0.7)/0.02)
#joint_2_vel_norm = norm_input(x_min_des= -1., x_max_des = 1., x_min_in = - (2.*0.436332)/0.02, x_max_in = (2.*0.436332)/0.02)
#joint_3_vel_norm = norm_input(x_min_des= -1., x_max_des = 1., x_min_in = -(2.*1.13446)/0.02, x_max_in = (2.*1.13446)/0.02) #-1.13446</lower>  <upper>0.174533
joint_1_vel_norm = norm_input(x_min_des= -1., x_max_des = 1., x_min_in = -pose_max_range[0]/0.02, x_max_in = pose_max_range[0]/0.02)
joint_2_vel_norm = norm_input(x_min_des= -1., x_max_des = 1., x_min_in = -pose_max_range[1]/0.02, x_max_in = pose_max_range[1]/0.02)
joint_3_vel_norm = norm_input(x_min_des= -1., x_max_des = 1., x_min_in = -pose_max_range[2]/0.02, x_max_in = pose_max_range[2]/0.02) #-1.13446</lower>  <upper>0.174533

joint_1_des_vel_norm = norm_input(x_min_des= -1., x_max_des = 1., x_min_in = -velocity_max_range_des[0], x_max_in = velocity_max_range_des[0])
joint_2_des_vel_norm = norm_input(x_min_des= -1., x_max_des = 1.,  x_min_in = -velocity_max_range_des[1], x_max_in = velocity_max_range_des[1])
joint_3_des_vel_norm = norm_input(x_min_des= -1., x_max_des = 1.,  x_min_in = -velocity_max_range_des[2], x_max_in = velocity_max_range_des[2]) #-1.13446</lower>  <upper>0.174533

# Start time for the vision-based controller
start_time = 5

global cerebellum, cerebellum_fwd
t_cerebellum     = 1 #enable cerebellum 1, disable 0
debug_cerebellum = 0
experiment       = 1
test_lwpr = 1

if t_cerebellum == 1 :
    #               [ [ j_1_pos, ... , j_n_pos], [ j_1_vel, ... , j_n_vel] ]
    init_d_mf     = [ [  0.5  ,  0.5  ,  0.5  ], [  0.5  ,  0.5  ,  0.5  ] ]
    init_alpha_mf = [ [ 190.   , 190.   , 190.   ], [ 90.   , 90.   , 90.   ] ]
    w_g_mf        = [ [  0.4  ,  0.4  ,  0.4  ], [  0.4  ,  0.4  ,  0.4  ] ]
    w_pru_mf      = [ [  0.8 ,  0.8 ,  0.8 ], [  0.98 ,  0.98 ,  0.98 ] ]
    Beta = 7.* math.pow(10, -3) #2
    
    n_input_lwpr_pf        = 9-3       # number of LWPR input - Mossy fiber = 1 desired pose + 1 desired vel+ + 1 desired acc + 2 current pose + 2 current vel +  2 current acc --> for each joint considered input both the  acceleration[rad/sec^2], velocity[rad/sec] and the position [rad]
    n_input_lwpr_mossy     = [ 5, 5,5] # per each joint : joint_pose , error ball pose x,  error ball pose y,
    n_input_lwpr_mossy_vel = [ 5, 5,5]
    
    n_lwpr_out   = 1
    n_ulm_joints = 1

    cerebellum = Cerebellum( njoints , n_lwpr_out , n_input_lwpr_mossy , n_input_lwpr_mossy_vel, n_input_lwpr_pf)
    cerebellum.name = " Cerebellum inverse "
    # ********************** Cerebellum Parameters **********************
    max_dist_pos = 0.4#np.sqrt(2.*(150./300.)**2.)
    max_dist_vel = max_dist_pos/0.02#np.sqrt(2.*(150./(0.02*300))**2.)    
    # Teaching signal constraints
    cerebellum.min_teach_pos =  [0., 0., 0.]
    cerebellum.max_teach_pos = error_max_range#[max([abs(i) for i in prosup_range] ), max([abs(i) for i in yaw_range] ), max([abs(i) for i in pitch_range] )]#[pose_max_range[0],pose_max_range[1],pose_max_range[2]]#[velocity_max_range[0], velocity_max_range[1], velocity_max_range[2]] #[ abs(prosup_range[0]- prosup_range[1]), abs(yaw_range[0]- yaw_range[1]), abs(pitch_range[0]- pitch_range[1])]#[ max_dist_pos , max_dist_pos, max_dist_pos]
    cerebellum.min_teach_vel =  [0.,0., 0.]
    cerebellum.max_teach_vel =  error_vel_max_range#[ v/0.02 for v in cerebellum.max_teach_pos ]#[ max_dist_vel, max_dist_vel, max_dist_vel]
    cerebellum.signal_normalization = 1
    cerebellum.tau_norm_sign = 0
    cerebellum.min_signal_pos    = [0. for i in range( 0, njoints)]       
    cerebellum.max_signal_pos    = [(Kp_init[i]*error_max_range[i] + Kd_init[i]*error_max_range[i]/0.02 + Ki_init[i]*error_max_range[i]/10.) for i in range( 0, njoints)]#[(Kp_init[i]*0.4 + 20.*Kd_init[i] + 0.4*Ki_init[i]) for i in range( 0, njoints)]
    cerebellum.min_signal_vel    = [0. for i in range( 0, njoints)]       
    cerebellum.max_signal_vel    = [vte for vte in cerebellum.max_signal_pos]
    cerebellum.mean_torq_dcn = 1
    cerebellum.debug_prediction = 0
    cerebellum.rfs_print  = 1
    # Mossy Fibers Sensorial Information Mapping
    cerebellum.diag_only_mf  = bool(1)
    cerebellum.update_D_mf   = bool(0)
    cerebellum.meta_mf       = bool(0)
    for q in range(0,2):
        for i in range( 0 , cerebellum.n_uml ):
            cerebellum.init_D_mf[q][i]        = init_d_mf[q][i]
            cerebellum.init_alpha_mf[q][i]    = init_alpha_mf[q][i]
            cerebellum.w_gen_mf[q][i]         = w_g_mf[q][i]
            #cerebellum.init_lambda_mf[i]   = init_lambda_mf[i]
            #cerebellum.tau_lambda_mf[i]    = [i]
            #cerebellum.final_lambda_mf[i]  = 
            cerebellum.w_prune_mf[q][i]       = w_pru_mf[q][i]
            #cerebellum.meta_rate_mf[i]     = 
            #cerebellum.add_threshold_mf[i] = 
    
    # Parallel Fibers Sensorial Information Mapping
    cerebellum.init_D_pf     = 10.#50.#100.
    cerebellum.init_alpha_pf = 180.#100.
    cerebellum.w_gen_pf      = 0.4
    cerebellum.w_prune_pf    = 0.95
    cerebellum.diag_only_pf  = bool(1)  #1
    cerebellum.update_D_pf   = bool(0) #0
    cerebellum.meta_pf       = bool(0)
    
    # *** Plasticity ***
    #exc
    cerebellum.ltpPF_PC_max = 1. * 10**(-3) # -4
    cerebellum.ltdPF_PC_max = 1. * 10**(-3) # -5
    #inh
    cerebellum.ltpPC_DCN_max = 1. * 10**(-4) #-2  # 4
    cerebellum.ltdPC_DCN_max = 1. * 10**(-4) #-3  # 3
    #exc
    cerebellum.ltpMF_DCN_max = 1. * 10**(-4) #-3
    cerebellum.ltdMF_DCN_max = 1. * 10**(-4) #-2
    #exc
    cerebellum.ltpIO_DCN_max = 1. * 10**(-4) #-4
    cerebellum.ltdIO_DCN_max = 1. * 10**(-5) #-3
     
    #self.alpha = 1
    cerebellum.alphaPF_PC_pos  = 170.  #self.ltd_max / self.ltp_max 50, 7
    cerebellum.alphaPF_PC_vel  = 170.    
    cerebellum.alphaPC_DCN     = 2. # self.ltd_PC_DCN_max / self.ltp_PC_DCN_max
    cerebellum.alphaMF_DCN     = 2. # self.ltd_PC_DCN_max / self.ltp_PC_DCN_max
    cerebellum.alphaIO_DCN     = 100.
    cerebellum.IO_on = False
    cerebellum.PC_on = True
    cerebellum.MF_on = True
    
    #cerebellum.w_pf_pct = [0.5 for i in range( 0, njoints)]
    #cerebellum.w_pf_pct_vel = [0.5 for i in range( 0, njoints)]
    
    # ******************************************************  forward cerebellum
        #               [ [ j_1_pos, ... , j_n_pos], [ j_1_vel, ... , j_n_vel] ]
    init_d_mf     = [ [  0.5  ,  0.5  ,  0.5  ], [  0.5  ,  0.5  ,  0.5  ] ]
    init_alpha_mf = [ [ 90.   , 90.   , 90.   ], [ 90.   , 90.   , 90.   ] ]
    w_g_mf        = [ [  0.4  ,  0.4  ,  0.4  ], [  0.4  ,  0.4  ,  0.4  ] ]
    w_pru_mf      = [ [  0.98 ,  0.98 ,  0.98 ], [  0.98 ,  0.98 ,  0.98 ] ]
    Beta = 7.* math.pow(10, -3) #2
    
    n_input_lwpr_pf        = 9       # number of LWPR input - Mossy fiber = 1 desired pose + 1 desired vel+ + 1 desired acc + 2 current pose + 2 current vel +  2 current acc --> for each joint considered input both the  acceleration[rad/sec^2], velocity[rad/sec] and the position [rad]
    n_input_lwpr_mossy     = [ 4, 4,4] # per each joint : joint_pose , error ball pose x,  error ball pose y,
    n_input_lwpr_mossy_vel = [ 4, 4,4]
    
    n_lwpr_out   = 1
    n_ulm_joints = 1

    cerebellum_fwd = Cerebellum( njoints , n_lwpr_out , n_input_lwpr_mossy , n_input_lwpr_mossy_vel, n_input_lwpr_pf)
    cerebellum_fwd.name = " Cerebellum foward "
    # ********************** Cerebellum Parameters **********************
    max_dist_pos = 0.5#0.5#np.sqrt(2.*(150./300.)**2.)
    max_dist_vel = 0.5#25.#5.#max_dist_pos#np.sqrt(2.*(150./(0.02*300))**2.)    
    # Teaching signal constraints
    cerebellum_fwd.min_teach_pos =  [0., 0., 0.]
    cerebellum_fwd.max_teach_pos =  [ max_dist_pos , max_dist_pos, max_dist_pos]
    cerebellum_fwd.min_teach_vel =  [0.,0., 0.]
    cerebellum_fwd.max_teach_vel =  [ max_dist_vel, max_dist_vel, max_dist_vel]
    cerebellum_fwd.signal_normalization = 1
    cerebellum_fwd.tau_norm_sign = 0
    
    cerebellum_fwd.min_signal_pos    = [ prosup_range[0], yaw_range[0], pitch_range[0]] #[0. for i in range( 0, njoints)]       
    cerebellum_fwd.max_signal_pos    = [ prosup_range[1], yaw_range[1], pitch_range[1]]#[0.4 for i in range( 0,njoints)]
    cerebellum_fwd.min_signal_vel    = [ prosup_range[0], yaw_range[0], pitch_range[0]]#[0. for i in range( 0, njoints)]       
    cerebellum_fwd.max_signal_vel    = [ prosup_range[1], yaw_range[1], pitch_range[1]]#[20. for i in range( 0, njoints)]
    
    cerebellum_fwd.mean_torq_dcn = 1
    cerebellum_fwd.debug_prediction = 0
    cerebellum_fwd.rfs_print  = 1
    # Mossy Fibers Sensorial Information Mapping
    cerebellum_fwd.diag_only_mf  = bool(1)
    cerebellum_fwd.update_D_mf   = bool(0)
    cerebellum_fwd.meta_mf       = bool(0)
    for q in range(0,2):
        for i in range( 0 , cerebellum_fwd.n_uml ):
            cerebellum_fwd.init_D_mf[q][i]        = init_d_mf[q][i]
            cerebellum_fwd.init_alpha_mf[q][i]    = init_alpha_mf[q][i]
            cerebellum_fwd.w_gen_mf[q][i]         = w_g_mf[q][i]
            #cerebellum_fwd.init_lambda_mf[i]   = init_lambda_mf[i]
            #cerebellum_fwd.tau_lambda_mf[i]    = [i]
            #cerebellum_fwd.final_lambda_mf[i]  = 
            cerebellum_fwd.w_prune_mf[q][i]       = w_pru_mf[q][i]
            #cerebellum_fwd.meta_rate_mf[i]     = 
            #cerebellum_fwd.add_threshold_mf[i] = 
    
    # Parallel Fibers Sensorial Information Mapping
    cerebellum_fwd.init_D_pf     = 5.#0.3
    cerebellum_fwd.init_alpha_pf = 90.
    cerebellum_fwd.w_gen_pf      = 0.4
    cerebellum_fwd.w_prune_pf    = 0.98
    cerebellum_fwd.diag_only_pf  = bool(1)  #1
    cerebellum_fwd.update_D_pf   = bool(0) #0
    cerebellum_fwd.meta_pf       = bool(0)
    
    # *** Plasticity ***
    #exc

    cerebellum_fwd.ltpPF_PC_max = 1. * 10**(-3) # -4
    cerebellum_fwd.ltdPF_PC_max = 1. * 10**(-3) # -5
    #inh
    cerebellum_fwd.ltpPC_DCN_max = 1. * 10**(-4) #-2  # 4
    cerebellum_fwd.ltdPC_DCN_max = 1. * 10**(-4) #-3  # 3
    #exc
    cerebellum_fwd.ltpMF_DCN_max = 1. * 10**(-4) #-3
    cerebellum_fwd.ltdMF_DCN_max = 1. * 10**(-4) #-2
    #exc
    cerebellum_fwd.ltpIO_DCN_max = 1. * 10**(-5) #-4
    cerebellum_fwd.ltdIO_DCN_max = 1. * 10**(-5) #-3
     
    #self.alpha = 1
    cerebellum_fwd.alphaPF_PC_pos  = 20.  #self.ltd_max / self.ltp_max 50, 7
    cerebellum_fwd.alphaPF_PC_vel  = 20.    
    cerebellum_fwd.alphaPC_DCN     = 2. # self.ltd_PC_DCN_max / self.ltp_PC_DCN_max
    cerebellum_fwd.alphaMF_DCN     = 2. # self.ltd_PC_DCN_max / self.ltp_PC_DCN_max
    cerebellum_fwd.alphaIO_DCN     = 2.
    
    cerebellum_fwd.IO_on = False
    cerebellum_fwd.PC_on = True
    cerebellum_fwd.MF_on = True


cerebellum.create_models()
cerebellum_fwd.create_models()
clientLogger.info("\n The Cerebellum is Initialized ")
t_cerebellum = 20.
t_cerebellum_predict = t_cerebellum + 20.
t_cerebellum_update = t_cerebellum_predict + 120.
# robot parameters
@nrp.MapVariable("n_joints",            initial_value = njoints)
@nrp.MapVariable("joints_idx",          initial_value = joints_index)
@nrp.MapVariable("current_joints",      initial_value = curr_joints)
@nrp.MapVariable("previous_joints",      initial_value = prev_joints)
@nrp.MapVariable("joints_range",        initial_value = joint_rng)
@nrp.MapVariable("trajectory_phase",    initial_value = traj_phase)
@nrp.MapVariable("joint_error",    initial_value = error)
# variable related to the control commands



@nrp.MapVariable("DCNcommand",          initial_value  = DCNcmd)
@nrp.MapVariable("controlcommand",      initial_value  = cntr_in)

@nrp.MapVariable("desired_init",        initial_value  = des_init)
@nrp.MapVariable("desired_joint",        initial_value  = des_joint)

@nrp.MapVariable("delta_error",      initial_value = delta_err)
# time parameters
@nrp.MapVariable("starting_time",       initial_value = start_time)
@nrp.MapVariable("LFdebug",             initial_value = False)
@nrp.MapVariable("init_pose",           initial_value = True)

@nrp.MapVariable("cerebellum_debug",             initial_value = False)
@nrp.MapVariable("cerebellum_on",          initial_value = True)
@nrp.MapVariable("cerebellum_dynamics_on", initial_value = True)
@nrp.MapVariable("cerebellum_kinematics_on", initial_value = False)

@nrp.MapVariable("time_cerebellum",     initial_value = t_cerebellum)
@nrp.MapVariable("time_cerebellum_predict",     initial_value = t_cerebellum_predict)
@nrp.MapVariable("time_cerebellum_update",     initial_value = t_cerebellum_update)
# variable related to the effort service
#@nrp.MapVariable("proxy",               initial_value = service_proxy)
#@nrp.MapVariable("duration",            initial_value = wrench_dt)

# ** Map subscribers **
@nrp.MapRobotSubscriber("joints", Topic("/robot/joints", sensor_msgs.msg.JointState))
@nrp.MapRobotSubscriber("ball_error", Topic('/ball/error', sensor_msgs.msg.JointState))
@nrp.MapRobotSubscriber("ball_distance", Topic('/ball/distance', std_msgs.msg.Float64MultiArray) )
@nrp.MapRobotSubscriber("pose_ball_sub",  Topic('/ball/pose', sensor_msgs.msg.JointState)) # table pose in effort information

# ** Map publisher **
@nrp.MapRobotPublisher("command_publisher", Topic('/effort_command', std_msgs.msg.Float64MultiArray ))

# -----------------------------------------------------------------------#
# ** Recording the control input **

@nrp.MapCSVRecorder("command_recorder", filename="command.csv", headers=["time",
                                                                         "tau_{PID,0}","tau_{DCN,0}", "tau_{tot,0}",
                                                                         "tau_{PID,1}","tau_{DCN,1}", "tau_{tot,1}",
                                                                         "tau_{PID,2}","tau_{DCN,2}", "tau_{tot,2}"
                                                                         ])
@nrp.MapCSVRecorder("joint_recorder", filename="joint_info.csv", headers=["time", 
                                                                          #"vel_{rc,0}", "vel_{rc,1}", "vel_{rc,2}",
                                                                          "pose_{r,0}", "pose_{c,0}", "vel_{r,0}", "vel_{c,0}", "acc_{r,0}", "acc_{c,0}",
                                                                          "pose_{r,1}", "pose_{c,1}", "vel_{r,1}", "vel_{c,1}", "acc_{r,1}", "acc_{c,1}",
                                                                          "pose_{r,2}", "pose_{c,2}", "vel_{r,2}", "vel_{c,2}", "acc_{r,2}", "acc_{c,2}",
                                                                          "e_0", "dot_e_0", "dot_dot_e_0", "delta_e_0" ,        
                                                                          "e_1", "dot_e_1", "dot_dot_e_1", "delta_e_1" ,
                                                                          "e_2", "dot_e_2", "dot_dot_e_2", "delta_e_2" 
                                                                          ])    
@nrp.MapCSVRecorder("cerebellum_kinematics_recorder", filename="cerebellum_kinematics.csv", headers=[ "time",
                                                                                   #"rfs_mossy_pos",
                                                                                   #"rfs_mossy_vel",
                                                                                   #"rfs_pf",
                                                                                   "kine_input_{pf,j0}", "kine_input_{pf,j1}", "kine_input_{pf,j2]",
                                                                                   "kine_output_{pf,j0}", "kine_output_{pf,j1}", "kine_output_{pf,j2]",
                                                                                   "kine_output_{io,pos,j0}", "kine_output_{io,pos,j1}", "kine_output_{io,pos,j2}",
                                                                                   "kine_output_{io,vel,j0}", "kine_output_{io,vel,j1}", "kine_output_{io,vel,j2}",  
                                                                                   
                                                                                   "kine_w_{pf-pc,pos,j0}", "kine_w_{pf-pc,pos,j1}", "kine_w_{pf-pc,pos,j2}",
                                                                                   "kine_w_{pf-pc,vel,j0}", "kine_w_{pf-pc,vel,j1}", "kine_w_{pf-pc,vel,j2}",
                                                                                   "kine_output_{pc,pos,j0}", "kine_output_{pc,pos,j1}", "kine_output_{pc,pos,j2}",
                                                                                   "kine_output_{pc,vel,j0}", "kine_output_{pc,vel,j1}", "kine_output_{pc,vel,j2}",
                                                                                   
                                                                                   "kine_w_{pc-dcn,pos,j0}", "kine_w_{pc-dcn,pos,j1}", "kine_w_{pc-dcn,pos,j2}",
                                                                                   "kine_w_{pc-dcn,vel,j0}", "kine_w_{pc-dcn,vel,j1}", "kine_w_{pc-dcn,vel,j2}",                    
                                                                                   "kine_output_{pc-dcn,pos,j0}", "kine_output_{pc-dcn,pos,j1}", "kine_output_{pc-dcn,pos,j2}",
                                                                                   "kine_output_{pc-dcn,vel,j0}", "kine_output_{pc-dcn,vel,j1}", "kine_output_{pc-dcn,vel,j2}",
                                                                                   
                                                                                   "kine_output_{mossy,pos,j0}", "kine_output_{mossy,pos,j1}", "kine_output_{mossy,pos,j2}",
                                                                                   "kine_output_{mossy,vel,j0}", "kine_output_{mossy,vel,j1}", "kine_output_{mossy,vel,j2}",
                                                                                   "kine_w_{mf-dcn,pos,j0}", "kine_w_{mf-dcn,pos,j1}", "kine_w_{mf-dcn,pos,j2}",
                                                                                   "kine_w_{mf-dcn,vel,j0}", "kine_w_{mf-dcn,vel,j1}", "kine_w_{mf-dcn,vel,j2}",
                                                                                   
                                                                                   "kine_output_{mf-dcn,pos,j0}", "kine_output_{mf-dcn,pos,j1}", "kine_output_{mf-dcn,pos,j2}",
                                                                                   "kine_output_{mf-dcn,vel,j0}", "kine_output_{mf-dcn,vel,j1}", "kine_output_{mf-dcn,vel,j2}",
                                                                                   
                                                                                   "kine_w_{io-dcn,pos,j0}", "kine_w_{io-dcn,pos,j1}", "kine_w_{io-dcn,pos,j2}",
                                                                                   "kine_w_{io-dcn,vel,j0}", "kine_w_{io-dcn,vel,j1}", "kine_w_{io-dcn,vel,j2}",  
                                                                                   #"output_IO-DCN_pos",
                                                                                   #"output_IO-DCN_vel",
                                                                                   "kine_output_{dcn,j0}", "kine_output_{dcn,j1}", "kine_output_{dcn,j2}"
                                                                                   ])
@nrp.MapCSVRecorder("cerebellum_dynamics_recorder", filename="cerebellum_dynamics.csv", headers=[ "time",
                                                                                   #"rfs_mossy_pos",
                                                                                   #"rfs_mossy_vel",
                                                                                   #"rfs_pf",
                                                                                   "dyna_input_{pf,j0}", "dyna_input_{pf,j1}", "dyna_input_{pf,j2}",
                                                                                   "dyna_output_{pf,j0}", "dyna_output_{pf,j1}", "dyna_output_{pf,j2}",
                                                                                   "dyna_output_{io,pos,j0}", "dyna_output_{io,pos,j1}", "dyna_output_{io,pos,j2}",
                                                                                   "dyna_output_{io,vel,j0}", "dyna_output_{io,vel,j1}", "dyna_output_{io,vel,j2}",  
                                                                                   
                                                                                   "dyna_w_{pf-pc,pos,j0}", "dyna_w_{pf-pc,pos,j1}", "dyna_w_{pf-pc,pos,j2}",
                                                                                   "dyna_w_{pf-pc,vel,j0}", "dyna_w_{pf-pc,vel,j1}", "dyna_w_{pf-pc,vel,j2}",
                                                                                   "dyna_output_{pc,pos,j0}", "dyna_output_{pc,pos,j1}", "dyna_output_{pc,pos,j2}",
                                                                                   "dyna_output_{pc,vel,j0}", "dyna_output_{pc,vel,j1}", "dyna_output_{pc,vel,j2}",
                                                                                   
                                                                                   "dyna_w_{pc-dcn,pos,j0}", "dyna_w_{pc-dcn,pos,j1}", "dyna_w_{pc-dcn,pos,j2}",
                                                                                   "dyna_w_{pc-dcn,vel,j0}", "dyna_w_{pc-dcn,vel,j1}", "dyna_w_{pc-dcn,vel,j2}",                    
                                                                                   "dyna_output_{pc-dcn,pos,j0}", "dyna_output_{pc-dcn,pos,j1}", "dyna_output_{pc-dcn,pos,j2}",
                                                                                   "dyna_output_{pc-dcn,vel,j0}", "dyna_output_{pc-dcn,vel,j1}", "dyna_output_{pc-dcn,vel,j2}",
                                                                                   
                                                                                   "dyna_output_{mossy,pos,j0}", "dyna_output_{mossy,pos,j1}", "dyna_output_{mossy,pos,j2}",
                                                                                   "dyna_output_{mossy,vel,j0}", "dyna_output_{mossy,vel,j1}", "dyna_output_{mossy,vel,j2}",
                                                                                   "dyna_w_{mf-dcn,pos,j0}", "dyna_w_{mf-dcn,pos,j1}", "dyna_w_{mf-dcn,pos,j2}",
                                                                                   "dyna_w_{mf-dcn,vel,j0}", "dyna_w_{mf-dcn,vel,j1}", "dyna_w_{mf-dcn,vel,j2}",
                                                                                   
                                                                                   "dyna_output_{mf-dcn,pos,j0}", "dyna_output_{mf-dcn,pos,j1}", "dyna_output_{mf-dcn,pos,j2}",
                                                                                   "dyna_output_{mf-dcn,vel,j0}", "dyna_output_{mf-dcn,vel,j1}", "dyna_output_{mf-dcn,vel,j2}",
                                                                                   
                                                                                   "dyna_w_{io-dcn,pos,j0}", "dyna_w_{io-dcn,pos,j1}", "dyna_w_{io-dcn,pos,j2}",
                                                                                   "dyna_w_{io-dcn,vel,j0}", "dyna_w_{io-dcn,vel,j1}", "dyna_w_{io-dcn,vel,j2}",  
                                                                                   #"output_IO-DCN_pos",
                                                                                   #"output_IO-DCN_vel",
                                                                                   "dyna_output_{dcn,j0}", "dyna_output_{dcn,j1}", "dyna_output_{dcn,j2}"
                                                                                   ])

                                                                                    
def balance_control(t, 
                    starting_time, LFdebug, init_pose,
                    #proxy, duration,
                    n_joints,joints_idx, current_joints, joints, previous_joints, 
                    desired_joint,
                    DCNcommand, controlcommand,
                    command_publisher,
                    desired_init, joints_range, trajectory_phase,
                    cerebellum_on, cerebellum_dynamics_on, cerebellum_kinematics_on,
                    time_cerebellum, time_cerebellum_predict, time_cerebellum_update, cerebellum_debug,
                    ball_error, ball_distance, pose_ball_sub,
                    command_recorder, joint_recorder, cerebellum_dynamics_recorder, cerebellum_kinematics_recorder,
                    delta_error, joint_error
                    ):
    try:
        init_joint_idx = 0
        # Read encoders
        for idx in range( init_joint_idx ,n_joints.value):
            
        
            current_joints.value.position[idx] = joints.value.position[joints.value.name.index(joints_idx.value[idx]) ]
            current_joints.value.velocity[idx] = joints.value.velocity[joints.value.name.index(joints_idx.value[idx]) ]
            current_joints.value.effort[idx]   = joints.value.effort[joints.value.name.index(joints_idx.value[idx]) ]
        
        
        if t >3.:
 
            for idx in range( init_joint_idx , n_joints.value):
                
                if init_pose.value == True:
                    init_pose.value = False
                    previous_joints.value.position[idx] = current_joints.value.position[idx]
                desired_init.value.position[idx] =  joints_range.value[idx]*np.sin(2.*np.pi*(1./4.)*t + trajectory_phase.value[idx] )      
                desired_init.value.velocity[idx] = 2.*np.pi*(1./4.)*joints_range.value[idx]*np.cos(2.*np.pi*(1./4.)*t + trajectory_phase.value[idx] )   
                desired_init.value.effort[idx] =  -4.*(np.pi**2.)*(1./4.**2.)*joints_range.value[idx]*np.sin(2.*np.pi*(1./4.)*t + trajectory_phase.value[idx] )  
                joint_error.value.position[idx] = desired_init.value.position[idx] -  current_joints.value.position[idx]
            
            
                
            # =================================== Cerebellum Storing sensorial information ===================================
            
            if cerebellum_on.value == 1 and t > time_cerebellum.value :
                
                # ****** Correction Dynamics ******
                if cerebellum_dynamics_on.value == True :
                    
                    # Input parallel fibers
                    cerebellum.input_pf = [   #joint_1_pose_norm.get_norm( current_joints.value.position[0] ), joint_2_pose_norm.get_norm( current_joints.value.position[1] ), joint_3_pose_norm.get_norm( current_joints.value.position[2] ),
                                              #joint_1_vel_norm.get_norm(  current_joints.value.velocity[0] ),     joint_2_vel_norm.get_norm(  current_joints.value.velocity[1] ), joint_3_vel_norm.get_norm(  current_joints.value.velocity[2] ), 
                                              #joint_1_des_vel_norm.get_norm(    desired_init.value.velocity[0] ),     joint_2_des_vel_norm.get_norm(    desired_init.value.velocity[1] ), joint_3_des_vel_norm.get_norm(    desired_init.value.velocity[2] )
                                              #current_joints.value.position[0], current_joints.value.position[1], current_joints.value.position[2],
                                              current_joints.value.velocity[0], current_joints.value.velocity[1], current_joints.value.velocity[2],
                                              desired_init.value.velocity[0], desired_init.value.velocity[1], desired_init.value.velocity[2]
                                              ]
                    # Input mossy fibers
                    for j in range(init_joint_idx , n_joints.value) :
                        cerebellum.input_mossy[j]     = controlcommand.value.data[j]
                        cerebellum.input_mossy_vel[j] = controlcommand.value.data[j]   
                    
                    # Debug
                    if cerebellum_debug.value == True:
                        clientLogger.info("\n Dynamics: input parallel fibers   :\n"+str(cerebellum.input_pf))
                        clientLogger.info("\n Dynamics: input mossy fibers pose :\n"+str(cerebellum.input_mossy))
                        clientLogger.info("\n Dynamics: input mossy fibers vel  :\n"+str(cerebellum.input_mossy_vel))
                    
                # ****** Correction Kinematics ******                                                     
                if cerebellum_kinematics_on.value == True :
                    
                    # Input parallel fibers
                    cerebellum_fwd.input_pf = [    #joint_1_pose_norm.get_norm( current_joints.value.position[0] ),     joint_2_pose_norm.get_norm( current_joints.value.position[1] ), joint_3_pose_norm.get_norm( current_joints.value.position[2] ),
                                               joint_1_vel_norm.get_norm(  current_joints.value.velocity[0] ),     joint_2_vel_norm.get_norm(  current_joints.value.velocity[1] ), joint_3_vel_norm.get_norm(  current_joints.value.velocity[2] ), 
                                               joint_1_vel_norm.get_norm(    desired_init.value.velocity[0] ),     joint_2_vel_norm.get_norm(    desired_init.value.velocity[1] ), joint_3_vel_norm.get_norm(    desired_init.value.velocity[2] ),
                                                                            pose_ball_sub.value.position[0]/240.,                               pose_ball_sub.value.position[1]/320.,
                                                                            pose_ball_sub.value.velocity[0]/240.,                               pose_ball_sub.value.velocity[1]/320.,
                                                                              pose_ball_sub.value.effort[0]/240.,                                 pose_ball_sub.value.effort[1]/320. ]
                    # Input mossy fibers
                    for j in range(init_joint_idx , n_joints.value) :
                        cerebellum_fwd.input_mossy[j]     = desired_init.value.velocity[j]#static_control.error_position[j]
                        cerebellum_fwd.input_mossy_vel[j] = desired_init.value.velocity[j]#static_control.error_position[j]
                    
                    # Debug
                    if cerebellum_debug.value == True:
                        clientLogger.info("\n Dynamics: input parallel fibers   :\n"+str(cerebellum_fwd.input_pf))
                        clientLogger.info("\n Dynamics: input mossy fibers pose :\n"+str(cerebellum_fwd.input_mossy))
                        clientLogger.info("\n Dynamics: input mossy fibers vel  :\n"+str(cerebellum_fwd.input_mossy_vel))
                


                # =================================== Cerebellum Update  ===================================
                if t <= time_cerebellum_update.value :
                    
                    if cerebellum_dynamics_on.value == True:
                        cerebellum.update_models( controlcommand.value.data, controlcommand.value.data)
                    elif cerebellum_kinematics_on.value == True:
                        cerebellum_fwd.update_models( desired_joint.value.velocity ,desired_joint.value.velocity  )#static_control.error_position, static_control.error_position)


                # =================================== Cerebellum Foward Model Predict ===================================
                
                if t > time_cerebellum_predict.value and cerebellum_kinematics_on.value == True :
                    #DCNcommand.value.data  = cerebellum.prediction( static_control.PID, [ ball_distance.value.data[0] for j in range(0, n_joints.value)], [ ball_distance.value.data[0] for j in range(0, n_joints.value) ] )# mean_tau.value.data )
                    
                    delta_error.value.data = cerebellum_fwd.prediction( [ abs(ball_distance.value.data[0]) for j in range(0, n_joints.value)], [ abs(ball_distance.value.data[1]) for j in range(0, n_joints.value) ] )                    
                                        
                    if cerebellum_debug.value == True:
                        clientLogger.info("\n Foward model correction : "+str(delta_error.value.data))

    
            # =================================== Static Control Command ===================================
            # correct the trajectory
            for idx in range(init_joint_idx , n_joints.value):
                desired_joint.value.velocity[idx] = desired_init.value.velocity[idx] + delta_error.value.data[idx]
            
            # Compute static controller contribution
            #PID = control( target, current state, delta error)
            
            #static_control.control( desired_init.value.velocity, current_joints.value.velocity, delta_error.value.data, t)
            static_control.control( desired_joint.value.velocity, current_joints.value.velocity, [0. for i in range( init_joint_idx , n_joints.value)], t)
            
            # Debug            
            if LFdebug.value == True:
                clientLogger.info("\n static controller contribution : "+str(static_control.PID))
            
            
            # =================================== Cerebellum Inverse Model Predict ===================================
            
            if cerebellum_dynamics_on.value == True and t > time_cerebellum_predict.value and cerebellum_on.value == 1 :
                
                # Correction = self.prediction( IO pose input, Io vel input)                
                DCNcommand.value.data  = cerebellum.prediction(  joint_error.value.position, static_control.error_position )#, static_control.error_velocity )
                
                if cerebellum_debug.value == True:
                     clientLogger.info("\n DCN output : "+str(DCNcommand.value.data))
            
            # =================================== Control Command ===================================

            for idx in range( init_joint_idx , n_joints.value):
                controlcommand.value.data[idx] = static_control.PID[idx] + DCNcommand.value.data[idx]- cerebellum.output_IO_DCN[idx] # cerebellum.output_pc[idx] + cerebellum.output_pc_vel[idx]#+ DCNcommand.value.data[idx]
            if LFdebug.value == True:
                         clientLogger.info("\n control output : "+str(controlcommand.value.data))  
            command_publisher.send_message( controlcommand.value )
            
            


            # =================================== Record  ===================================

            joint_recorder.record_entry(t,     #desired_joint.value.velocity[0],desired_joint.value.velocity[1],desired_joint.value.velocity[2],  
                                        desired_init.value.position[0], current_joints.value.position[0], desired_init.value.velocity[0], current_joints.value.velocity[0], desired_init.value.effort[0],  current_joints.value.effort[0],
                                                 desired_init.value.position[1], current_joints.value.position[1], desired_init.value.velocity[1], current_joints.value.velocity[1], desired_init.value.effort[1],  current_joints.value.effort[1],
                                                 desired_init.value.position[2], current_joints.value.position[2], desired_init.value.velocity[2], current_joints.value.velocity[2], desired_init.value.effort[2],  current_joints.value.effort[2],
                                               static_control.error_position[0], static_control.error_velocity[0], static_control.error_effort[0], delta_error.value.data[0],
                                               static_control.error_position[1], static_control.error_velocity[1], static_control.error_effort[1], delta_error.value.data[1],
                                               static_control.error_position[2], static_control.error_velocity[2], static_control.error_effort[2], delta_error.value.data[2]
                                          )
            command_recorder.record_entry(t,
                                          
                                          static_control.PID[0], DCNcommand.value.data[0], controlcommand.value.data[0],
                                          static_control.PID[1], DCNcommand.value.data[1], controlcommand.value.data[1],
                                          static_control.PID[2], DCNcommand.value.data[2], controlcommand.value.data[2]
                                              )
            if t > time_cerebellum_predict.value:
                if cerebellum_kinematics_on.value == True :
                    cerebellum_kinematics_recorder.record_entry(t, # 22
                                                               #cerebellum.uml_mossy[0].num_rfs[0], 
                                                               #cerebellum.uml_mossy_vel[0].num_rfs[0],
                                                               #cerebellum.uml_pf.num_rfs[0],
                                                               #static_control.error_position[0], static_control.error_position[1],static_control.error_position[2],
                                                               desired_joint.value.velocity[0], desired_joint.value.velocity[1], desired_joint.value.velocity[2],
                                                               cerebellum_fwd.output_x_pf[0], cerebellum_fwd.output_x_pf[1], cerebellum_fwd.output_x_pf[2],
                    
                                                               cerebellum_fwd.IO_fbacktorq[0], cerebellum_fwd.IO_fbacktorq[1], cerebellum_fwd.IO_fbacktorq[2],
                                                               cerebellum_fwd.vel_IO_fbacktorq[0], cerebellum_fwd.vel_IO_fbacktorq[1], cerebellum_fwd.vel_IO_fbacktorq[2],
                                                               cerebellum_fwd.w_pf_pc[0],         cerebellum_fwd.w_pf_pc[1],    cerebellum_fwd.w_pf_pc[2],
                                                               cerebellum_fwd.w_pf_pc_vel[0], cerebellum_fwd.w_pf_pc_vel[1], cerebellum_fwd.w_pf_pc_vel[2],
                                                               cerebellum_fwd.output_pc[0],     cerebellum_fwd.output_pc[1],     cerebellum_fwd.output_pc[2],  
                                                               cerebellum_fwd.output_pc_vel[0], cerebellum_fwd.output_pc_vel[1], cerebellum_fwd.output_pc_vel[2],
                                                               
                                                               cerebellum_fwd.w_pc_dcn[0], cerebellum_fwd.w_pc_dcn[1], cerebellum_fwd.w_pc_dcn[2],
                                                               cerebellum_fwd.w_pc_dcn_vel[0], cerebellum_fwd.w_pc_dcn_vel[1], cerebellum_fwd.w_pc_dcn_vel[2], 
                                                               cerebellum_fwd.output_pc_dcn_pos[0], cerebellum_fwd.output_pc_dcn_pos[1], cerebellum_fwd.output_pc_dcn_pos[2],  
                                                               cerebellum_fwd.output_pc_dcn_vel[0], cerebellum_fwd.output_pc_dcn_vel[1], cerebellum_fwd.output_pc_dcn_vel[2],
                                                               
                                                               #ball_distance.value.data[0], ball_distance.value.data[0], ball_distance.value.data[0],
                                                               #ball_distance.value.data[1], ball_distance.value.data[1], ball_distance.value.data[1],
                                                               cerebellum_fwd.input_mossy[0], cerebellum.input_mossy[0], cerebellum.input_mossy[0],
                                                               cerebellum_fwd.input_mossy[1], cerebellum.input_mossy[1], cerebellum.input_mossy[1],
                                                               
                                                               #cerebellum_fwd.output_x_mossy[0][0],     cerebellum_fwd.output_x_mossy[1][0],     cerebellum_fwd.output_x_mossy[2][0],
                                                               #cerebellum_fwd.output_x_mossy_vel[0][0], cerebellum_fwd.output_x_mossy_vel[1][0], cerebellum_fwd.output_x_mossy_vel[2][0], 
                                                               cerebellum_fwd.w_mf_dcn[0], cerebellum_fwd.w_mf_dcn[1], cerebellum_fwd.w_mf_dcn[2], 
                                                               cerebellum_fwd.w_mf_dcn_vel[0], cerebellum_fwd.w_mf_dcn_vel[1], cerebellum_fwd.w_mf_dcn_vel[2],
                                                               
                                                               cerebellum_fwd.output_C_mossy[0], cerebellum_fwd.output_C_mossy[1], cerebellum_fwd.output_C_mossy[2],
                                                               cerebellum_fwd.output_C_mossy_vel[0], cerebellum_fwd.output_C_mossy_vel[1], cerebellum_fwd.output_C_mossy_vel[2], 
                    
                                                               cerebellum_fwd.w_io_dcn[0], cerebellum_fwd.w_io_dcn[1], cerebellum_fwd.w_io_dcn[2],
                                                               cerebellum_fwd.w_io_dcn_vel[0], cerebellum_fwd.w_io_dcn_vel[1], cerebellum_fwd.w_io_dcn_vel[2], 
                                                               #cerebellum.output_IO_DCN[0], 
                                                               #cerebellum.output_IO_DCN_vel[0],
                                                               delta_error.value.data[0], delta_error.value.data[1], delta_error.value.data[2]
                    
                                                               )
                                    
                if cerebellum_dynamics_on.value == True :
                    cerebellum_dynamics_recorder.record_entry(t, # 22
                                                               #cerebellum.uml_mossy[0].num_rfs[0], 
                                                               #cerebellum.uml_mossy_vel[0].num_rfs[0],
                                                               #cerebellum.uml_pf.num_rfs[0],
                                                               #static_control.error_position[0], static_control.error_position[1],static_control.error_position[2],
                                                               controlcommand.value.data[0], controlcommand.value.data[1], controlcommand.value.data[2],
                                                               cerebellum.output_x_pf[0], cerebellum.output_x_pf[1], cerebellum.output_x_pf[2],
                    
                                                               cerebellum.IO_fbacktorq[0], cerebellum.IO_fbacktorq[1], cerebellum.IO_fbacktorq[2],
                                                               cerebellum.vel_IO_fbacktorq[0], cerebellum.vel_IO_fbacktorq[1], cerebellum.vel_IO_fbacktorq[2],
                                                               cerebellum.w_pf_pc[0],         cerebellum.w_pf_pc[1],    cerebellum.w_pf_pc[2],
                                                               cerebellum.w_pf_pc_vel[0], cerebellum.w_pf_pc_vel[1], cerebellum.w_pf_pc_vel[2],
                                                               cerebellum.output_pc[0],     cerebellum.output_pc[1],     cerebellum.output_pc[2],  
                                                               cerebellum.output_pc_vel[0], cerebellum.output_pc_vel[1], cerebellum.output_pc_vel[2],
                                                               
                                                               cerebellum.w_pc_dcn[0], cerebellum.w_pc_dcn[1], cerebellum.w_pc_dcn[2],
                                                               cerebellum.w_pc_dcn_vel[0], cerebellum.w_pc_dcn_vel[1], cerebellum.w_pc_dcn_vel[2], 
                                                               cerebellum.output_pc_dcn_pos[0], cerebellum.output_pc_dcn_pos[1], cerebellum.output_pc_dcn_pos[2],  
                                                               cerebellum.output_pc_dcn_vel[0], cerebellum.output_pc_dcn_vel[1], cerebellum.output_pc_dcn_vel[2],
                                                               
                                                               cerebellum.input_mossy[0], cerebellum.input_mossy[0], cerebellum.input_mossy[0],
                                                               cerebellum.input_mossy[1], cerebellum.input_mossy[1], cerebellum.input_mossy[1],
                                                               
                                                               #cerebellum_fwd.output_x_mossy[0][0],     cerebellum_fwd.output_x_mossy[1][0],     cerebellum_fwd.output_x_mossy[2][0],
                                                               #cerebellum_fwd.output_x_mossy_vel[0][0], cerebellum_fwd.output_x_mossy_vel[1][0], cerebellum_fwd.output_x_mossy_vel[2][0], 
                                                               cerebellum.w_mf_dcn[0], cerebellum.w_mf_dcn[1], cerebellum.w_mf_dcn[2], 
                                                               cerebellum.w_mf_dcn_vel[0], cerebellum.w_mf_dcn_vel[1], cerebellum.w_mf_dcn_vel[2],
                                                               
                                                               cerebellum.output_C_mossy[0], cerebellum.output_C_mossy[1], cerebellum.output_C_mossy[2],
                                                               cerebellum.output_C_mossy_vel[0], cerebellum.output_C_mossy_vel[1], cerebellum.output_C_mossy_vel[2], 
                    
                                                               cerebellum.w_io_dcn[0], cerebellum.w_io_dcn[1], cerebellum.w_io_dcn[2],
                                                               cerebellum.w_io_dcn_vel[0], cerebellum.w_io_dcn_vel[1], cerebellum.w_io_dcn_vel[2], 
                                                               #cerebellum.output_IO_DCN[0], 
                                                               #cerebellum.output_IO_DCN_vel[0],
                                                               DCNcommand.value.data[0], DCNcommand.value.data[1], DCNcommand.value.data[2]
                    
                                                               )

    except Exception as e:
        clientLogger.info(" --> Controller Exception: "+str(e))
        