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


LWPRcmd = std_msgs.msg.Float64MultiArray()
LWPRcmd.data   = [0. for i in range( 0, njoints)]
DCNcmd = std_msgs.msg.Float64MultiArray()
DCNcmd.data   = [0. for i in range( 0, njoints)]

cntr_in = std_msgs.msg.Float64MultiArray()
cntr_in.data   = [0. for i in range( 0, njoints)]

des_init = sensor_msgs.msg.JointState()
des_init.position = [0. for i in range( 0, njoints)]
des_init.velocity = [0. for i in range( 0, njoints)]
des_init.effort = [0. for i in range( 0, njoints)]

prev_joints = sensor_msgs.msg.JointState()
prev_joints.position = [0. for i in range( 0, njoints)]
prev_joints.velocity = [0. for i in range( 0, njoints)]
prev_joints.effort = [0. for i in range( 0, njoints)]

delta_err        = std_msgs.msg.Float64MultiArray()
delta_err.data   = [0. for i in range( 0, njoints)]
#des_init = std_msgs.msg.Float64MultiArray()
#des_init.data =[ -0.09, -0.14, 0.] #[0. , 0.09, 0.78]
# -0.000105763160133 prosup , 0.436335761307 yaw, 0.799917785189 elbow
# w yaw - 0.14, w pitch 0.
pitch_range = [-1.13446, 0.174533] # effort 0.65
prosup_range  = [ -0.872665, 0.872665] # effort 0.45
yaw_range   = [-0.436332, 0.436332]
joint_rng = [ 0.872665 - 0.7, 0.436332 - 0.3, 0.13446 - 0.1]
traj_phase = [ 0.5*np.pi ,0.5*np.pi , 0.]


# Start time for the vision-based controller
start_time = 5

global cerebellum, cerebellum_fwd
t_cerebellum     = 1 #enable cerebellum 1, disable 0
debug_cerebellum = 0
experiment       = 1
test_lwpr = 1

if t_cerebellum == 1 :
    #               [ [ j_1_pos, ... , j_n_pos], [ j_1_vel, ... , j_n_vel] ]
    init_d_mf     = [ [  0.5  ,  0.5  ,  0.5  ], [  0.05  ,  0.05  ,  0.05  ] ]
    init_alpha_mf = [ [ 190.   , 190.   , 190.   ], [ 90.   , 90.   , 90.   ] ]
    w_g_mf        = [ [  0.4  ,  0.4  ,  0.4  ], [  0.4  ,  0.4  ,  0.4  ] ]
    w_pru_mf      = [ [  0.8 ,  0.8 ,  0.8 ], [  0.98 ,  0.98 ,  0.98 ] ]
    Beta = 7.* math.pow(10, -3) #2
    
    n_input_lwpr_pf        = 13       # number of LWPR input - Mossy fiber = 1 desired pose + 1 desired vel+ + 1 desired acc + 2 current pose + 2 current vel +  2 current acc --> for each joint considered input both the  acceleration[rad/sec^2], velocity[rad/sec] and the position [rad]
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
    cerebellum.max_teach_pos =  [ abs(prosup_range[0]- prosup_range[1]), abs(yaw_range[0]- yaw_range[1]), abs(pitch_range[0]- pitch_range[1])]#[ max_dist_pos , max_dist_pos, max_dist_pos]
    cerebellum.min_teach_vel =  [0.,0., 0.]
    cerebellum.max_teach_vel =  [ v/0.02 for v in cerebellum.max_teach_pos ]#[ max_dist_vel, max_dist_vel, max_dist_vel]
    cerebellum.signal_normalization = 1
    cerebellum.tau_norm_sign = 0
    cerebellum.min_signal_pos    = [0. for i in range( 0, njoints)]       
    cerebellum.max_signal_pos    = [(Kp_init[i]*0.4 + 20.*Kd_init[i] + 0.4*Ki_init[i]) for i in range( 0, njoints)]
    cerebellum.min_signal_vel    = [0. for i in range( 0, njoints)]       
    cerebellum.max_signal_vel    = [(Kp_init[i]*0.4 + 20.*Kd_init[i] + 0.4*Ki_init[i]) for i in range( 0, njoints)]
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
    cerebellum.init_D_pf     = 0.3
    cerebellum.init_alpha_pf = 90.
    cerebellum.w_gen_pf      = 0.4
    cerebellum.w_prune_pf    = 0.95
    cerebellum.diag_only_pf  = bool(1)  #1
    cerebellum.update_D_pf   = bool(0) #0
    cerebellum.meta_pf       = bool(0)
    
    # *** Plasticity ***
    #exc
    cerebellum.ltpPF_PC_max = 1. * 10**(-4) # -4
    cerebellum.ltdPF_PC_max = 1. * 10**(-4) # -5
    #inh
    cerebellum.ltpPC_DCN_max = 1. * 10**(-3) #-2  # 4
    cerebellum.ltdPC_DCN_max = 1. * 10**(-3) #-3  # 3
    #exc
    cerebellum.ltpMF_DCN_max = 1. * 10**(-3) #-3
    cerebellum.ltdMF_DCN_max = 1. * 10**(-3) #-2
    #exc
    cerebellum.ltpIO_DCN_max = 1. * 10**(-4) #-4
    cerebellum.ltdIO_DCN_max = 1. * 10**(-5) #-3
     
    #self.alpha = 1
    cerebellum.alphaPF_PC_pos  = 2.  #self.ltd_max / self.ltp_max 50, 7
    cerebellum.alphaPF_PC_vel  = 2.    
    cerebellum.alphaPC_DCN     = 2. # self.ltd_PC_DCN_max / self.ltp_PC_DCN_max
    cerebellum.alphaMF_DCN     = 2. # self.ltd_PC_DCN_max / self.ltp_PC_DCN_max
    cerebellum.alphaIO_DCN     = 10.
    cerebellum.IO_on = False
    cerebellum.PC_on = True
    cerebellum.MF_on = True
    
    
    # ******************************************************  forward cerebellum
        #               [ [ j_1_pos, ... , j_n_pos], [ j_1_vel, ... , j_n_vel] ]
    init_d_mf     = [ [  0.5  ,  0.5  ,  0.5  ], [  0.5  ,  0.5  ,  0.5  ] ]
    init_alpha_mf = [ [ 90.   , 90.   , 90.   ], [ 90.   , 90.   , 90.   ] ]
    w_g_mf        = [ [  0.4  ,  0.4  ,  0.4  ], [  0.4  ,  0.4  ,  0.4  ] ]
    w_pru_mf      = [ [  0.98 ,  0.98 ,  0.98 ], [  0.98 ,  0.98 ,  0.98 ] ]
    Beta = 7.* math.pow(10, -3) #2
    
    n_input_lwpr_pf        = 13       # number of LWPR input - Mossy fiber = 1 desired pose + 1 desired vel+ + 1 desired acc + 2 current pose + 2 current vel +  2 current acc --> for each joint considered input both the  acceleration[rad/sec^2], velocity[rad/sec] and the position [rad]
    n_input_lwpr_mossy     = [ 4, 4,4] # per each joint : joint_pose , error ball pose x,  error ball pose y,
    n_input_lwpr_mossy_vel = [ 4, 4,4]
    
    n_lwpr_out   = 1
    n_ulm_joints = 1

    cerebellum_fwd = Cerebellum( njoints , n_lwpr_out , n_input_lwpr_mossy , n_input_lwpr_mossy_vel, n_input_lwpr_pf)
    cerebellum_fwd.name = " Cerebellum foward "
    # ********************** Cerebellum Parameters **********************
    max_dist_pos = 0.5#0.5#np.sqrt(2.*(150./300.)**2.)
    max_dist_vel = 25.#5.#max_dist_pos#np.sqrt(2.*(150./(0.02*300))**2.)    
    # Teaching signal constraints
    cerebellum_fwd.min_teach_pos =  [0., 0., 0.]
    cerebellum_fwd.max_teach_pos =  [ max_dist_pos , max_dist_pos, max_dist_pos]
    cerebellum_fwd.min_teach_vel =  [0.,0., 0.]
    cerebellum_fwd.max_teach_vel =  [ max_dist_vel, max_dist_vel, max_dist_vel]
    cerebellum_fwd.signal_normalization = 1
    cerebellum_fwd.tau_norm_sign = 0
    
    cerebellum_fwd.min_signal_pos    = [0. for i in range( 0, njoints)]       
    cerebellum_fwd.max_signal_pos    = [0.4 for i in range( 0,njoints)]
    cerebellum_fwd.min_signal_vel    = [0. for i in range( 0, njoints)]       
    cerebellum_fwd.max_signal_vel    = [20. for i in range( 0, njoints)]
    
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
    cerebellum_fwd.init_D_pf     = 0.5
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
    cerebellum_fwd.ltpPC_DCN_max = 1. * 10**(-3) #-2  # 4
    cerebellum_fwd.ltdPC_DCN_max = 1. * 10**(-3) #-3  # 3
    #exc
    cerebellum_fwd.ltpMF_DCN_max = 1. * 10**(-3) #-3
    cerebellum_fwd.ltdMF_DCN_max = 1. * 10**(-3) #-2
    #exc
    cerebellum_fwd.ltpIO_DCN_max = 1. * 10**(-5) #-4
    cerebellum_fwd.ltdIO_DCN_max = 1. * 10**(-5) #-3
     
    #self.alpha = 1
    cerebellum_fwd.alphaPF_PC_pos  = 32.  #self.ltd_max / self.ltp_max 50, 7
    cerebellum_fwd.alphaPF_PC_vel  = 20.    
    cerebellum_fwd.alphaPC_DCN     = 10. # self.ltd_PC_DCN_max / self.ltp_PC_DCN_max
    cerebellum_fwd.alphaMF_DCN     = 2. # self.ltd_PC_DCN_max / self.ltp_PC_DCN_max
    cerebellum_fwd.alphaIO_DCN     = 2.
    
    cerebellum_fwd.IO_on = False
    cerebellum_fwd.PC_on = True
    cerebellum_fwd.MF_on = True


cerebellum.create_models()
cerebellum_fwd.create_models()
clientLogger.info("\n The Cerebellum is Initialized ")
t_cerebellum = 30.
t_cerebellum_predict = t_cerebellum + 10.
t_cerebellum_update = t_cerebellum_predict + 120.
# robot parameters
@nrp.MapVariable("n_joints",            initial_value = njoints)
@nrp.MapVariable("joints_idx",          initial_value = joints_index)
@nrp.MapVariable("current_joints",      initial_value = curr_joints)
@nrp.MapVariable("previous_joints",      initial_value = prev_joints)
@nrp.MapVariable("joints_range",        initial_value = joint_rng)
@nrp.MapVariable("trajectory_phase",    initial_value = traj_phase)
# variable related to the control commands


@nrp.MapVariable("LWPRcommand",         initial_value  = LWPRcmd)
@nrp.MapVariable("DCNcommand",          initial_value  = DCNcmd)
@nrp.MapVariable("controlcommand",      initial_value  = cntr_in)

@nrp.MapVariable("desired_init",        initial_value  = des_init)

@nrp.MapVariable("delta_error",      initial_value = delta_err)
# time parameters
@nrp.MapVariable("starting_time",       initial_value = start_time)
@nrp.MapVariable("LFdebug",             initial_value = False)
@nrp.MapVariable("init_pose",           initial_value = True)

@nrp.MapVariable("cerebellum_debug",             initial_value = False)
@nrp.MapVariable("cerebellum_on",          initial_value = True)
@nrp.MapVariable("cerebellum_dynamics_on", initial_value = False)

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
@nrp.MapCSVRecorder("cerebellum_recorder_j1", filename="cerebellum_dynamics.csv", headers=[ "time",
                                                                                   #"rfs_mossy_pos",
                                                                                   #"rfs_mossy_vel",
                                                                                   #"rfs_pf",
                                                                                   #"output_pf",
                                                                                   #"IO_pos",
                                                                                   #"IO_vel",
                                                                                   #"w_pf_pc_pos",
                                                                                   #"w_pf_pc_vel",
                                                                                   #"output_pc" ,
                                                                                   #"output_pc_vel" ,
                                                                                   #"w_pc_dcn_pos",
                                                                                   #"w_pc_dcn_vel",
                                                                                   #"output_PC-DCN_pos" ,
                                                                                   #"output_PC-DCN_vel" ,
                                                                                   #"w_mf_dcn_pos" ,
                                                                                   #"w_mf_dcn_vel" ,
                                                                                   #"output_mossy_pos",
                                                                                   #"output_mossy_vel",
                                                                                   #"output_mossy-DCN_pos" ,
                                                                                   #"output_mossy-DCN_vel"
                                                                                   #"w_io_dcn_pos" ,
                                                                                   #"w_io_dcn_vel" ,
                                                                                   #"output_IO-DCN_pos",
                                                                                   #"output_IO-DCN_vel",
                                                                                   "output_DCN" 
                                                                                   ])

                                                                                    
def balance_control(t, 
                    starting_time, LFdebug, init_pose,
                    #proxy, duration,
                    n_joints,joints_idx, current_joints, joints, previous_joints, 
                    LWPRcommand, DCNcommand, controlcommand,
                    command_publisher,
                    desired_init, joints_range, trajectory_phase,
                    cerebellum_on, cerebellum_dynamics_on,
                    time_cerebellum, time_cerebellum_predict, time_cerebellum_update, cerebellum_debug,
                    ball_error, ball_distance,
                    command_recorder, joint_recorder, cerebellum_recorder_j1, cerebellum_kinematics_recorder,
                    delta_error
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
                
            
            
                
            # =================================== Cerebellum Storing sensorial information ===================================
            
            if cerebellum_on.value == 1 and t > time_cerebellum.value :
                cerebellum.input_pf = [   current_joints.value.position[0], current_joints.value.position[1], current_joints.value.position[2],
                                                                  current_joints.value.velocity[0], current_joints.value.velocity[1], current_joints.value.velocity[2], 
                                                                    desired_init.value.velocity[0],   desired_init.value.velocity[1],   desired_init.value.velocity[2],
                                                                      ball_error.value.position[0], ball_error.value.position[1],
                                                                      ball_error.value.velocity[0], ball_error.value.velocity[1] ]
                cerebellum_fwd.input_pf = [   current_joints.value.position[0], current_joints.value.position[1], current_joints.value.position[2],
                                              current_joints.value.velocity[0], current_joints.value.velocity[1], current_joints.value.velocity[2], 
                                                desired_init.value.velocity[0],   desired_init.value.velocity[1],   desired_init.value.velocity[2],
                                                  ball_error.value.position[0], ball_error.value.position[1],
                                                  ball_error.value.velocity[0], ball_error.value.velocity[1] 
                                          ]
                                              
                if cerebellum_debug.value == True:
                    clientLogger.info("\n input parallel fibers :\n"+str(cerebellum.input_pf))
                    
                for j in range(0, n_joints.value) :
                    cerebellum_fwd.input_mossy[j]     = [       ball_error.value.position[0], ball_error.value.position[1], 
                                                            current_joints.value.position[j],
                                                              desired_init.value.position[j]  
                                                        ]
                    cerebellum_fwd.input_mossy_vel[j] = [       ball_error.value.velocity[0], ball_error.value.velocity[1],
                                                            current_joints.value.velocity[j],
                                                              desired_init.value.velocity[j]   
                                                        ]
                    cerebellum.input_mossy[j]         = [       ball_error.value.position[0], ball_error.value.position[1], 
                                                                 ball_distance.value.data[0],
                                                            current_joints.value.position[j],
                                                              desired_init.value.position[j]  
                                                        ]
                                                                       

                    cerebellum.input_mossy_vel[j] = [       ball_error.value.velocity[0], ball_error.value.velocity[1],
                                                             ball_distance.value.data[0],
                                                        current_joints.value.velocity[j],
                                                          desired_init.value.velocity[j]   
                                                    ]                           
                    if cerebellum_debug.value == True:
                        clientLogger.info("\n input mossy joint "+str(j)+" :\n position: "+str(cerebellum.input_mossy[j]))
                        clientLogger.info("\n velocity: "+str(cerebellum.input_mossy_vel[j]))
                
                # =================================== Cerebellum Update  ===================================
                if t <= time_cerebellum_update.value :
                    if cerebellum_dynamics_on.value == True:
                        cerebellum.update_models( controlcommand.value.data, controlcommand.value.data)
    
                    cerebellum_fwd.update_models( static_control.error_position, static_control.error_position)
                # =================================== Cerebellum Foward Model Predict ===================================
                
                if t > time_cerebellum_predict.value:
                    #DCNcommand.value.data  = cerebellum.prediction( static_control.PID, [ ball_distance.value.data[0] for j in range(0, n_joints.value)], [ ball_distance.value.data[0] for j in range(0, n_joints.value) ] )# mean_tau.value.data )
                    
                    delta_error.value.data = cerebellum_fwd.prediction( static_control.error_position, [ abs(ball_distance.value.data[0]) for j in range(0, n_joints.value)], [ abs(ball_distance.value.data[0]) for j in range(0, n_joints.value) ] )                    
                    if cerebellum_debug.value == True:
                        clientLogger.info("\n delta error "+str(delta_error.value.data))


            # =================================== Static Control Command ===================================

            static_control.control( desired_init.value.velocity, current_joints.value.velocity, delta_error.value.data, t)
            if LFdebug.value == True:
                clientLogger.info("\n static_control.PID "+str(static_control.PID))
            
            
            # =================================== Cerebellum Inverse Model Predict ===================================
            
            if cerebellum_dynamics_on.value == True and t > time_cerebellum_predict.value:                   
                DCNcommand.value.data  = cerebellum.prediction( static_control.PID, [ abs(ball_distance.value.data[0]) for j in range(0, n_joints.value)], [ abs(ball_distance.value.data[1]) for j in range(0, n_joints.value) ] ) #static_control.error_position, static_control.error_velocity )
                
                LWPRcommand.value.data = cerebellum.output_x_pf  
                if cerebellum_debug.value == True:
                     clientLogger.info("\n DCN output : "+str(DCNcommand.value.data))
                     clientLogger.info("\n Parellel fibers output : "+str(LWPRcommand.value.data))
            
            # =================================== Control Command ===================================

            for idx in range( init_joint_idx , n_joints.value):
                controlcommand.value.data[idx] = static_control.PID[idx]+ DCNcommand.value.data[idx]
            if LFdebug.value == True:
                         clientLogger.info("\n control output : "+str(controlcommand.value.data))  
            command_publisher.send_message( controlcommand.value )
            
            


            # =================================== Record  ===================================

            joint_recorder.record_entry(t,       desired_init.value.position[0], current_joints.value.position[0], desired_init.value.velocity[0], current_joints.value.velocity[0], desired_init.value.effort[0],  current_joints.value.effort[0],
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
                cerebellum_kinematics_recorder.record_entry(t, # 22
                                           #cerebellum.uml_mossy[0].num_rfs[0], 
                                           #cerebellum.uml_mossy_vel[0].num_rfs[0],
                                           #cerebellum.uml_pf.num_rfs[0],
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
                                           
                                           cerebellum_fwd.output_x_mossy[0][0],     cerebellum_fwd.output_x_mossy[1][0],     cerebellum_fwd.output_x_mossy[2][0],
                                           cerebellum_fwd.output_x_mossy_vel[0][0], cerebellum_fwd.output_x_mossy_vel[1][0], cerebellum_fwd.output_x_mossy_vel[2][0], 
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
                
                cerebellum_recorder_j1.record_entry(t, # 22
                                           #cerebellum.uml_mossy[0].num_rfs[0], 
                                           #cerebellum.uml_mossy_vel[0].num_rfs[0],
                                           #cerebellum.uml_pf.num_rfs[0],
                                           #cerebellum.output_x_pf[0], cerebellum.output_x_pf[0], cerebellum.output_x_pf[0],

                                           #cerebellum.IO_fbacktorq[0], 
                                           #cerebellum.vel_IO_fbacktorq[0], 
                                           #cerebellum.w_pf_pc[0],
                                           #cerebellum.w_pf_pc_vel[0],
                                           #cerebellum.output_pc[0], 
                                           #cerebellum.output_pc_vel[0],
                                           
                                           #cerebellum.w_pc_dcn[0], 
                                           #cerebellum.w_pc_dcn_vel[0], 
                                           #cerebellum.output_pc_dcn[0], 
                                           #cerebellum.output_pc_dcn_vel[0], 

                                           #cerebellum.w_mf_dcn[0], 
                                           #cerebellum.w_mf_dcn_vel[0],
                                           #cerebellum.output_x_mossy[0][0]  , 
                                           #cerebellum.output_x_mossy_vel[0][0]  , 
                                           #cerebellum.output_C_mossy[0], 
                                           #cerebellum.output_C_mossy_vel[0], 

                                           #cerebellum.w_io_dcn[0],
                                           #cerebellum.w_io_dcn_vel[0], 
                                           #cerebellum.output_IO_DCN[0], 
                                           #cerebellum.output_IO_DCN_vel[0],
                                           cerebellum.output_DCN[0]

                                           )

    except Exception as e:
        clientLogger.info(" --> Controller Exception: "+str(e))
        