# Imported Python Transfer Function
import numpy as np
from numpy import pi as pi
import math
from std_msgs.msg    import Float64, Float64MultiArray, Bool
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import LinkStates
from gazebo_msgs.srv import ApplyJointEffort
from rospy import ServiceProxy, wait_for_service, Duration
import sys
sys.path.append("/home/dtu-neurorobotics2/Documents/robotics-toolbox-python")
#sys.path.append("/home/dtu-neurorobotics2/Documents/LWPR")
#sys.path.append("/usr/local/lib")
sys.path.append("/home/dtu-neurorobotics2/Documents/analog_cerebellum")
import robot
global robot
from norma import*
from LWPRandC_class import MLandC
#from analog_cerebellum import*
# -----------------------------------------------------------------------#
# ** Create robot model for inverse dynamics **
fable_a     = [ 0.,  0.06    ]
fable_alpha = [ 0., -0.5 * pi]
fable_theta = [ 0.,  0.      ]
fable_d 	 = [ 0.04, 0.,   0.4]
fable_m 	 = [ 1000, 0.01, 0.1]
fable_sigma = [ 0, 0, 0] # 0 for a revolute joint, non-zero for prismatic
n_links  = 2
n_joints = 2
jm = [200e-3, 200e-3]
fable_links = []
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
fable = robot.Robot( fable_links, name='fable', manuf='Unimation', comment='00')
joints_name = ['joint_1', 'joint_2']



# -----------------------------------------------------------------------#
# ** LF Controller Ks **


Kp = [ 0.98,  1.9   ] # [ 0.98,  1.9   ][ 0.68,   0.95   ]
Kd = [ 0.01,   0.01  ] # [ 0.04,   0.04  ]  [ 0.004,   0.009  ]
Ki = [ 0.00001, 0.00001] # [ 0.00001, 0.00001]
prev_error = [ 0., 0.]
prev_tau = std_msgs.msg.Float64MultiArray()
prev_tau.data   = [0.,0.]

prev_pos = std_msgs.msg.Float64MultiArray()
prev_pos.data   = [0.,0.]

vel = std_msgs.msg.Float64MultiArray()
vel.data   = [0.,0.]


err = std_msgs.msg.Float64MultiArray()
err.data   = [0.,0.]
errd = std_msgs.msg.Float64MultiArray()
errd.data   = [0.,0.]


# -----------------------------------------------------------------------#
# ** connecting to the ros service to control the robot in torque **
clientLogger.info('Waiting for ROS Service /gazebo/apply_joint_effort')
wait_for_service('/gazebo/apply_joint_effort')
clientLogger.info('Found ROS Service /gazebo/apply_joint_effort')
service_proxy = ServiceProxy('/gazebo/apply_joint_effort', gazebo_msgs.srv.ApplyJointEffort, persistent=True)
wrench_dt = Duration.from_sec(0.08)#0.2

# -----------------------------------------------------------------------#
# ** defining the unite learning machine **

t_cerebellum     = 1 # enable cerebellum 1, disable 0
debug_cerebellum = 0
experiment       = 1

if t_cerebellum == 1 :
    init_d =[.04 , .04]#[.01 , .01]# [.01 , .08]
    init_alpha = [90. , 100.]#[90. , 150.] #[90. , 90.]#[90. , 100.]
    w_g = [0.1, 0.1]
    Beta = 7 * math.pow(10, -2)
    global mlcj_n
    if experiment == 1:
        n_lwpr_in    = 9-3                  # number of LWPR input - Mossy fiber = 1 desired pose + 1 desired vel+ + 1 desired acc + 2 current pose + 2 current vel +  2 current acc --> for each joint considered input both the  acceleration[rad/sec^2], velocity[rad/sec] and the position [rad]
        n_lwpr_out   = 1
        n_ulm_joints = 1
        mlcj_n = []
        for i in range(n_joints):
            mlcj_n.append(MLandC(n_lwpr_in,n_ulm_joints, n_lwpr_out)) # each ULM is specialized on one joint
            mlcj_n[i].model.w_gen = w_g[i]
            mlcj_n[i].model.init_D = init_d[i]*np.eye(n_lwpr_in)
            mlcj_n[i].model.init_alpha = init_alpha[i]*np.ones([n_lwpr_in, n_lwpr_in])
            #mlcj_n[i].model.add_threshold = 0.3
            #mlcj_n[i].model.w_prune = 0.98
            #mlcj_n[i].model.init_lambda  = 0.99
            #mlcj_n[i].model.tau_lambda   = 0.9
            #mlcj_n[i].model.final_lambda = 0.99999
            mlcj_n[i].beta = Beta
    elif experiment == 2:
        n_lwpr_in    = 12                  # number of LWPR input - Mossy fiber = 2 desired pose + 2 desired vel + 2 desired acc + 2 current pose + 2 current vel +  2 current acc --> for each joint considered input both the acceleration[rad/sec^2], velocity[rad/sec] and the position [rad]
        n_lwpr_out   = 2
        n_ulm_joints = 2
        mlcj_n = MLandC(n_lwpr_in,n_ulm_joints, n_lwpr_out)
        mlcj_n.model.w_gen = 0.0083
        mlcj_n.model.w_prune = 0.98


norm_lwpr_in = norm_input(-np.pi*0.5, np.pi*0.5)
norm_lwpr_tau = norm_input(-1.56, 1.56)
max_tau = [Kp[0]*np.pi + Kd[0]*np.pi+ Ki[0]*np.pi, Kp[1]*np.pi + Kd[1]*np.pi+ Ki[1]*np.pi]
LWPRcmd = std_msgs.msg.Float64MultiArray()
LWPRcmd.data   = [0.,0.]
DCNcmd = std_msgs.msg.Float64MultiArray()
DCNcmd.data   = [0.,0.]
LFcmd = std_msgs.msg.Float64MultiArray()
LFcmd.data   = [0.,0.]
cntr_in = std_msgs.msg.Float64MultiArray()
cntr_in.data   = [0.,0.]

# -----------------------------------------------------------------------#
# **  map **
@nrp.MapVariable("debug", 			 initial_value = 0)

# variable related to the lf
@nrp.MapVariable("LF", 			      initial_value = 1)
@nrp.MapVariable("previous_tau",  	 initial_value = prev_tau)
@nrp.MapVariable("K_p", 			 initial_value = Kp)
@nrp.MapVariable("K_d", 			 initial_value = Kd)
@nrp.MapVariable("K_i", 			 initial_value = Ki)

@nrp.MapVariable("error_q", 			 initial_value = err)
@nrp.MapVariable("error_qd", 	      initial_value = errd)

@nrp.MapVariable("previous_pos", 	 initial_value = prev_pos)
@nrp.MapVariable("veloc", 			 initial_value = vel)

# variable related to the effort service
@nrp.MapVariable("proxy", 	 		 initial_value = service_proxy)
@nrp.MapVariable("duration", 		 initial_value = wrench_dt)

# variable related to the robot model
@nrp.MapVariable("fable_robot", 	 initial_value = fable)
@nrp.MapVariable("n_joints", 	 initial_value = n_joints)
@nrp.MapVariable("joints_name", 	 initial_value = joints_name)

# variable related to the cerebellum
@nrp.MapVariable("uml_experiment",    initial_value = experiment)
@nrp.MapVariable("tau_cerebellum",    initial_value = t_cerebellum)
@nrp.MapVariable("norm_pos",          initial_value = norm_lwpr_in )# np.deg2rad(90))
@nrp.MapVariable("norm_in",           initial_value = 0 )
@nrp.MapVariable("norm_in_tau",       initial_value = 0 )
@nrp.MapVariable("norm_tau", 	     initial_value = norm_lwpr_tau)
@nrp.MapVariable("max_tau_lf", 	     initial_value = max_tau)
@nrp.MapVariable("max_pose", 	     initial_value = np.pi*0.5)
@nrp.MapVariable("max_vel", 	          initial_value = np.pi*0.5/0.02)

# variable related to the control commands
@nrp.MapVariable("LWPRcommand",       initial_value = LWPRcmd)
@nrp.MapVariable("DCNcommand", 	      initial_value = DCNcmd)
@nrp.MapVariable("LFcommand", 	      initial_value = LFcmd)
@nrp.MapVariable("controlcommand",    initial_value = cntr_in)


# -----------------------------------------------------------------------#
# ** subscribe to ros topics **

# ** Current Joint states **
@nrp.MapRobotSubscriber("joints_current",      Topic('/joint_states', sensor_msgs.msg.JointState))
# ** Desired Joints trajectory **
@nrp.MapRobotSubscriber("desired_joints_trajectory", Topic('/robot/desired_trajectory/joints',  sensor_msgs.msg.JointState))


# -----------------------------------------------------------------------#
# ** publish on ros topics **
# joint position and velocity error publishers
#@nrp.MapRobotSubscriber("error_q",   Topic('/robot/joint_error/position',     std_msgs.msg.Float64MultiArray))
#@nrp.MapRobotSubscriber("error_qd",  Topic('/robot/joint_error/velocity',     std_msgs.msg.Float64MultiArray))
#@nrp.MapRobotSubscriber("error_qdd", Topic('/robot/joint_error/acceleration', std_msgs.msg.Float64MultiArray))
# comunication with planner node
@nrp.MapRobotPublisher("plan_pub", Topic('/robot/plan', std_msgs.msg.Bool))

# -----------------------------------------------------------------------#
# ** Recording the control input **
@nrp.MapCSVRecorder("command_recorder", filename="command.csv", headers=["time", "tau_inv1", "tau_inv2","tau_lf1","tau_lf2",
                                                                         "dcn1","dcn2","lwpr1", "lwpr2",
                                                                         "control_input1","control_input2"])
@nrp.MapCSVRecorder("joint_recorder", filename="joint_info.csv", headers=["time", "q1","q2","qd1","qd2",  "q1_des","q2_des","qd1_des","qd2_des","qdd1_des","qdd2_des" ,"e_q1","e_q2","e_qd1","e_qd2"])


# -----------------------------------------------------------------------#
# ** "Brain" comunication **
@nrp.Neuron2Robot()
def controller( t, debug, proxy, duration,
                fable_robot, 
                desired_joints_trajectory,
                n_joints, joints_name, joints_current,
                error_q, error_qd,# error_qdd,
                previous_pos,veloc,
                LF, previous_tau, 
                K_p, K_d, K_i,
                command_recorder, joint_recorder,
                plan_pub,
                uml_experiment, tau_cerebellum, LWPRcommand, DCNcommand, LFcommand,# input_lwpr,
                controlcommand,
                norm_pos, norm_tau, max_tau_lf, max_pose, max_vel,
                norm_in, norm_in_tau
                ):
	try:


         # =================================== Inverse Dynamics ===================================

         # ex.: tau = robot.dynamics.rne(robot, q, qd, qdd)
         if LF.value == 1:
              # setting qd and qdd to 0. and 1. respectively gives the Inertia matrix
             tau_inv = robot.dynamics.rne(fable_robot.value, desired_joints_trajectory.value.position,[ 0., 0.], [ 1., 1.])
         else:
             # Inverse dynamics given desired joints states
             tau_inv = robot.dynamics.rne(fable_robot.value, desired_joints_trajectory.value.position, desired_joints_trajectory.value.velocity, desired_joints_trajectory.value.effort)


         # =================================== LF Controller ===================================
         # tau_lf = (pid + previous_tau_lf)*Inertia

         if LF.value == 1:
			for j in range(0, n_joints.value):
					
					error_q.value.data[j]    = desired_joints_trajectory.value.position[j] - joints_current.value.position[j]
					veloc.value.data[j]      = (joints_current.value.position[j] - previous_pos.value.data[j])/0.0085
					
					previous_pos.value.data[j] = joints_current.value.position[j]
					error_qd.value.data[j]   = desired_joints_trajectory.value.velocity[j] - veloc.value.data[j] #joints_current.value.velocity[j]
					
					tau_dummy = ( K_p.value[j]*error_q.value.data[j] + K_d.value[j]*error_qd.value.data[j]  + previous_tau.value.data[j] )*tau_inv[0,j]  #  + K_i.value[j]*error_qdd.value.data[j]

					if norm_in_tau == 0 :
						LFcommand.value.data[j] = tau_dummy
					else:
						LFcommand.value.data[j] = norm_tau.value.get_norm( tau_dummy, -max_tau_lf.value[j]*abs(tau_inv[0,j]) , max_tau_lf.value[j]*abs(tau_inv[0,j]) )

                           # store previous torque lf
					previous_tau.value.data[j] = LFcommand.value.data[j]
			
			if debug.value == 1:
				clientLogger.info("\n tau_inv "+str(tau_inv)+"\n tau_LF "+str(LFcommand.value.data[j] )+"\n previous_tau: "+str(previous_tau.value.data ))
				clientLogger.info("Joint "+str(j)+"\n KP contribution "+str(K_p.value[j]*error_q.value.data[j])+"\n KD contribution "+str(K_d.value[j]*error_qd.value.data[j] ))#+"\n Ki contribution: "+str( K_i.value[j]*error_qdd.value.data[j] ))
         else:
             LFcommand.value.data = [ 1.*tau_inv[0,0], 1.*tau_inv[0,1] ] 


         # =================================== Cerebellum Predict ===================================
         if tau_cerebellum.value == 1:

            input_lwpr = []
            # ==== Experiment 1 ===
            if uml_experiment.value == 1:
                 if t < 0.03:
                     clientLogger.info("\n ** Running the first experiment : each joint has a dedicate UML! ** ")

                 for j in range(0, n_joints.value):

                     if norm_in.value == 1:
                         input_lwpr.append( np.array([  norm_pos.value.get_norm(desired_joints_trajectory.value.position[j], -max_pose.value, max_pose.value ), norm_pos.value.get_norm(desired_joints_trajectory.value.velocity[j], -max_vel.value, max_vel.value ), #desired_joints_trajectory.value.effort[j],
                                                        norm_pos.value.get_norm(desired_joints_trajectory.value.position[0], -max_pose.value, max_pose.value ), norm_pos.value.get_norm(desired_joints_trajectory.value.velocity[0], -max_vel.value, max_vel.value ),            #joints_current.value.effort[0],
                                                        norm_pos.value.get_norm(desired_joints_trajectory.value.position[1], -max_pose.value, max_pose.value ), norm_pos.value.get_norm(desired_joints_trajectory.value.velocity[1], -max_vel.value, max_vel.value ),            #joints_current.value.effort[1]
                                                     ]))
                     else:                     
                         input_lwpr.append(np.array([   desired_joints_trajectory.value.position[j], desired_joints_trajectory.value.velocity[j],
                                                        joints_current.value.position[0],            joints_current.value.velocity[0],           
                                                        joints_current.value.position[1],            joints_current.value.velocity[1],           
                                                    ]))
                     #clientLogger.info("\n input_lwpr[:][j]: "+str( input_lwpr[:][j]) )
                     (LWPRcommand.value.data[j], DCNcommand.value.data[j]) = mlcj_n[j].ML_prediction( input_lwpr[:][j], np.array([LFcommand.value.data[j] ]) )
                     if debug.value == 1:
                         clientLogger.info("\n LWPRcommand: "+str(LWPRcommand.value.data)+"\n DCNcommand: "+str(DCNcommand.value.data))
                 
                 #clientLogger.info("\n input_lwpr joint"+str(j)+" : "+str(np.array(input_lwpr[:][j]) ))
            # ==== Experiment 2 ===
            elif uml_experiment.value == 2:
                if t < 0.03:
                     clientLogger.info("\n ** Running the second experiment : one UML for all the joints! ** ")
                     clientLogger.info('---> mlcj_n '+str(mlcj_n.model))
                input_lwpr = np.array([ desired_joints_trajectory.value.position[0], desired_joints_trajectory.value.velocity[0], desired_joints_trajectory.value.effort[0],
                                        desired_joints_trajectory.value.position[1], desired_joints_trajectory.value.velocity[1], desired_joints_trajectory.value.effort[1],
                                        joints_current.value.position[0],            joints_current.value.velocity[0],            joints_current.value.effort[0],
                                        joints_current.value.position[1],            joints_current.value.velocity[1],            joints_current.value.effort[1]
                                        ])
                (LWPRcommand.value.data[:], DCNcommand.value.data[:]) = mlcj_n.ML_prediction(input_lwpr, np.array([LFcommand.value.data[0], LFcommand.value.data[1] ] ) )
                if debug.value == 1:
                    clientLogger.info("\n LWPRcommand: "+str(LWPRcommand.value.data)+"\n DCNcommand: "+str(DCNcommand.value.data))
                
            if debug.value == 1:
                clientLogger.info("\n input_lwpr: "+str(input_lwpr))
         else:
             LWPRcommand.value.data = [0., 0.]
             DCNcommand.value.data = [0., 0.]

         # =================================== Control Input ===================================

         for j in range(0, n_joints.value):
             
             if tau_cerebellum.value == 1:
                 # ==== Experiment 1  ===
                 if uml_experiment.value == 1:
                     controlcommand.value.data[j] = LFcommand.value.data[j] + LWPRcommand.value.data[j][0] + DCNcommand.value.data[j][0]
                     #clientLogger.info("\n control_input joint "+str(j)+"  "+str(control_input[j])+"\n  LWPRcommand.value.data[j][0] "+str(  LWPRcommand.value.data[j][0])+"\n  DCNcommand.value.data[j][0] "+str(  DCNcommand.value.data[j][0]))
                 elif uml_experiment.value == 2:
                     controlcommand.value.data[j] =  LFcommand.value.data[j] + LWPRcommand.value.data[j] + DCNcommand.value.data[j] 
             else:
                 controlcommand.value.data[j] =  LFcommand.value.data[j]
         if debug.value == 1:
             clientLogger.info("\n control_input: "+str(controlcommand.value.data))

         # =================================== Cerebellum Update =================================== 

         if tau_cerebellum.value == 1:
             # ==== Experiment 1  ===
             if uml_experiment.value == 1:  
                 for j in range(0, n_joints.value):
                     mlcj_n[j].ML_update( input_lwpr[:][j] , controlcommand.value.data[j])
                     mlcj_n[j].ML_rfs() 
                     #clientLogger.info(" receptive fields for joint "+str(j)+" : "+str(mlcj_n[j].ML_rfs() ))
             # ==== Experiment 2  ===
             elif uml_experiment.value == 2:
                 mlcj_n.ML_update(input_lwpr, controlcommand.value.data[:])
                 mlcj_n.ML_rfs() 

         # =================================== Send the Control Input ===================================     

         # ** Sends the control input (effort) to the plant **
         for j in range(0, n_joints.value):
             proxy.value.call(joints_name.value[j], controlcommand.value.data[j],  None, duration.value)
             if debug.value == 1:
                 clientLogger.info("control_input sent!")

         # =================================== Comunication with the planner ===================================       

         plan_pub.send_message(True)

         # =================================== Record the Data ===================================
         # ** record ** --- t,"tau_inv","tau_lf","dcn","lwpr","control_input"
         if tau_cerebellum.value == 1:
             if uml_experiment.value == 1:         
                 command_recorder.record_entry(t, tau_inv[0,0],tau_inv[0,1], 
                                           LFcommand.value.data[0], LFcommand.value.data[1], 
                                           DCNcommand.value.data[0][0], DCNcommand.value.data[1][0], 
                                           LWPRcommand.value.data[0][0], LWPRcommand.value.data[1][0], 
                                           controlcommand.value.data[0], controlcommand.value.data[1])
             else:
                  command_recorder.record_entry(t, tau_inv[0,0],tau_inv[0,1], 
                                           LFcommand.value.data[0], LFcommand.value.data[1], 
                                           DCNcommand.value.data[0], DCNcommand.value.data[1], 
                                           LWPRcommand.value.data[0], LWPRcommand.value.data[1], 
                                           controlcommand.value.data[0], controlcommand.value.data[1] )
         else:
             command_recorder.record_entry(t, tau_inv[0,0],tau_inv[0,1], 
                                       LFcommand.value.data[0], LFcommand.value.data[1], 
                                       0., 0.,
                                       0., 0., 
                                       controlcommand.value.data[0], controlcommand.value.data[1] )
         joint_recorder.record_entry(t, joints_current.value.position[0], joints_current.value.position[1],veloc.value.data[0], veloc.value.data[1] , #joints_current.value.velocity[0],joints_current.value.velocity[1],#current_acceleration.value.data[0],current_acceleration.value.data[1],
                              desired_joints_trajectory.value.position[0], desired_joints_trajectory.value.position[1],desired_joints_trajectory.value.velocity[0],desired_joints_trajectory.value.velocity[1],desired_joints_trajectory.value.effort[0],desired_joints_trajectory.value.effort[1],
                              error_q.value.data[0],error_q.value.data[1],error_qd.value.data[0],error_qd.value.data[1])#,error_qdd.value.data[0],error_qdd.value.data[1])
	except Exception as e:
		clientLogger.info(" --> Controller Exception: "+str(e))
