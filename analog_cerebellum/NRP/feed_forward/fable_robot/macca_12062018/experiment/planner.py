import numpy as np
from std_msgs.msg    import Float64,Float64MultiArray, Bool
from sensor_msgs.msg import JointState
from numpy           import pi as pi


j_des  = sensor_msgs.msg.JointState()
j_des.effort    = [0.0]*2
j_des.velocity  = [0.0]*2
j_des.position = [0.0]*2
ee_des = std_msgs.msg.Float64MultiArray()
ee_des.data = [0.0]*2

# -----------------------------------------------------------------------#
# **  map **

@nrp.MapVariable("debug_traj", 			 initial_value = 0)

@nrp.MapVariable("desired_joints", 		 initial_value = j_des)
@nrp.MapVariable("desired_end_effector",    initial_value = ee_des)

@nrp.MapVariable("can_plan", 			 initial_value = True)
@nrp.MapVariable("time", 			       initial_value = 0.)


# ** Input the trajectory type: 1) infinity loop; 2) circle; 3) fixed
@nrp.MapVariable("trajectory_type", 	 initial_value = 1)

# ** Input the fixed reference
@nrp.MapVariable("fixed_ref_x", initial_value = np.deg2rad(45.))
@nrp.MapVariable("fixed_ref_y", initial_value = np.deg2rad(45.))


# -----------------------------------------------------------------------#
# ** publish on ros topics **

# ** Desired end-effector trajectory **
@nrp.MapRobotPublisher("desired_ee_trajectory_pub",     Topic('/robot/desired_trajectory/end_effector', std_msgs.msg.Float64MultiArray))
# ** Desired Joints trajectory **
@nrp.MapRobotPublisher("desired_joints_trajectory_pub", Topic('/robot/desired_trajectory/joints',       sensor_msgs.msg.JointState))

@nrp.MapRobotPublisher("plan_pub", Topic('/robot/plan', std_msgs.msg.Bool))
@nrp.MapRobotSubscriber("plan_sub", Topic('/robot/plan', std_msgs.msg.Bool))

@nrp.Robot2Neuron()

def planner(t, debug_traj, trajectory_type, 
            desired_end_effector, desired_joints, 
            fixed_ref_x, fixed_ref_y, 
            plan_sub, plan_pub, can_plan, time,
            desired_ee_trajectory_pub, desired_joints_trajectory_pub
            ):
	
    # ** oscillator parameters **
    amplitude = np.deg2rad(40.)
    period    = 10.
    frequency = 2.*np.pi/period
    dt = 0.0085#0.02#0.04#.009#0.01
    

    try:
        can_plan.value = plan_sub.value.data
    except Exception as e:
            clientLogger.info(" --> Planner Exception: "+str(e))   
    if can_plan.value == True:
        if trajectory_type.value == 1:
            if t < 0.01:
                 clientLogger.info("trajectory : infinity loop with amplitude "+str(np.rad2deg(amplitude))+" deg" )
            phase = np.rad2deg(0.)
            scale = 2 / (3 - np.cos( 2* (frequency*time.value + phase) ))


            desired_end_effector.value.data[0]   = scale * np.cos(np.deg2rad(frequency*time.value + phase))
            desired_end_effector.value.data[1]   = scale * np.sin(np.deg2rad(frequency*time.value + phase))/ 2
            
            phase_j2 = np.deg2rad(90)
            desired_joints.value.effort[0]   =   -4*np.power(np.pi, 2) *amplitude * np.sin(2 * np.pi *time.value)#"accelleration"
            desired_joints.value.velocity[0] =   2*np.pi*amplitude * np.cos(2 * np.pi *time.value) 
            desired_joints.value.position[0] =   amplitude * np.sin(2 * np.pi *time.value) 
            		
            desired_joints.value.effort[1]   =    -16*np.power(np.pi, 2) *amplitude * np.cos(4* np.pi *time.value + phase_j2)#"accelleration"
            desired_joints.value.velocity[1] =  -4*np.pi*amplitude * np.sin(4 * np.pi *time.value + phase_j2) 
            desired_joints.value.position[1] =   amplitude * np.cos( 4* np.pi *time.value + phase_j2)
		

        elif trajectory_type.value == 2:
            if t < 0.01:
                clientLogger.info("trajectory : circle with amplitude "+str(np.rad2deg(amplitude))+" deg")
            phase_j2 = np.deg2rad(90)
            		
            desired_end_effector.value.data[0]   =   amplitude*np.cos(np.deg2rad(frequency*time.value )) 
            desired_end_effector.value.data[1]   =   amplitude*np.cos(np.deg2rad(frequency*time.value + phase_j2)) 
            
            desired_joints.value.effort[0]   =  -4*np.power(np.pi, 2) *amplitude * np.cos(2 * np.pi *time.value) #"accelleration"
            desired_joints.value.velocity[0] =  -2*np.pi*amplitude * np.sin(2 * np.pi *time.value) 
            desired_joints.value.position[0] =   amplitude * np.cos(2 * np.pi *time.value) 
            
            desired_joints.value.effort[1]   =	-4*np.power(np.pi, 2) *amplitude * np.cos(2 * np.pi *time.value + phase_j2) #"accelleration"
            desired_joints.value.velocity[1] =  -2*np.pi*amplitude * np.sin(2 * np.pi *time.value + phase_j2) 
            desired_joints.value.position[1] =   amplitude * np.cos(2 * np.pi *time.value + phase_j2)
		

        elif trajectory_type.value == 3:
            if t < 0.01:
                clientLogger.info("trajectory :  steady reference with amplitude ("+str(fixed_ref_x.value)+","+str(fixed_ref_x.value)+")")

            desired_end_effector.value.data[0]   = amplitude*np.cos(fixed_ref_x.value) 
            desired_end_effector.value.data[1]   = amplitude*np.sin(fixed_ref_y.value) 
            
            
            desired_joints.value.effort[0]   = 0.
            desired_joints.value.velocity[0] = 0.
            desired_joints.value.position[0] = fixed_ref_x.value
            
            desired_joints.value.effort[1]   = 0.
            desired_joints.value.velocity[1] = 0. 
            desired_joints.value.position[1] = fixed_ref_y.value

        else:
            clientLogger.info("** No trajectory type selected! **")


        if debug_traj.value == 1:
            clientLogger.info("desired end-effector: "+str(desired_end_effector.value.data))
            clientLogger.info("desired joints: "+str(desired_joints.value.position))
            
        try:
            # ** Publish desired Joints value **
            
            desired_joints_trajectory_pub.send_message(desired_joints.value)
            # ** Publish desired End-effector position **
            desired_ee_trajectory_pub.send_message(desired_end_effector.value)
            can_plan.value = False
            
            
            
            plan_pub.send_message(can_plan.value)
            time.value = time.value + dt
        except Exception as e:
            clientLogger.info(" --> Planner Exception: "+str(e))
		