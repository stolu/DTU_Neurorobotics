from std_msgs.msg import Float64
import numpy as np

@nrp.MapRobotPublisher("motor1", Topic('/robot/motor_neck/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("motor2", Topic('/robot/neck_motor2/cmd_pos', std_msgs.msg.Float64))

@nrp.MapVariable("init_motor1", initial_value= np.deg2rad(0.))
@nrp.MapVariable("init_motor2", initial_value= np.deg2rad(0.))
@nrp.Neuron2Robot()

def init(t, motor1, init_motor1, motor2, init_motor2):
	if t < 0.1:
		motor1.send_message(std_msgs.msg.Float64(init_motor1.value))
		motor2.send_message(std_msgs.msg.Float64(init_motor2.value))

