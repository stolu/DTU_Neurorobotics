# Imported Python Transfer Function
from std_msgs.msg import Float64
@nrp.MapRobotPublisher("eye_tilt", Topic('/robot/eye_tilt/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("elbow", Topic('/robot/r_elbow/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("neck_pitch", Topic('/robot/neck_pitch/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("neck_yaw", Topic('/robot/neck_yaw/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("neck_roll", Topic('/robot/neck_roll/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("shoulder", Topic('/robot/r_shoulder_pitch/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("shoulder_roll", Topic('/robot/r_shoulder_roll/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("shoulder_yaw", Topic('/robot/r_shoulder_yaw/pos', std_msgs.msg.Float64))

@nrp.MapRobotPublisher("wrist_prosup", Topic('/robot/r_wrist_prosup/pos', std_msgs.msg.Float64))

@nrp.MapRobotPublisher("wrist_yaw", Topic('/robot/r_wrist_yaw/pos', std_msgs.msg.Float64))

@nrp.MapRobotPublisher("wrist_pitch", Topic('/robot/r_wrist_pitch/pos', std_msgs.msg.Float64))

@nrp.MapRobotPublisher("eye_version", Topic('/robot/eye_version/pos', std_msgs.msg.Float64))
@nrp.Neuron2Robot()
def init_pose(t, eye_tilt, elbow, neck_pitch, neck_yaw, neck_roll, shoulder, shoulder_roll, eye_version, wrist_pitch, wrist_yaw, wrist_prosup, shoulder_yaw):
    if t > 2. :
        eye_tilt.send_message(std_msgs.msg.Float64(-0.56))
        eye_version.send_message(std_msgs.msg.Float64(0.))
        neck_pitch.send_message(std_msgs.msg.Float64(-0.4))
        neck_yaw.send_message(std_msgs.msg.Float64(-0.35))
        neck_roll.send_message(std_msgs.msg.Float64(0.1))
        shoulder.send_message(std_msgs.msg.Float64(-.9)) # -1.25
        elbow.send_message(std_msgs.msg.Float64(1.14))
        shoulder_roll.send_message(std_msgs.msg.Float64(.1))
        shoulder_yaw.send_message(std_msgs.msg.Float64(-.1))
        #wrist_pitch.send_message(std_msgs.msg.Float64(.0))
        #wrist_yaw.send_message(std_msgs.msg.Float64(-.14))
        #wrist_prosup.send_message(std_msgs.msg.Float64(-0.2))