from rospy import ServiceProxy, wait_for_service, Duration
clientLogger.info('Waiting for ROS Service /gazebo/apply_joint_effort')
wait_for_service('/gazebo/apply_joint_effort')
clientLogger.info('Found ROS Service /gazebo/apply_joint_effort')
service_proxy = ServiceProxy('/gazebo/apply_joint_effort', gazebo_msgs.srv.ApplyJointEffort, persistent=True)
wrench_dt = Duration.from_sec(0.5)#0.08

joints_index = ["r_wrist_prosup","r_wrist_yaw", "r_wrist_pitch"]
@nrp.MapVariable("joints_idx",          initial_value = joints_index)
@nrp.MapRobotSubscriber("pitch_sub", Topic('/effort_command', std_msgs.msg.Float64MultiArray ))

# variable related to the effort service
@nrp.MapVariable("proxy_pitch",               initial_value = service_proxy)
@nrp.MapVariable("duration",            initial_value = wrench_dt)

@nrp.Robot2Neuron()
def pitch_effort (t, pitch_sub, joints_idx, proxy_pitch, duration):
    try:
        if t >3.:
            proxy_pitch.value.call( joints_idx.value[2] , pitch_sub.value.data[2],  None, duration.value)
            #clientLogger.info("prosup_sub "+str(prosup_sub.value.data[0]) )
    except Exception as e:
        clientLogger.info(" --> effort pitch Exception: "+str(e))
        