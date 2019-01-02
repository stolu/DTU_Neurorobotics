# Imported Python Transfer Function
from gazebo_msgs.msg import ModelState, ModelStates
m = ModelState()
m.model_name = 'green_ball'
# Set orientation RYP axes
m.pose.orientation.x = 0
m.pose.orientation.y = 0
m.pose.orientation.z = 0
m.reference_frame = 'world'
m.pose.position.x = 0.03#241
m.pose.position.y = -0.32#90
m.pose.position.z = 1.0
m.scale.x = m.scale.y = m.scale.z = 1.0
@nrp.MapVariable( "moved", initial_value=False )
@nrp.MapVariable( "ball", initial_value =m )
@nrp.MapRobotPublisher( "ballmover", Topic('/gazebo/set_model_state', ModelState) )
@nrp.MapRobotSubscriber( "ballstate", Topic('/gazebo/model_states', ModelStates) )
@nrp.Neuron2Robot()
def spawn_ball (t, moved, ball, ballmover, ballstate):
    #clientLogger.info(ballstate.value.pose[0].position.z)
    if t > 5 and ballstate.value.pose[0].position.z < 0.1: #not moved.value:
        moved.value = True
        ballmover.send_message(ball.value)
        clientLogger.info("spawn ball")
