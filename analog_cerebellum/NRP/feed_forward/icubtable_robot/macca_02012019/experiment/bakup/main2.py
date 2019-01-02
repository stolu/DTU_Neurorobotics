# Imported Python Transfer Function
import sys, math
from std_msgs.msg import Float64,Float64MultiArray, Bool
from sensor_msgs.msg import JointState
import numpy as np

sys.path.append("/usr/local/lib")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/home/silvia-neurorobotics/Documents/LWPR")
sys.path.append("/home/silvia-neurorobotics/Documents/NRP/Experiments/icub_ball_balancing")


from AFEL_LWPRandC_class_nrp import MLandC
global MLandC
njoints = 2
n_iter  = 500000
nin     = 8                     # ball_pos table_pos 2 * pos and vel joints
njoints = 2                     # Wrist roll and pitch

LWPRcmd = std_msgs.msg.Float64MultiArray()
LWPRcmd.data   = [0. for i in range( 0, njoints)]
DCNcmd = std_msgs.msg.Float64MultiArray()
DCNcmd.data   = [0. for i in range( 0, njoints)]
LFcmd = std_msgs.msg.Float64MultiArray()
LFcmd.data   = [0. for i in range( 0, njoints)]
cntr_in = std_msgs.msg.Float64MultiArray()
cntr_in.data   = [0. for i in range( 0, njoints)]

# Start time for the vision-based controller
start_time = 5
# variable related to the control commands
@nrp.MapVariable("LWPRcommand",       initial_value = LWPRcmd)
@nrp.MapVariable("DCNcommand", 	      initial_value = DCNcmd)
@nrp.MapVariable("LFcommand", 	      initial_value = LFcmd)
@nrp.MapVariable("controlcommand",    initial_value = cntr_in)

# time parameters
@nrp.MapVariable("starting_time",    initial_value = start_time)

# previous MapVariable
@nrp.MapVariable("PDcommand", initial_value=np.zeros((njoints, n_iter + 1), dtype=np.double))
@nrp.MapVariable("LWPRcommand", initial_value=np.zeros((njoints, n_iter + 1), dtype=np.double))
@nrp.MapVariable("Ccommand", initial_value=np.zeros((njoints, n_iter + 1), dtype=np.double))
@nrp.MapVariable("DCNcommand", initial_value=np.zeros((njoints, n_iter + 1), dtype=np.double))
@nrp.MapVariable("cycle", initial_value=int(0))
@nrp.MapVariable("mlcj_1", initial_value=MLandC(10, 2), scope=nrp.GLOBAL)
@nrp.MapVariable("previous_center", initial_value=(160, 120))
@nrp.MapVariable("previous_ball", initial_value=(160, 120))
@nrp.MapVariable("previous_errors", initial_value=(0.0, 0.0, 0.0, 0.0))
@nrp.MapVariable("rectify_matrix")
@nrp.MapVariable("inv_rectify_matrix")
@nrp.MapRobotSubscriber("joints", Topic("/robot/joints", JointState))
@nrp.MapRobotPublisher("handroll", Topic('/robot/r_wrist_prosup/vel', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("handpitch", Topic('/robot/r_wrist_yaw/vel', std_msgs.msg.Float64))
@nrp.MapRobotSubscriber("camera", Topic('/icub_model/right_eye_camera/image_raw', sensor_msgs.msg.Image))
@nrp.MapRobotPublisher("camera_copy", Topic('/icub_model/right_eye_camera/image_raw2', sensor_msgs.msg.Image))
@nrp.MapRobotPublisher("warped_view", Topic('/icub_model/right_eye_camera/warp_view', sensor_msgs.msg.Image))
@nrp.MapRobotPublisher("error_roll_pos", Topic('/error/roll/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("error_roll_vel", Topic('/error/roll/vel', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("error_pitch_pos", Topic('/error/pitch/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("error_pitch_vel", Topic('/error/pitch/vel', std_msgs.msg.Float64))




@nrp.Neuron2Robot()
def balancing_control(t, PDcommand, LWPRcommand, Ccommand, DCNcommand, cycle, mlcj_1, previous_center, previous_ball,
                      previous_errors, rectify_matrix, inv_rectify_matrix,  joints, handroll, handpitch, camera, camera_copy, 
                      warped_view, error_roll_pos, error_roll_vel, error_pitch_pos, error_pitch_vel,
                      
                      LWPRcommand, DCNcommand, LFcommand, controlcommand,
                      starting_time
                      ):
    
    
    # Umpack LocalData since we want the values
    PDcommand   = PDcommand.value
    LWPRcommand = LWPRcommand.value
    Ccommand    = Ccommand.value
    DCNcommand  = DCNcommand.value
    cycle       = cycle.value
    mlcj_1      = mlcj_1.value
    
    # Read encoders
    joints = joints.value
    handrollpos  = joints.position[joints.name.index('r_wrist_prosup')]
    handrollvel  = joints.velocity[joints.name.index('r_wrist_prosup')]
    handpitchpos = joints.position[joints.name.index('r_wrist_yaw')]
    handpitchvel = joints.velocity[joints.name.index('r_wrist_yaw')]
    # Image processing
    if isinstance(camera.value, type(None)):  # Problem: starts as NoneType
        return
    bridge = CvBridge()
    tf = hbp_nrp_cle.tf_framework.tf_lib
    xy_ball_pos = tf.find_centroid_hsv(camera.value, [50, 100, 100], [70, 255, 255]) \
    or None
    rectified_target_pos = None
    img_in = bridge.imgmsg_to_cv2(camera.value, "rgb8")
    hsv_im   = cv2.cvtColor(img_in, cv2.COLOR_RGB2HSV)
    # Red detection
    lower    = [159, 135, 135]
    upper    = [20, 255, 255]
    lower_np = np.array(lower, dtype = "uint8")
    upper_np = np.array(upper, dtype = "uint8")
    zero     = np.array([0, lower[1], lower[2]], dtype = "uint8")
    end      = np.array([179, upper[1], upper[2]], dtype = "uint8")
    mask1   = cv2.inRange(hsv_im, zero, upper_np)
    mask2   = cv2.inRange(hsv_im, lower_np, end)
    mask    = cv2.bitwise_or(mask1, mask2)
    img_out = cv2.bitwise_and(img_in, img_in, mask=mask)
    # Detect red blobs
    r, g, b = cv2.split(img_out)
    (_, cnts, _) = cv2.findContours(r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #(cnts, _) = cv2.findContours(r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contour of the table
    approx = []
    for c in cnts:
        if cv2.contourArea(c) < 1500:
            clientLogger.info(str("Area wrong = ") + str(cv2.contourArea(c)))
            continue
        c = cv2.convexHull(c, returnPoints=True)
        epsilon = 0.02 * cv2.arcLength(c, True)
        approx  = cv2.approxPolyDP(c, epsilon, True)
        cv2.drawContours(img_in, [approx], -1, (0, 0, 255), 4)
    if len(approx) == 0:
        xy_table_center_pos = previous_center.value
    else:                     
        try:                
            pts1 = np.float32([
                        approx[0][0] + [ 25,  25], 
                        approx[1][0] + [-25,  25],
                        approx[2][0] + [-5, -5],
                        approx[3][0] + [ 5, -5]
                   ])
        except:
            pts1 = np.float32( [x[0] for x in approx] )                   # Trapezoid contour of the table
        pts2 = np.float32( [[300, 300], [0, 300], [0, 0], [300, 0]] )     # Square equivalent
        rectify_matrix     = cv2.getPerspectiveTransform(pts1, pts2)
        inv_rectify_matrix = cv2.getPerspectiveTransform(pts2, pts1)
        rectified_table_center_pos = (150, 100)
        xy_table_center_pos = list(
                                cv2.perspectiveTransform(
                                    np.float32([
                                        ( (rectified_table_center_pos), )
                                    ]), 
                                    inv_rectify_matrix
                                )[0][0]
                              )
    # Draw the center of the table
    thickness = 5
    cv2.line(img_in, tuple(xy_table_center_pos), tuple(xy_table_center_pos), (0, 0, 255), thickness)
    # Draw ball center
    if xy_ball_pos is not None:
        cv2.line(img_in, tuple(xy_ball_pos), tuple(xy_ball_pos), (255, 255, 0), thickness)
    # Send image message
    msg_frame = bridge.cv2_to_imgmsg(img_in, 'rgb8')
    camera_copy.send_message(msg_frame)
    # Compute error
    if t < starting_time.value or xy_ball_pos is None or len([x[0] for x in approx]) != 4:
        errrollpos  = 0.1 - handrollpos
        errrollvel  = 0.0 - handrollvel
        errpitchpos = 0.09 - handpitchpos
        errpitchvel = 0.0  - handpitchvel
        Kp1 = 2.0
        Kd1 = 0.1
        Kp2 = 2.0
        Kd2 = 0.1
    else:
        rectified_target_pos = list(cv2.perspectiveTransform(np.float32([((xy_ball_pos),)]), rectify_matrix)[0][0])
        rectified_target_area_img = cv2.warpPerspective(img_in, rectify_matrix, (300, 300))  # Rectify image to a 300x300 image 
        warped_view.send_message(bridge.cv2_to_imgmsg(rectified_target_area_img, 'rgb8'))
        ballspeed = (np.array(rectified_target_pos) - np.array(previous_ball.value)) / 0.02  # dt = 0.02 - 50Hz
        errrollpos  = -(rectified_table_center_pos[0] - rectified_target_pos[0]) / 300.0
        errrollvel  = -(0.0 - ballspeed[0]) / 300.0                                          # Center of the table is fixed => Vel=0
        errpitchpos = -(rectified_table_center_pos[1] - rectified_target_pos[1]) / 300.0
        errpitchvel = -(0.0 - ballspeed[1]) / 300.0
        previous_errors.value = (errrollpos, errrollvel, errpitchpos, errpitchvel)
        error_roll_pos.send_message( std_msgs.msg.Float64(errrollpos) )
        error_roll_vel.send_message( std_msgs.msg.Float64(errrollvel) )
        error_pitch_pos.send_message( std_msgs.msg.Float64(errpitchpos) )
        error_pitch_vel.send_message( std_msgs.msg.Float64(errpitchvel) )
        Kp1 = 0.4
        Kd1 = 0.1   
        Kp2 = 0.4
        Kd2 = 0.1
    if rectified_target_pos is not None:
        previous_ball.value = rectified_target_pos
    # PD
    PDcommand[0, cycle] = Kp1 * errrollpos  + Kd1 * errrollvel
    PDcommand[1, cycle] = Kp2 * errpitchpos + Kd2 * errpitchvel
    # LWPR and DCN
    if t < starting_time.value:
        cmdroll  = PDcommand[0, cycle]
        cmdpitch = PDcommand[1, cycle]
        mlcj_1 = MLandC(10, 2)
    else:            
        try:
            # LWPR prediction
            (LWPRcommand[:, cycle], DCNcommand[:, cycle]) = mlcj_1.ML_prediction(
                        np.array([
                            xy_ball_pos[0],          xy_ball_pos[1],
                            xy_table_center_pos[0],  xy_table_center_pos[1],
                            ballspeed[0],            ballspeed[1],
                            handrollpos,             handrollvel,
                            handpitchpos,            handpitchvel
                        ]),
                        PDcommand[:, cycle]
            )
        except:
            pass # clientLogger.info(str("No LWPR prediction"))
        # Send commands
        cmdroll  = PDcommand[0, cycle] + LWPRcommand[0, cycle] + DCNcommand[0, cycle]
        cmdpitch = PDcommand[1, cycle] + LWPRcommand[1, cycle] + DCNcommand[1, cycle]
        try:
            # Update LWPR
            mlcj_1.ML_update(
                        np.array([
                            xy_ball_pos[0],          xy_ball_pos[1],
                            xy_table_center_pos[0],  xy_table_center_pos[1],
                            ballspeed[0],            ballspeed[1],
                            handrollpos,             handrollvel,
                            handpitchpos,            handpitchvel
                        ]),
                        np.array([cmdroll, cmdpitch])
            )
        except:
            pass #clientLogger.info(str("No LWPR update"))
    cycle = cycle + 1
    handroll.send_message( std_msgs.msg.Float64(cmdroll) )
    handpitch.send_message( std_msgs.msg.Float64(cmdpitch) )
    r_flds = mlcj_1.ML_rfs()
    clientLogger.info(str("receptive fields = ")+str(r_flds))

