# Imported Python Transfer Function
import sys, math
from std_msgs.msg import Float64,Float64MultiArray, Bool
from sensor_msgs.msg import JointState
import numpy as np
import cv2
sys.path.append("/usr/local/lib")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/home/silvia-neurorobotics/Documents/LWPR")
sys.path.append("/home/silvia-neurorobotics/Documents/NRP/Experiments/icub_ball_balancing")


from AFEL_LWPRandC_class_nrp import MLandC
global MLandC

n_iter  = 500000
nin     = 8                     # ball_pos table_pos 2 * pos and vel joints
njoints = 3                     # Wrist roll and pitch


# Robot information
joints_name = { 0:"r_wrist_prosup",1:"r_wrist_yaw", 2:"r_elbow"}
joints_index = ["r_wrist_prosup","r_wrist_yaw", "r_elbow" ]
curr_joints = sensor_msgs.msg.JointState()
curr_joints.position = [0. for i in range( 0, njoints)]
curr_joints.velocity = [0. for i in range( 0, njoints)]



# Control parameters

Kp_init = [2., 2., 2.]
Kd_init = [0.1 , 0.1, 0.1]
K_in = std_msgs.msg.Float64MultiArray()
K_in.data = [ Kp_init, Kd_init]


Kp_cereb = [0.4, .4, .8]
Kd_cereb = [0.1 , 0.1, 0.1]
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

des_init = std_msgs.msg.Float64MultiArray()
des_init.data = [0. , 0.09, 0.78]

error = sensor_msgs.msg.JointState()
error.position = [0. for i in range( 0, njoints)]
error.velocity = [0. for i in range( 0, njoints)]

# Start time for the vision-based controller
start_time = 5

# Creating the cerebellum
global mlcj_1
n_lwpr_in = 12 
mlcj_1 = MLandC(n_lwpr_in, njoints)



# parameters image processing
# Red detection

lower_np = np.array([159, 135, 135], dtype = "uint8")
upper_np = np.array([20, 255, 255], dtype = "uint8")
zero     = np.array([  0, lower_np[1], lower_np[2]], dtype = "uint8")
end      = np.array([179, upper_np[1], upper_np[2]], dtype = "uint8")
    
pts2 = np.array( [[300, 300], [0, 300], [0, 0], [300, 0]],  dtype='float32' ) 
rectified_table_center_pos = (150, 100)
bridge = CvBridge()
tf = hbp_nrp_cle.tf_framework.tf_lib

# variable related to the control commands
@nrp.MapVariable("K_init",    initial_value = K_in)
@nrp.MapVariable("K_cerebellum",    initial_value = K_cereb)

@nrp.MapVariable("LWPRcommand",       initial_value = LWPRcmd)
@nrp.MapVariable("DCNcommand", 	      initial_value = DCNcmd)
@nrp.MapVariable("LFcommand", 	      initial_value = LFcmd)
@nrp.MapVariable("controlcommand",    initial_value = cntr_in)
@nrp.MapVariable("error",    initial_value = error)
@nrp.MapVariable("desired_init",    initial_value = des_init)


# time parameters
@nrp.MapVariable("starting_time",    initial_value = start_time)


# robot parameters
@nrp.MapVariable("n_joints", 	 initial_value = njoints)
@nrp.MapVariable("joints_idx", 	 initial_value = joints_index)
@nrp.MapVariable("current_joints",    initial_value = curr_joints)




@nrp.MapVariable("previous_center", initial_value=(160, 120))
@nrp.MapVariable("previous_ball", initial_value=(160, 120))
#@nrp.MapVariable("previous_errors", initial_value=(0.0, 0.0, 0.0, 0.0))
@nrp.MapVariable("image_lower_bound", initial_value = lower_np)
@nrp.MapVariable("image_upper_bound", initial_value = upper_np)
@nrp.MapVariable("image_zero", initial_value = zero)
@nrp.MapVariable("image_end", initial_value = end )
@nrp.MapVariable("square2", initial_value = pts2 )
@nrp.MapVariable("rectified_table_center_pos", initial_value = rectified_table_center_pos )
@nrp.MapVariable("CvBridge", initial_value = bridge )
@nrp.MapVariable("nrp_tf", initial_value = tf )

@nrp.MapRobotSubscriber("joints", Topic("/robot/joints", sensor_msgs.msg.JointState))
@nrp.MapRobotPublisher("handroll", Topic('/robot/r_wrist_prosup/vel', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("handpitch", Topic('/robot/r_wrist_yaw/vel', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("elbowpitch", Topic('/robot/r_elbow/vel', std_msgs.msg.Float64))
@nrp.MapRobotSubscriber("camera", Topic('/icub_model/right_eye_camera/image_raw', sensor_msgs.msg.Image))
@nrp.MapRobotPublisher("camera_copy", Topic('/icub_model/right_eye_camera/image_raw2', sensor_msgs.msg.Image))
@nrp.MapRobotPublisher("warped_view", Topic('/icub_model/right_eye_camera/warp_view', sensor_msgs.msg.Image))
@nrp.Neuron2Robot()
def balancing_control(t, starting_time,
                      previous_center, previous_ball, #previous_errors,
                      n_joints, joints_idx, current_joints,
                      joints, handroll, handpitch, elbowpitch,
                      camera, camera_copy, warped_view,
                      image_lower_bound, image_upper_bound, image_zero, image_end,
                      square2, rectified_table_center_pos,
                      CvBridge, nrp_tf,
                      K_init, K_cerebellum,
                      LWPRcommand, DCNcommand, LFcommand, controlcommand,error, desired_init
                      ):

    # Read encoders
    for idx,item in enumerate(joints_idx.value):
        current_joints.value.position[idx] = joints.value.position[joints.value.name.index(item) ]
        current_joints.value.velocity[idx] = joints.value.velocity[joints.value.name.index(item) ]
        
    # =================================== Image Processing ===================================
    try:
        if isinstance(camera.value, type(None)):  # Problem: starts as NoneType
            return

        # Find ball centroid
        xy_ball_pos = nrp_tf.value.find_centroid_hsv(camera.value, [50, 100, 100], [70, 255, 255]) \
        or None
        rectified_target_pos = None
        
        # convert the ros image into a cv matrix
        img_in = CvBridge.value.imgmsg_to_cv2(camera.value, "rgb8")
        # convert the image from one color space to another (RGB to HSV)
        hsv_im   = cv2.cvtColor(img_in, cv2.COLOR_RGB2HSV)

        # Red detection
        mask1   = cv2.inRange(hsv_im, image_zero.value, image_upper_bound.value)
        mask2   = cv2.inRange(hsv_im, image_lower_bound.value, image_end.value)
        mask    = cv2.bitwise_or(mask1, mask2)
        img_out = cv2.bitwise_and(img_in, img_in, mask=mask)

        # extract r g b matrix
        r, g, b = cv2.split(img_out)
        # Detect red blobs
        try:
            (_, cnts, _) = cv2.findContours(r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except:
            (cnts, _) = cv2.findContours(r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        
        
        # Draw contour of the table
        approx = []
        for c in cnts:
            if cv2.contourArea(c) < 1500:
                #clientLogger.info(str("Area wrong = ") + str(cv2.contourArea(c)))
                continue

            c = cv2.convexHull(c, returnPoints=True)
            epsilon = 0.02 * cv2.arcLength(c, True)
            approx  = cv2.approxPolyDP(c, epsilon, True)
            cv2.drawContours(img_in, [approx], -1, (0, 0, 255), 4)
            #clientLogger.info(str(approx))
        
        if len(approx) == 0:
            xy_table_center_pos = previous_center.value

        else:          
            try:
                pts1 = np.array( [ approx[0][0] + [ 25,  25],
                                    approx[1][0] + [-25,  25],
                                    approx[2][0] + [-5, -5],
                                    approx[3][0] + [ 5, -5] ],dtype='float32')

            except:
                pts1 = np.array( [x[0] for x in approx], dtype='float32' )                   # Trapezoid contour of the table


            rectify_matrix     = cv2.getPerspectiveTransform(pts1, square2.value)
            inv_rectify_matrix = cv2.getPerspectiveTransform(square2.value, pts1)
            
            xy_table_center_pos = list( cv2.perspectiveTransform( np.float32([( (rectified_table_center_pos.value), ) ]), 
                                        inv_rectify_matrix )[0][0] )
        
        # Draw the center of the table
        thickness = 5
        cv2.line(img_in, tuple(xy_table_center_pos), tuple(xy_table_center_pos), (0, 0, 255), thickness)

        # Draw ball center
        if xy_ball_pos is not None:
            cv2.line(img_in, tuple(xy_ball_pos), tuple(xy_ball_pos), (255, 255, 0), thickness)

        # Send image message
        msg_frame = CvBridge.value.cv2_to_imgmsg(img_in, 'rgb8')
        camera_copy.send_message(msg_frame)
        
        if t > starting_time.value or xy_ball_pos is not None:
            if len(approx) != 0:
                rectified_target_pos = list(cv2.perspectiveTransform(np.float32([((xy_ball_pos),)]), rectify_matrix)[0][0])

                rectified_target_area_img = cv2.warpPerspective(img_in, rectify_matrix, (300, 300))  # Rectify image to a 300x300 image 

                warped_view.send_message(CvBridge.value.cv2_to_imgmsg(rectified_target_area_img, 'rgb8'))

                ballspeed = (np.array(rectified_target_pos) - np.array(previous_ball.value)) / 0.02  # dt = 0.02 - 50Hz
      
    except Exception as e:
            clientLogger.info(" --> Image Processing Exception: "+str(e)) 

    if rectified_target_pos is not None:
        previous_ball.value = rectified_target_pos
    # =================================== Static Controller ===================================
    if t < starting_time.value or xy_ball_pos is None or len([x[0] for x in approx]) != 4:

        
        for idx in range(0,n_joints.value):
            error.value.position[idx] = desired_init.value.data[idx] - current_joints.value.position[idx]
            error.value.velocity[idx] = 0. - current_joints.value.velocity[idx]
            #LFcommand.value.data[idx] = K_init.value[0][idx]*error.value.position[idx] + K_init.value[1][idx]*error.value.velocity[idx] 
            clientLogger.info(" --> Error joint "+str(idx)+" : "+str(error.value.position[idx]))
            controlcommand.value.data[idx] = LFcommand.value.data[idx]  =current_joints.value.position[idx] +  K_cerebellum.value.data[0][idx]*error.value.position[idx] + K_cerebellum.value.data[1][idx]*error.value.velocity[idx] 
    else:
        


        for idx in range(0,n_joints.value-1):
            error.value.position[idx] = -(rectified_table_center_pos.value[idx] - rectified_target_pos[idx]) / 300.0
            error.value.velocity[idx] = (0.0 - ballspeed[idx]) / 300.0  
        error.value.position[2] = - error.value.position[1] 
        error.value.velocity[2] = - error.value.velocity[1] 
        
        #previous_errors.value = (errrollpos, errrollvel, errpitchpos, errpitchvel)

        
        for idx in range(0,n_joints.value):
            LFcommand.value.data[idx] = K_cerebellum.value[0][idx]*error.value.position[idx] + K_cerebellum.value[1][idx]*error.value.velocity[idx] 
            

        # =================================== Cerebellum Predict ===================================
        input_lwpr = np.array([
                                    xy_ball_pos[0],          xy_ball_pos[1],
                                    xy_table_center_pos[0],  xy_table_center_pos[1],
                                    ballspeed[0],            ballspeed[1],
                                    current_joints.value.position[0], current_joints.value.velocity[0],
                                    current_joints.value.position[1], current_joints.value.velocity[1],
                                    current_joints.value.position[2], current_joints.value.velocity[2]
                                    ])
        try:
            
            # LWPR prediction
            LWPRcommand.value.data,DCNcommand.value.data = mlcj_1.ML_prediction(input_lwpr, LFcommand.value.data)
                                                                                

        except Exception as e:
            clientLogger.info(" --> Prediction Exception: "+str(e)) 

        # =================================== Control Input ===================================
        for idx in range(0,n_joints.value):
            controlcommand.value.data[idx] = LFcommand.value.data[idx] + LWPRcommand.value.data[idx] + DCNcommand.value.data[idx] 
        
        
        # =================================== Cerebellum Update =================================== 
        try:
            # Update LWPR
            mlcj_1.ML_update(input_lwpr,  np.array([ teach for teach in controlcommand.value.data ]) )

        
        except Exception as e:
            clientLogger.info(" --> Update Exception: "+str(e))


    
    handroll.send_message(   std_msgs.msg.Float64( controlcommand.value.data[0] ) )
    handpitch.send_message(  std_msgs.msg.Float64( controlcommand.value.data[1] ) )
    elbowpitch.send_message( std_msgs.msg.Float64( controlcommand.value.data[2] ) )

