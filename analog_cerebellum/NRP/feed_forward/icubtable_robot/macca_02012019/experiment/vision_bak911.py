# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:20:50 2018

@author: silvia-neurorobotics
"""

# Imported Python Transfer Function
import sys, math
sys.path.append("/usr/local/lib/python2.7/dist-packages")

from std_msgs.msg import Float64, Float64MultiArray
import cv2
import numpy as np

bridge = CvBridge()
tf = hbp_nrp_cle.tf_framework.tf_lib

# Red detection
lower_np = np.array([159, 135, 135], dtype = "uint8")
upper_np = np.array([20, 255, 255], dtype = "uint8")
zero     = np.array([  0, lower_np[1], lower_np[2]], dtype = "uint8")
end      = np.array([179, upper_np[1], upper_np[2]], dtype = "uint8")

    
pts2 = np.array( [[300, 300], [0, 300], [0, 0], [300, 0]],  dtype='float32' ) 
rectified_table_center_pos = (150, 100)


# Ball error

error = sensor_msgs.msg.JointState() # it is not per joint but per coordinate
error.position = [0. for i in range( 0, 2)]
error.velocity = [0. for i in range( 0, 2)]

distance = std_msgs.msg.Float64MultiArray()
distance.data   = [0. for i in range( 0, 2)]

@nrp.MapVariable("vision_debug", initial_value = True)
@nrp.MapVariable("starting_time", initial_value = 5)

@nrp.MapVariable("previous_center", initial_value=(160, 120))
@nrp.MapVariable("previous_ball", initial_value=(160, 120))
@nrp.MapVariable("previous_errors", initial_value=(0.0, 0.0, 0.0, 0.0))
@nrp.MapVariable("error",    initial_value = error)
@nrp.MapVariable("ball_distance",    initial_value = distance)

@nrp.MapVariable("ballspeed",    initial_value = [0.,0.])

@nrp.MapVariable("rectify_matrix")
@nrp.MapVariable("inv_rectify_matrix")


@nrp.MapVariable("cvbridge", initial_value = bridge )
@nrp.MapVariable("nrp_tf", initial_value = tf )
@nrp.MapVariable("square2", initial_value = pts2 )
@nrp.MapVariable("rectified_table_center_pos", initial_value = rectified_table_center_pos )
@nrp.MapVariable("prev_rectified_ball_pos", initial_value = [150.,150.] )
@nrp.MapVariable("rectified_target_pos", initial_value = [0.,0.] )

@nrp.MapVariable("image_lower_bound", initial_value = lower_np)
@nrp.MapVariable("image_upper_bound", initial_value = upper_np)
@nrp.MapVariable("image_zero", initial_value = zero)
@nrp.MapVariable("image_end", initial_value = end )


# ** Map subscribers **
@nrp.MapRobotSubscriber("camera", Topic('/icub_model/right_eye_camera/image_raw', sensor_msgs.msg.Image))

# ** Map publisher **
@nrp.MapRobotPublisher("camera_copy", Topic('/icub_model/right_eye_camera/image_raw2', sensor_msgs.msg.Image))
@nrp.MapRobotPublisher("warped_view", Topic('/icub_model/right_eye_camera/warp_view', sensor_msgs.msg.Image))
@nrp.MapRobotPublisher("error_ball", Topic('/ball/error', sensor_msgs.msg.JointState))
@nrp.MapRobotPublisher("distance_ball", Topic('/ball/distance', std_msgs.msg.Float64MultiArray) )


# -----------------------------------------------------------------------#
# ** Recording the control input **

@nrp.MapCSVRecorder("ball_recorder", filename="ball.csv", headers=["time",
                                                                         "x","y", 
                                                                         "vel_x", "vel_y",
                                                                         "dist_pose", " dist_vel"
                                                                         ])

@nrp.Neuron2Robot()
def vision (t, starting_time, vision_debug,
            previous_center, previous_ball,previous_errors,
            cvbridge, nrp_tf,square2, rectified_table_center_pos,
            error, ball_distance,
            image_lower_bound, image_upper_bound, image_zero, image_end,
            rectify_matrix, inv_rectify_matrix,  camera,
            rectified_target_pos, prev_rectified_ball_pos,
            camera_copy, warped_view, 
            error_ball, distance_ball,
            ballspeed,
            
            ball_recorder):
    
    try:
    
        # Image processing
        if isinstance(camera.value, type(None)):  # Problem: starts as NoneType
            return
        
        # Find ball centroid
        xy_ball_pos = nrp_tf.value.find_centroid_hsv(camera.value, [50, 100, 100], [70, 255, 255]) \
        or None
          
        
        
        img_in = cvbridge.value.imgmsg_to_cv2(camera.value, "rgb8")
        hsv_im   = cv2.cvtColor(img_in, cv2.COLOR_RGB2HSV)
        
        
        # Red detection
        mask1   = cv2.inRange(hsv_im, image_zero.value, image_upper_bound.value)
        mask2   = cv2.inRange(hsv_im, image_lower_bound.value, image_end.value)
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
                #clientLogger.info(str("Area wrong = ") + str(cv2.contourArea(c)))
                continue
            c = cv2.convexHull(c, returnPoints=True)
            epsilon = 0.02 * cv2.arcLength(c, True)
            approx  = cv2.approxPolyDP(c, epsilon, True)
            cv2.drawContours(img_in, [approx], -1, (0, 0, 255), 4)
        if len(approx) == 0:
            xy_table_center_pos = [160,120]#previous_center.value
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
            
            rectify_matrix     = cv2.getPerspectiveTransform(pts1, square2.value)
            inv_rectify_matrix = cv2.getPerspectiveTransform(square2.value, pts1)
            
            xy_table_center_pos = list(
                                    cv2.perspectiveTransform(
                                        np.float32([
                                            ( (rectified_table_center_pos.value), )
                                        ]), 
                                        inv_rectify_matrix
                                    )[0][0]
                                  )
            previous_center.value = xy_table_center_pos ## remember this change!!
        
        # Draw the center of the table
        thickness = 5
        cv2.line(img_in, tuple(xy_table_center_pos), tuple(xy_table_center_pos), (0, 0, 255), thickness)
        
        # Draw ball center
        if xy_ball_pos is not None:
            cv2.line(img_in, tuple(xy_ball_pos), tuple(xy_ball_pos), (255, 255, 0), thickness)
        # Send image message
        msg_frame = cvbridge.value.cv2_to_imgmsg(img_in, 'rgb8')
        camera_copy.send_message(msg_frame)
        # Compute error
        if t > starting_time.value:
            if xy_ball_pos is None or len([x[0] for x in approx]) != 4:
                for idx in range(0,2):
                    rectified_target_pos.value[idx] = np.sign(prev_rectified_ball_pos.value[idx])*150.
                clientLogger.info("\n   rectified_target_pos "+str(rectified_target_pos.value))
                
            else:
                rectified_target_pos.value = list(cv2.perspectiveTransform(np.float32([((xy_ball_pos),)]), rectify_matrix)[0][0])
                rectified_target_area_img = cv2.warpPerspective(img_in, rectify_matrix, (300, 300))  # Rectify image to a 300x300 image 
                warped_view.send_message(cvbridge.value.cv2_to_imgmsg(rectified_target_area_img, 'rgb8'))
                
                
            for idx in range(0,2):
                ballspeed.value[idx] = (np.array(rectified_target_pos.value[idx]) - np.array(previous_ball.value[idx])) / 0.02  # dt = 0.02 - 50Hz
                clientLogger.info("\n ball  speed "+str(ballspeed.value[idx]))
                
                error.value.position[idx] = (rectified_table_center_pos.value[idx] - rectified_target_pos.value[idx]) / 300.0 #-(rectified_table_center_pos.value[idx] - rectified_target_pos[idx]) / 300.
                clientLogger.info("\n  rectified_target_pos "+str(idx)+" "+str( rectified_target_pos.value[idx]))
                error.value.velocity[idx] = (0.0 - ballspeed.value[idx]) / 300.0 #-(0.0 - ballspeed[idx]) / 300.0
                prev_rectified_ball_pos.value[idx] = rectified_target_pos.value[idx]
            ball_distance.value.data[0] =(np.sign(error.value.position[0]*error.value.position[1]))* np.sqrt(np.sum([ pose**2 for pose in error.value.position]) )
            ball_distance.value.data[1] = np.sqrt(np.sum([  vel**2 for  vel in error.value.velocity]) )            
            if vision_debug.value == True :            
                clientLogger.info("\n error position "+str(error.value.position))
                clientLogger.info("\n error velocity "+str(error.value.velocity))
                clientLogger.info("\n ball  distance "+str(ball_distance.value.data))
                
            #previous_errors.value = (errrollpos, errrollvel, errpitchpos, errpitchvel)
            error_ball.send_message( error.value )
            distance_ball.send_message( ball_distance.value )
            
            ball_recorder.record_entry(t,     error.value.position[0],     error.value.position[1],
                                              error.value.velocity[0],     error.value.velocity[1],
                                          ball_distance.value.data[0], ball_distance.value.data[1] )
        if rectified_target_pos.value is not None:
            previous_ball.value = rectified_target_pos.value
    except Exception as e:
        clientLogger.info(" --> Vision Exception: "+str(e))

        