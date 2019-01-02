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

from table_detect import*
global detect

detect = detect_table(True)
detect.image_lower_bound_m1 = np.array([   0,  50,  50], dtype = "uint8")
detect.image_upper_bound_m1 = np.array([  10, 255, 255], dtype = "uint8")
detect.image_lower_bound_m2 = np.array([  170,  50,  50], dtype = "uint8")
detect.image_upper_bound_m2 = np.array([ 180, 255, 255], dtype = "uint8")
detect.max_offset = 20.
detect.debug = False

detect.show_contour = False
detect.previous_center = [137,120]
bridge = CvBridge()
tf = hbp_nrp_cle.tf_framework.tf_lib


# Ball error

error = sensor_msgs.msg.JointState() # it is not per joint but per coordinate
error.position = [0. for i in range( 0, 2)]
error.velocity = [0. for i in range( 0, 2)]

distance = std_msgs.msg.Float64MultiArray()
distance.data   = [0. for i in range( 0, 2)]

b_pos = sensor_msgs.msg.JointState() # it is not per joint but per coordinate
b_pos.position = [0.5 for i in range( 0, 2)]
b_pos.velocity = [0.5 for i in range( 0, 2)]
b_pos.effort   = [0.5 for i in range( 0, 2)]

@nrp.MapVariable("vision_debug", initial_value = False)
@nrp.MapVariable("starting_time", initial_value = 5)
@nrp.MapVariable("xy_table_center_pos", initial_value=[0.6, 0.4])
@nrp.MapVariable("previous_center", initial_value=[0.6 , .4])
@nrp.MapVariable("previous_ball", initial_value=[0.5, 0.5])
@nrp.MapVariable("previous_errors", initial_value=[0.0, 0.0])
@nrp.MapVariable("error",    initial_value = error)
@nrp.MapVariable("ball_distance",    initial_value = distance)

#@nrp.MapVariable("ballspeed",    initial_value = [0.,0.])

@nrp.MapVariable("image_shape",    initial_value = [240.,320.])
@nrp.MapVariable("rectify_matrix")
@nrp.MapVariable("inv_rectify_matrix")


@nrp.MapVariable("cvbridge", initial_value = bridge )
@nrp.MapVariable("nrp_tf", initial_value = tf )


@nrp.MapVariable("ball_pos", initial_value = b_pos )


# ** Map subscribers **
@nrp.MapRobotSubscriber("camera", Topic('/icub_model/right_eye_camera/image_raw', sensor_msgs.msg.Image))

# ** Map publisher **
@nrp.MapRobotPublisher("camera_copy", Topic('/icub_model/right_eye_camera/image_raw2', sensor_msgs.msg.Image))
@nrp.MapRobotPublisher("warped_view", Topic('/icub_model/right_eye_camera/warp_view', sensor_msgs.msg.Image))
@nrp.MapRobotPublisher("pose_ball_pub",  Topic('/ball/pose', sensor_msgs.msg.JointState)) # table pose in effort information
@nrp.MapRobotPublisher("error_ball_pub",  Topic('/ball/error', sensor_msgs.msg.JointState))
@nrp.MapRobotPublisher("distance_ball_pub", Topic('/ball/distance', std_msgs.msg.Float64MultiArray) )


# -----------------------------------------------------------------------#
# ** Recording the control input **

@nrp.MapCSVRecorder("ball_recorder", filename="ball.csv", headers=["time", 
                                                                   "x_table", "y_table",
                                                                   "x_ball","y_ball",
                                                                   "e_x","e_y",
                                                                    "e_vel_x", "e_vel_y",
                                                                    "dist_pose", " dist_vel"
                                                                         ])

@nrp.Neuron2Robot()
def vision_simple (t, 
                    starting_time, vision_debug,
                    previous_center, previous_ball,previous_errors, xy_table_center_pos,
                    cvbridge, nrp_tf, 
                    error, ball_distance,
                    rectify_matrix, inv_rectify_matrix,  camera,
                    ball_pos, 
                    camera_copy, warped_view, 
                    pose_ball_pub, error_ball_pub, distance_ball_pub,
                    
                    image_shape,
                    ball_recorder):
    
    try:

        # Image processing
        if isinstance(camera.value, type(None)):
            return
        img_in = cvbridge.value.imgmsg_to_cv2(camera.value, "rgb8")

        if t > 30 and t < 40:
            detect.max_offset = 10.
        if t > starting_time.value:
            xy_table, rectified_target_area_img = detect.find_center(img_in)
            
            for idx in range(0,2):
                xy_table_center_pos.value[idx] = float(xy_table[idx])#/image_shape.value[idx]
                ball_pos.value.effort[idx] = xy_table_center_pos.value[idx]
            if vision_debug.value == True :
                clientLogger.info("table center "+str(xy_table_center_pos.value))
           
            
            # =================================== Ball Detection ===================================
            # Find ball centroid
            msg_frame_2 = cvbridge.value.cv2_to_imgmsg(rectified_target_area_img, 'rgb8')
            #warped_pose =  nrp_tf.value.find_centroid_hsv(msg_frame_2, [50, 100, 100], [70, 255, 255])
            #clientLogger.info("warped pose"+str(warped_pose))
            xy_ball_pos = nrp_tf.value.find_centroid_hsv(camera.value, [50, 100, 100], [70, 255, 255]) \
            or None
            if xy_ball_pos is not None:
                if vision_debug.value == True :   
                    clientLogger.info("xy_ball_pos pose"+str(xy_ball_pos))
                detect_ball = detect.ball_pose( xy_ball_pos  )
                
                if vision_debug.value == True :   
                    clientLogger.info("warped pose"+str(detect_ball))
            #xy_ball_pos = nrp_tf.value.find_centroid_hsv(camera.value, [50, 100, 100], [70, 255, 255]) \
            #or None
            
            if vision_debug.value == True :    
                clientLogger.info("ball pose "+str(xy_ball_pos))
            # Draw ball center
            if xy_ball_pos is not None:
                cv2.line(img_in, tuple(xy_ball_pos), tuple(xy_ball_pos), (255, 255, 0), 5) 
                #cv2.circle(img_in,tuple(xy_ball_pos), tuple(xy_ball_pos), (255, 255, 0), -1)
                cv2.putText(img_in, "center%s"%str(xy_ball_pos), xy_ball_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2) 


            if xy_ball_pos is None :
                
                for idx in range(0,2):
                    ball_pos.value.position[idx] = np.sign(previous_ball.value[idx])#*image_shape.value[idx]
                if vision_debug.value == True :    
                    clientLogger.info("\n   ball_pos is none "+str(ball_pos.value))
                
            else:
                
                for i in range(0,2): 
                    ball_pos.value.position[i] = float(xy_ball_pos[i])#/image_shape.value[idx]
                if vision_debug.value == True : 
                    clientLogger.info("new ball "+str(ball_pos.value.position ))
            
            # =================================== Send image message ===================================

            msg_frame = cvbridge.value.cv2_to_imgmsg(img_in, 'rgb8')
            camera_copy.send_message(msg_frame)
            
            #warped_view.send_message(msg_frame_2)
            
            # =================================== Compute Error ===================================


            for idx in range(0,2):
                ball_pos.value.velocity[idx] = ( ball_pos.value.position[idx] - previous_ball.value[idx] ) / 0.02  # dt = 0.02 - 50Hz
                
                if vision_debug.value == True :    
                    clientLogger.info("\n ball  speed "+str(ball_pos.value.velocity[idx]))
                
                error.value.position[idx] = ( float(xy_table_center_pos.value[idx]) - ball_pos.value.position[idx])/150.
                error.value.velocity[idx] = (error.value.position[idx] - previous_errors.value[idx])/0.02#( 0.0 - ballspeed.value[idx]) 
                
                previous_ball.value[idx] = ball_pos.value.position[idx]
                previous_errors.value[idx] = error.value.position[idx]
            
            if vision_debug.value == True :            
                clientLogger.info("\n error position "+str(error.value.position))
                clientLogger.info("\n error velocity "+str(error.value.velocity))
            
            # =================================== Compute Distance ===================================
            
            # ball_distance = [ position, velocity ]
            ball_distance.value.data[0] = error.value.position[0]#(np.sign(error.value.position[0]*error.value.position[1]))* np.sqrt(np.sum([ pose**2 for pose in error.value.position]) )
            ball_distance.value.data[1] = error.value.position[1] #np.sqrt(np.sum([  vel**2 for  vel in error.value.velocity]) ) 
            
            if vision_debug.value == True :            
                clientLogger.info("\n ball  distance "+str(ball_distance.value.data))
                
            # =================================== Send Message to Controller ===================================
            pose_ball_pub.send_message( ball_pos.value )
            error_ball_pub.send_message( error.value )
            distance_ball_pub.send_message( ball_distance.value )
            
        # =================================== Record ===================================        
        ball_recorder.record_entry(t,
                                     xy_table_center_pos.value[0], xy_table_center_pos.value[1],
                                                ball_pos.value.position[0],            ball_pos.value.position[1],
                                          error.value.position[0],      error.value.position[1],
                                          error.value.velocity[0],      error.value.velocity[1],
                                      ball_distance.value.data[0],  ball_distance.value.data[1] )
        
    except Exception as e:
        clientLogger.info(" --> Vision Exception: "+str(e))

        