"""
Created on  Oct 25 11:31:10 2018
@author: Marie Claire Capolei macca@elektro.dtu.dk
"""
import sys
sys.path.append("/usr/local/lib")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import numpy as np


class static_controller:
    
    def __init__(self, njoints, Kp, Kd, Ki, derivation_step, integration_step ):

        

        # Control parameters

        self.n_joints = njoints
        self.K_init = [ Kp, Kd , Ki ]


        self.PID   = [0. for i in range( 0, njoints)]

        self.error_position = [0. for i in range( 0, njoints)]
        self.error_velocity = [0. for i in range( 0, njoints)]
        self.error_effort   = [0. for i in range( 0, njoints)]
        
        self.previous_error_position = [0. for i in range( 0, njoints)]
        self.previous_error_velocity = [0. for i in range( 0, njoints)]
        self.previous_error_effort   = [0. for i in range( 0, njoints)]
        self.previous_errors         = [ [0.] for i in range( 0, njoints)]
        self.integral_time           = [ [0.] for i in range( 0, njoints)]
        
        self.dt = derivation_step
        self.N  = integration_step
        self.debug = False
    
    def integration(self, idx , t):
        
        if len(self.previous_errors[idx]) < int( self.N):
            self.previous_errors[idx].append( self.error_position[idx])
            self.integral_time[idx].append(t)
            dummy_N = len( self.previous_errors[idx])
            return ( ( self.integral_time[idx][ dummy_N-1] - self.integral_time[idx][0])/dummy_N)* ( self.previous_errors[idx][0]*0.5 + self.previous_errors[idx][dummy_N-1]*0.5 + np.sum( self.previous_errors[idx][1:-1]))
        
            
        else:
            self.previous_errors[idx].append( self.error_position[idx] )
            self.previous_errors[idx].pop(0)
            self.integral_time[idx].append(t)
            self.integral_time[idx].pop(0)
            return ( (self.integral_time[idx][int(self.N)-1] - self.integral_time[idx][0])/self.N)* ( self.previous_errors[idx][0]*0.5 + self.previous_errors[idx][int(self.N)-1]*0.5 + np.sum(self.previous_errors[idx][1:-1]))

    
    def control(self, desired_init, current_joints, delta_e, t):
         for idx in range(0, self.n_joints):

                # ** Error **
                self.error_position[idx] = desired_init[idx] - current_joints[idx] + delta_e[idx]
                self.error_velocity[idx] = ( self.error_position[idx] - self.previous_error_position[idx] )/self.dt
                self.error_effort[idx]   = self.integration(idx,t)
                
                self.previous_error_position[idx] = self.error_position[idx]
                #self.previous_error_velocity[idx] = self.error_velocity[idx]
                #self.previous_error_effort[idx]   = self.error_effort[idx]
                
                
                if self.debug == True:
                        print("erro joint "+str(idx)+"\n position "+str(self.error_position[idx])+"\n velocity "+str(self.error_velocity[idx])+"\n effort "+str(self.error_effort[idx]))
                
                
                
                self.PID[idx] = self.K_init[0][idx]*self.error_position[idx] + self.K_init[1][idx]*self.error_velocity[idx] + self.K_init[2][idx]*self.error_effort[idx]
    
          
        