
import matplotlib.pyplot as plt
import numpy as np
import random
import math

class RBF:

    def __init__(self, total_n_neurons, min_in_delta, max_in_delta, min_o, max_o):
        # Variables initialization
        self.n_neurons = total_n_neurons
        self.out   = [0.]*self.n_neurons
        #print self.out
        
        self.min_input_range    = min_in_delta
        self.max_input_range    = max_in_delta
        self.total_input_range  = self.max_input_range - self.min_input_range
        
        		
        
        	
        self.sub_range = self.total_input_range/self.n_neurons
        self.sigma     = self.sub_range/2.
        self.sub_range_center = [0.]*self.n_neurons
        #print self.sub_range_center
        j = 0
        self.sub_range_center[0] = self.min_input_range + self.sigma
        for i in range(1, self.n_neurons):
            self.sub_range_center[i] = self.sub_range_center[i-1] + 2*self.sigma
            j = j + 2
        print self.sub_range_center
        print("\n self.sub_range_center[self.n_neurons-1]"+str(self.sub_range_center[self.n_neurons-1]))
        
        self.out_max = max_o
        self.out_min = min_o
        self.in_max = ( ( 0. )**2 + (self.sigma)**2)**(-0.5)#1.0
        self.in_min = ( ( self.min_input_range - self.sub_range_center[self.n_neurons-1] )**2 + (self.sigma)**2)**(-0.5)
        #self.in_max = math.exp( -  (1./self.sigma)*(abs( 0. ))**2 )
        #self.in_min = math.exp( -  (1./self.sigma)*(abs( self.min_input_range - self.sub_range_center[self.n_neurons-1] ))**2 )
        print("\n self.in_min "+str(self.in_min))
    def norm(self, val_pre_scale):
        return ( self.out_max - self.out_min)*(val_pre_scale - self.in_min)/(self.in_max - self.in_min) + self.out_min   
     
    def function(self, inp):
        # Calculate position intervals for RBFs 
        
        for i in range(0, self.n_neurons):            
            #exponent       = self.sigma*(abs( inp - self.sub_range_center[i] ))**2 #( ( inp - self.sub_range_center[i] )/self.sigma )**2
            #value_prescale = math.exp( - exponent)
            value_prescale = ( ( inp - self.sub_range_center[i] )**2 + (self.sigma)**2)**(-0.5)
            self.out[i] = self.norm(value_prescale)
        
        return self.out
    
    def plot_rbf(self):
        plt.plot( self.sub_range_center, self.out , 'bx')
        #self.input_position_current , [np.linspace(0, 10, 100, endpoint=True) ]*len(self.input_position_current), 'gx', self.sub_range_current_mean , self.ac_source_amplitude[0][:], 'b.', self.sub_range_current_mean , self.ac_source_amplitude[1][:], 'r.')
        #plt.gcf().set_size_inches(8,3)
        #plt.gcf().tight_layout()
        plt.grid()
        plt.title('RBFs output')
        plt.xlabel('Angular position (radiants)')
        plt.ylabel('AC source amplitude (mA)')
        plt.show()





