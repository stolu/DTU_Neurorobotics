import numpy as np

class norm_input:
    def __init__(self, x_min_des, x_max_des):
        self.min_des = x_min_des
        self.max_des = x_max_des
    
    def get_norm(self,x, x_min, x_max):
        x_out = ( self.max_des - self.min_des)*(x - x_min)/(x_max - x_min) + self.min_des
        
        return x_out
    

