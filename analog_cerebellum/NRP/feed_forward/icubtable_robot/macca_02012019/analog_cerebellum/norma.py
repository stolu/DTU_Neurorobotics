import numpy as np

class norm_input:
    def __init__(self, x_min_des, x_max_des, x_min_in, x_max_in):
        self.min_des = x_min_des
        self.max_des = x_max_des
        self.min_input = x_min_in
        self.max_input = x_max_in
    

    def get_norm(self,x):
        return   ( self.max_des - self.min_des)*(x - self.min_input)/(self.max_input - self.min_input) + self.min_des

    

