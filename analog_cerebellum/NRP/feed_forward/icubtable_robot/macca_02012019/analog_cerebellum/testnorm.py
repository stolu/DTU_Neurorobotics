# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 13:05:42 2018

@author: silvia-neurorobotics
"""
import numpy as np
import matplotlib.pyplot as plt



global min_io, max_io
min_io = -1.
max_io = 1.


def saturate_signal( signal, min_bound, max_bound):
    if signal > max_bound:
        return max_bound
    elif signal < min_bound:
        return min_bound
    else:
        return signal
        
def norma( signal, min_signal, max_signal, sign):
    # Normalization formula : sign*( max_des - min_des)*(x - x_min)/(x_max - x_min) + min_des
    if sign == 1:
        sign_signal = np.sign(signal)
    else:
        sign_signal = 1.
    return sign_signal*(max_io - min_io )*( saturate_signal(signal, min_signal, max_signal) - min_signal)/(max_signal - min_signal)


c = norma(0.3, 0, 0.7 , 0)
print(c)
input_in = 0.5
input_min = 0.
input_max = 0.7
d = (max_io - min_io )*(input_in - input_min )/(input_max -input_min)
print(d)

f= []

max_input = 0.7
k = 1.
x0 = 0.
input_in = 0.3
t = -6.
dt=0.1
for i in range(0,100):
    #f.append(max_input/(1. + np.exp(-k*(t - x0))) )
    #f.append(0.5 +0.5*np.tanh(t/2.))    
    f.append(1./(1.+np.exp(-k*(t -0.5  ) ) ) -0.5 )    
    t=t+dt
#print(f)


x_in = 0.5
a = -1.
b = 1.
x_min = -0.2
x_max = 0.5

normalo = a+ (b-a)*(x_in - x_min)/(x_max-x_min)
print("norm "+str(normalo) )

for i in range(0,5):
    #f.append(max_input/(1. + np.exp(-k*(t - x0))) )
    #f.append(0.5 +0.5*np.tanh(t/2.))    
    f.append( a+ (b-a)*(x_in - x_min)/(x_max-x_min))    
    x_in+=dt

print(e)
plt.plot(f,'o')

plt.show()