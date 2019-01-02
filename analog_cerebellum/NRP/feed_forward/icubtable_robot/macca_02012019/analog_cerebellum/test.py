from rad_bas_f import RBF
import numpy as np

def read_datafile(file_name):
    # the skiprows keyword is for heading, but I don't know if trailing lines
    # can be specified
    data = np.loadtxt(file_name, delimiter=',', skiprows=1, dtype=float)
    title = np.loadtxt(file_name, delimiter=',', dtype='str')
    return data,title[0,:]
    
joint_data, title = read_datafile('joint_info.csv')

n_neur = 100
max_input_range = 20.
min_input_range = -20.

out_max = 1.
out_min = 0.00001

test1 = RBF(n_neur, min_input_range, max_input_range, out_min, out_max)


test1.total_input_range = np.pi
print(" \n joint_data[2,1]" +str(joint_data[2,1]))
test1.function(0.001)#joint_data[2,1])
test1.plot_rbf()
#print test1.ac_source_amplitude[:]
for i in range(0,n_neur):
    if float(test1.out[i])> 0.0005 :
        print test1.out[i]