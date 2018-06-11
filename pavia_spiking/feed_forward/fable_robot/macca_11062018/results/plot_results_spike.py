
import numpy as np

import matplotlib.pyplot as plt


def read_datafile(file_name):
    # the skiprows keyword is for heading, but I don't know if trailing lines
    # can be specified
    data = np.loadtxt(file_name, delimiter=',', skiprows=1, dtype=float)
    title = np.loadtxt(file_name, delimiter=',', dtype='str')
    return data,title[0,:]
    
joint_data, title = read_datafile('joint_info.csv')
control_data, title_cmd = read_datafile('command.csv')
#print joint_data[0,:]
#print title


fntsz = 14
y_range_limit = [-1.5, 1.5]
# ** Plotting position **
fig = 1
plot_joint = 1
plot_control = 1


if plot_joint == 1:
    # ** plotting the joint information
    plt.figure(fig)
    
    sbplt = 211
    i=0
    
    # ** Joint 1
    plt.subplot(sbplt)
    plt.title('$ joint_1 $ ')
    plt.plot(joint_data[:,0], joint_data[:,1], 'b',label = title[1])
    plt.plot(joint_data[:,0], joint_data[:,7], 'r',label = title[7] )
    plt.ylim(y_range_limit)
    plt.ylabel('position [ $ rad $ ] ', fontsize=fntsz, color='black')
    plt.legend()
    plt.grid(True)
    
    plt.annotate('Cerebellum in action', xy=(2, 0.3), xytext=(3, 1.5),
                arrowprops=dict(facecolor='black', shrink=0.05),
                )
    
    i = i + 1
    print i
    plt.subplot(sbplt+i)
    plt.title('$ error joint_1 $ ')
    plt.plot(joint_data[:,0], joint_data[:,13], 'b',label = title[13])
    plt.ylim(y_range_limit)
    #plt.ylabel('position error [ $ rad $ ] ', fontsize=fntsz, color='black')
    plt.legend()
    plt.grid(True)
    
    plt.annotate('Cerebellum in action', xy=(2, 0.3), xytext=(3, 1.5),
                arrowprops=dict(facecolor='black', shrink=0.05),
                )
    fig = fig+1  
    # ** Joint 2  
    plt.figure(fig)
    i = 0
    plt.subplot(sbplt+i)
    plt.title('$ joint_2 $ ')
    plt.plot(joint_data[:,0], joint_data[:,2], 'b',label = title[2])
    plt.plot(joint_data[:,0], joint_data[:,8], 'r',label = title[8] )
    plt.ylim(y_range_limit)
    plt.legend()
    plt.ylabel('position [$ rad $]', fontsize=fntsz, color='black')
    plt.xlabel('time [$\sec$]', fontsize=fntsz, color='black')
    plt.grid(True)
    
    i = i + 1
    plt.subplot(sbplt+i)
    plt.title('$ error joint_2 $ ')
    plt.plot(joint_data[:,0], joint_data[:,14], 'b',label = title[14])
    plt.ylim(y_range_limit)
    plt.ylabel('position error [ $ rad $ ] ', fontsize=fntsz, color='black')
    plt.legend()
    plt.grid(True)
    fig = fig+1 
    # ** xy-plane
    plt.figure(fig) #i = i + 1
    #plt.subplot(313)#sbplt+i)
    plt.title(' trajectory xy-plane  ')
    plt.plot(joint_data[:,1], joint_data[:,2], 'b',label = 'trajectory xy-plane')
    plt.plot(joint_data[:,7], joint_data[:,8], 'r',label = 'desired trajectory xy-plane' )
    plt.legend()
    plt.ylabel('joint 1 position [$ rad $]', fontsize=fntsz, color='black')
    plt.xlabel('joint 2 position [$ rad $]', fontsize=fntsz, color='black')
    plt.grid(True)

    
    plt.show()
    fig = fig +1



if plot_control == 1:
    # ** plotting the joint information
    plt.figure(fig)
    y_range_limit = []
    sbplt = 211
    i=0
    '''
    # ** Joint 1
    plt.subplot(sbplt)
    plt.title('$ joint_1 $ ')
    plt.plot(control_data[:,0], control_data[:,1], 'b',label = title_cmd[1])
    
    #plt.ylim(y_range_limit)
    plt.ylabel('position [ $ rad $ ] ', fontsize=fntsz, color='black')
    plt.legend()
    plt.grid(True)
    
    #plt.annotate('Cerebellum in action', xy=(2, 0.3), xytext=(3, 1.5), arrowprops=dict(facecolor='black', shrink=0.05),)
    
    i = i + 1
    plt.subplot(sbplt + i)
    plt.title('$ joint_1 $ ')
    plt.plot(control_data[:,0], control_data[:,3], 'r',label = title_cmd[3] )
    #plt.ylim(y_range_limit)
    plt.ylabel('position [ $ rad $ ] ', fontsize=fntsz, color='black')
    plt.legend()
    plt.grid(True)
    

    # ** Joint 2  
    
    i = i + 1  
    plt.subplot(sbplt+i)
    plt.title('$ joint_2 $ ')
    plt.plot(control_data[:,0], control_data[:,2], 'b',label = title_cmd[2])
    #plt.ylim(y_range_limit)
    plt.legend()
    plt.ylabel('position [$ rad $]', fontsize=fntsz, color='black')
    plt.xlabel('time [$\sec$]', fontsize=fntsz, color='black')
    plt.grid(True)
    
    i = i + 1
    plt.subplot(sbplt+i)
    plt.title('$ joint_2 $ ')
    plt.plot(control_data[:,0], control_data[:,4], 'r',label = title_cmd[4] )
    #plt.ylim(y_range_limit)
    plt.legend()
    plt.ylabel('position [$ rad $]', fontsize=fntsz, color='black')
    plt.xlabel('time [$\sec$]', fontsize=fntsz, color='black')
    plt.grid(True)
    '''
    # ** xy-plane
    #i = i + 1
    plt.subplot(sbplt+i)
    plt.title(' control input $joint_1$')
    plt.plot(control_data[:,0], (control_data[:,7] ), 'g',label = title_cmd[7] )
    
    plt.legend()
    plt.ylabel('joint 1 control [$ N $]', fontsize=fntsz, color='black')
    plt.xlabel('time [$ s $]', fontsize=fntsz, color='black')
    plt.grid(True)
    
    # ** xy-plane
    i = i + 1
    plt.subplot(sbplt+i)
    plt.title(' contribution cerebellum and lf ')
    plt.plot(control_data[:,0], ( control_data[:,5]  ), 'g',label = title_cmd[5] )
    plt.plot(control_data[:,0], (  control_data[:,3] ), 'r',label = title_cmd[3] )
    
    plt.legend()
    plt.ylabel('joint 1 control [$ N $]', fontsize=fntsz, color='black')
    plt.xlabel('time [$ s $]', fontsize=fntsz, color='black')
    plt.grid(True)
    
    plt.show()
    fig = fig +1
    
    
    plt.figure(fig)
    i = 0
    plt.subplot(sbplt+i)
    plt.title(' control input $joint_2$')
    plt.plot(control_data[:,0], (control_data[:,8] ), 'g',label = title_cmd[8] )
    
    plt.legend()
    plt.ylabel('joint 2 control [$ N $]', fontsize=fntsz, color='black')
    plt.xlabel('time [$ s $]', fontsize=fntsz, color='black')
    plt.grid(True)
    
    # ** xy-plane
    i = i + 1
    plt.subplot(sbplt+i)
    plt.title(' contribution cerebellum and lf ')
    plt.plot(control_data[:,0], ( control_data[:,6]  ), 'g',label = title_cmd[6] )
    plt.plot(control_data[:,0], (  control_data[:,4] ), 'r',label = title_cmd[4] )

    
    plt.legend()
    plt.ylabel('joint 2 control [$ N $]', fontsize=fntsz, color='black')
    plt.xlabel('time [$ s $]', fontsize=fntsz, color='black')
    plt.grid(True)
    
    plt.show()
    
    plt.figure(fig)
    y_range_limit = []
    sbplt = 211
    i=0
    
    plt.subplot(sbplt+i)
    plt.title(' control input $joint_1$')
    #plt.plot(control_data[:,0], control_data[:,9], 'c',label = title_cmd[9] )
    
    plt.plot(control_data[:,0], ( control_data[:,7] ), 'g',label = title_cmd[7] )
    #plt.plot(joint_data[:,0], joint_data[:,13], 'b',label = title[13])
    
    
    plt.legend()
    plt.ylabel('control input [$ N $] ', fontsize=fntsz, color='black')
    plt.xlabel('time [$ s $]', fontsize=fntsz, color='black')
    plt.grid(True)
    
    # ** xy-plane
    i = i + 1
    plt.subplot(sbplt+i)
    plt.title(' control input $joint_2$')
    plt.plot(control_data[:,0], ( control_data[:,8] ), 'g',label = title_cmd[8] )
    #plt.plot(joint_data[:,0], joint_data[:,14], 'b',label = title[14])
    
    
    plt.legend()
    plt.ylabel('control input [$ N $]  ', fontsize=fntsz, color='black')
    plt.xlabel('time [$ s $]', fontsize=fntsz, color='black')
    plt.grid(True)
    
    plt.show()
    fig = fig +1
    
    
    
