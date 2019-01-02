# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 15:11:40 2018

@author: silvia-neurorobotics
"""
import numpy as np
import matplotlib.pyplot as plt
# Cerebellum plasticities kerkel
fntsz = 12
sample = 1000
x = 0.
dt = 0.02
min_alpha = 2
max_alpha = 50
delta_alpha = 10
teach_signal = np.linspace(.0, 1.0, num = sample)
pc_signal = np.linspace(.0, 1.0, num = sample)
dcn_signal = np.linspace(.0, 1.0, num = sample)

# *** Plasticity ***
#exc
ltpPF_PC_max = 1. * 10**(-4) # -4
ltdPF_PC_max = 1. * 10**(-4) # -5
#inh
ltpPC_DCN_max = 1. * 10**(-4) #-2  # 4
ltdPC_DCN_max = 1. * 10**(-4) #-3  # 3
#exc
ltpMF_DCN_max = 1. * 10**(-4) #-3
ltdMF_DCN_max = 1. * 10**(-4) #-2
#exc
ltpIO_DCN_max = 1. * 10**(-4) #-4
ltdIO_DCN_max = 1. * 10**(-5) #-3
 
#self.alpha = 1
alphaPF_PC  = 2.  #self.ltd_max / self.ltp_max 50, 7
alphaPF_PC_vel  = 2.    
alphaPC_DCN     = 2. # self.ltd_PC_DCN_max / self.ltp_PC_DCN_max
alphaMF_DCN     = 2. # self.ltd_PC_DCN_max / self.ltp_PC_DCN_max
alphaIO_DCN     = 10.


fig_n = 0

f = teach_signal[0] + (teach_signal[1] if min_alpha == 2 else 0)
print(f)

f = teach_signal[0] +[  0.,teach_signal[1]  ][min_alpha == 2 ]
print(f)

# ----------------------------------------------------------------------------#
# ---------- *** Parallel Fibers - Purkinje with IO modulation *** -----------#
# ----------------------------------------------------------------------------# 

print("\n *** PF-PC plasticity ***")
fig_n += 1 
fig = plt.figure(fig_n)
sbplt = 211    
plt.subplot(sbplt)
ltpPF_PC_max = 1. * 10**(-4) # -4
ltdPF_PC_max = 1. * 10**(-4) # -5
plt.title('$ ltp_{pf-pc,max}=%f \\ ltp_{pf-pc,max}=%f $ ' %(ltpPF_PC_max,ltdPF_PC_max) )
for alphaPF_PC in range( min_alpha , max_alpha ,delta_alpha):
    w_pf_pc = ltpPF_PC_max /( teach_signal + 1.)**( alphaPF_PC) - ltdPF_PC_max*teach_signal
    plt.plot(teach_signal,w_pf_pc, label = r"$\alpha=%i$" %alphaPF_PC)
plt.ylabel(r"$w_{pf-pc}$" , fontsize=fntsz, color='black')
plt.xlabel(r"$IO$" , fontsize=fntsz, color='black')
plt.legend()
plt.grid()
#plt.show()


sbplt += 1    
plt.subplot(sbplt)
ltpPF_PC_max = 1. * 10**(-3) # -4
ltdPF_PC_max = 1. * 10**(-5) # -5
plt.title('$ ltp_{pf-pc,max}=%f \\ ltp_{pf-pc,max}=%f $ ' %(ltpPF_PC_max,ltdPF_PC_max) )
for alphaPF_PC in range( min_alpha , max_alpha ,delta_alpha):
    w_pf_pc = ltpPF_PC_max /( teach_signal + 1.)**( alphaPF_PC) - ltdPF_PC_max*teach_signal
    plt.plot(teach_signal,w_pf_pc, label = r"$\alpha=%i$" %alphaPF_PC)
plt.ylabel(r"$w_{pf-pc}$" , fontsize=fntsz, color='black')
plt.xlabel(r"$IO$" , fontsize=fntsz, color='black')
plt.legend()
plt.grid()
#plt.show()



# ----------------------------------------------------------------------------#
# -------------- *** Purkinje - Deep Cerebellar Nuclei *** -------------------#
# ----------------------------------------------------------------------------#  
print("\n *** PC-DCN plasticity ***")
fig_n += 1 
fig = plt.figure(fig_n)
sbplt = 211
ltpPC_DCN_max = 1. * 10**(-4) #-2  # 4
ltdPC_DCN_max = 1. * 10**(-4) 
plt.subplot(sbplt)
plt.title('$ ltp_{pc-dcn,max}=%f \\ ltp_{pc-dcn,max}=%f $ ' %(ltpPC_DCN_max,ltdPC_DCN_max) )
for alphaPC_DCN in range( min_alpha , max_alpha ,delta_alpha):
    w_pc_dcn_max_dcn = ( ltpPC_DCN_max*(pc_signal**alphaPC_DCN) )/( 1. + 1.)**alphaPC_DCN - ltdPC_DCN_max*( pc_signal  )           
    
    plt.plot(pc_signal,w_pc_dcn_max_dcn,label = r"$ max_{dcn} \alpha=%i$" %alphaPC_DCN)
    plt.grid()
    plt.ylabel(r"$w_{pc-dcn}$" , fontsize=fntsz, color='black')
    plt.xlabel(r"$PC$" , fontsize=fntsz, color='black')
    plt.legend()
for alphaPC_DCN in range( min_alpha , max_alpha ,2):
    w_pc_dcn_min_dcn = ( ltpPC_DCN_max*(pc_signal**alphaPC_DCN) )/( 1. + .01)**alphaPC_DCN - ltdPC_DCN_max*( pc_signal  ) 
    plt.subplot(sbplt+1)     
    plt.plot(pc_signal,w_pc_dcn_min_dcn ,label = r"$ max_{dcn} \alpha=%i$" %alphaPC_DCN)
    plt.grid()
    plt.ylabel(r"$w_{pc-dcn}$" , fontsize=fntsz, color='black')
    plt.xlabel(r"$PC$" , fontsize=fntsz, color='black')
    plt.legend()

#plt.show()


fig_n += 1 
fig = plt.figure(fig_n)
sbplt = 211
ltpPC_DCN_max = 1. * 10**(-4) #-2  # 4
ltdPC_DCN_max = 1. * 10**(-5) 
plt.subplot(sbplt)
plt.title('$ ltp_{pc-dcn,max}=%f \\ ltp_{pc-dcn,max}=%f $ ' %(ltpPC_DCN_max,ltdPC_DCN_max) )
for alphaPC_DCN in range( min_alpha , max_alpha ,delta_alpha):
    w_pc_dcn_max_dcn = ( ltpPC_DCN_max*(pc_signal**alphaPC_DCN) )/( 1. + 1.)**alphaPC_DCN - ltdPC_DCN_max*( pc_signal  )           
    
    plt.plot(pc_signal,w_pc_dcn_max_dcn,label = r"$ max_{dcn} \alpha=%i$" %alphaPC_DCN)
    plt.grid()
    plt.ylabel(r"$w_{pc-dcn}$" , fontsize=fntsz, color='black')
    plt.xlabel(r"$PC$" , fontsize=fntsz, color='black')
    plt.legend()
for alphaPC_DCN in range( min_alpha , max_alpha ,delta_alpha):
    w_pc_dcn_min_dcn = ( ltpPC_DCN_max*(pc_signal**alphaPC_DCN) )/( 1. + .01)**alphaPC_DCN - ltdPC_DCN_max*( pc_signal  ) 
    plt.subplot(sbplt+1)     
    plt.plot(pc_signal,w_pc_dcn_min_dcn ,label = r"$ min_{dcn} \alpha=%i$" %alphaPC_DCN)
    plt.grid()
    plt.ylabel(r"$w_{pc-dcn}$" , fontsize=fntsz, color='black')
    plt.xlabel(r"$PC$" , fontsize=fntsz, color='black')
    plt.legend()

# MF - DCN
print("\n *** MF-DCN plasticity ***")
fig_n += 1 
fig = plt.figure(fig_n)
sbplt = 211
ltpMF_DCN_max = 1. * 10**(-4) #-2  # 4
ltdMF_DCN_max = 1. * 10**(-4) 
plt.subplot(sbplt)
plt.title('$ ltp_{mf-dcn,max}=%f \\ ltp_{mf-dcn,max}=%f $ ' %(ltpMF_DCN_max,ltdMF_DCN_max) )

for alphaMF_DCN in range( min_alpha , max_alpha ,delta_alpha):
    w_mf_dcn = ltpMF_DCN_max/( pc_signal  + 1.)**alphaMF_DCN - ltdMF_DCN_max*pc_signal 
    plt.plot(pc_signal,w_mf_dcn, label = r"$alpha=%i$" %alphaMF_DCN)

plt.ylabel(r"$w_{mf-dcn}$" , fontsize=fntsz, color='black')
plt.xlabel(r"$PC$" , fontsize=fntsz, color='black')
plt.legend()
plt.grid()



ltpMF_DCN_max = 1. * 10**(-4) #-2  # 4
ltdMF_DCN_max = 1. * 10**(-5) 
sbplt += 1
plt.subplot(sbplt)
plt.title('$ ltp_{mf-dcn,max}=%f \\ ltp_{mf-dcn,max}=%f $ ' %(ltpMF_DCN_max,ltdMF_DCN_max) )

for alphaMF_DCN in range( min_alpha , max_alpha ,delta_alpha):
    w_mf_dcn = ltpMF_DCN_max/( pc_signal  + 1.)**alphaMF_DCN - ltdMF_DCN_max*pc_signal 
    plt.plot(pc_signal,w_mf_dcn, label = r"$alpha=%i$" %alphaMF_DCN)

plt.ylabel(r"$w_{mf-dcn}$" , fontsize=fntsz, color='black')
plt.xlabel(r"$PC$" , fontsize=fntsz, color='black')
plt.legend()
plt.grid()
#plt.show()


# IO - DCN
print("\n *** IO-DCN plasticity ***")
fig_n += 1 
fig = plt.figure(fig_n)
sbplt = 211

ltpIO_DCN_max = 1. * 10**(-4) #-2  # 4
ltdIO_DCN_max = 1. * 10**(-4) 
plt.subplot(sbplt)
plt.title('$ ltp_{io-dcn,max}=%f \\ ltp_{io-dcn,max}=%f $ ' %(ltpIO_DCN_max,ltdIO_DCN_max) )

for alphaIO_DCN in range( min_alpha , max_alpha ,delta_alpha):
    w_io_dcn  =  ltpIO_DCN_max*teach_signal - ltdIO_DCN_max/( (teach_signal+ 1.)**alphaIO_DCN  ) 
    plt.plot(teach_signal,w_io_dcn, label = r"$alpha=%i$" %alphaIO_DCN)
plt.ylabel(r"$w_{io-dcn}$" , fontsize=fntsz, color='black')
plt.xlabel(r"$IO$" , fontsize=fntsz, color='black')
plt.legend()
plt.grid()
            

ltpIO_DCN_max = 1. * 10**(-5) #-2  # 4
ltdIO_DCN_max = 1. * 10**(-4) 
sbplt += 1
plt.subplot(sbplt)
plt.title('$ ltp_{io-dcn,max}=%f \\ ltp_{io-dcn,max}=%f $ ' %(ltpIO_DCN_max,ltdIO_DCN_max) )

for alphaIO_DCN in range( min_alpha , max_alpha ,delta_alpha):
    w_io_dcn  =  ltpIO_DCN_max*teach_signal - ltdIO_DCN_max/( (teach_signal+ 1.)**alphaIO_DCN  ) 
    plt.plot(teach_signal,w_io_dcn, label = r"$alpha=%i$" %alphaIO_DCN)
plt.ylabel(r"$w_{io-dcn}$" , fontsize=fntsz, color='black')
plt.xlabel(r"$IO$" , fontsize=fntsz, color='black')
plt.legend()
plt.grid()
plt.show()            