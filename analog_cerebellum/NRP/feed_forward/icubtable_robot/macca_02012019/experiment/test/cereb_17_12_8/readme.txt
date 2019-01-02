cerebellum.ltpPF_PC_max = 1. * 10**(-3) # -4
    cerebellum.ltdPF_PC_max = 1. * 10**(-3) # -5
    #inh
    cerebellum.ltpPC_DCN_max = 1. * 10**(-4) #-2  # 4
    cerebellum.ltdPC_DCN_max = 1. * 10**(-4) #-3  # 3
    #exc
    cerebellum.ltpMF_DCN_max = 1. * 10**(-4) #-3
    cerebellum.ltdMF_DCN_max = 1. * 10**(-4) #-2
    #exc
    cerebellum.ltpIO_DCN_max = 1. * 10**(-4) #-4
    cerebellum.ltdIO_DCN_max = 1. * 10**(-5) #-3
     
    #self.alpha = 1
    cerebellum.alphaPF_PC_pos  = 170.  #self.ltd_max / self.ltp_max 50, 7
    cerebellum.alphaPF_PC_vel  = 170.    
    cerebellum.alphaPC_DCN     = 2. # self.ltd_PC_DCN_max / self.ltp_PC_DCN_max
    cerebellum.alphaMF_DCN     = 2. # self.ltd_PC_DCN_max / self.ltp_PC_DCN_max
    cerebellum.alphaIO_DCN     = 200.

for idx in range( init_joint_idx , n_joints.value):
                controlcommand.value.data[idx] = static_control.PID[idx] + DCNcommand.value.data[idx]- cerebellum.output_IO_DCN[idx]# cerebellum.output_pc[idx] + cerebellum.output_pc_vel[idx]#+ DCNcommand.value.data[idx]
