###################################################################################
             # DTU SPIKING CEREB TUNING #
###################################################################################

# -*- coding: utf-8 -*-
"""

"""

__author__ = 'Carlos Corchado Miralles'

import sys
import os
import glob
import matplotlib
import matplotlib.pyplot as plt
import h5py
import numpy as np
import colorama
from colorama import Fore, Back, Style
import pandas as pd


class CerebTestTune:
    """Class contains all tuning functionality"""

    colorama.init()
    matplotlib.rcParams['figure.figsize']=[36.0,24.0]
    default_model = {"model":"dtu_pavia_simple_cereb", 
                     "MF_number": 120,
                     "GR_number": 750,
                     "PC_number": 24, # Must be EVEN
                     "IO_number": 24,  # Must be EVEN
                     "DCN_number": 12} # Must be EVEN} 

    def __init__(self, TEST, RESULTS_PATH, MODULES, JOINTS, MODEL=default_model):
        self.test = TEST
        self.results_path = RESULTS_PATH
        self.test_path_raw = os.path.join(self.results_path, self.test)
        self.test_path_final = os.path.join(self.test_path_raw, "final_files")
        self.modules = MODULES
        self.joints = JOINTS
        self.all_contributions = False
        self.model = MODEL["model"]
        self.MF_number = MODEL["MF_number"]
        self.GR_number = MODEL["GR_number"]
        self.PC_number = MODEL["PC_number"] # Must be EVEN
        self.IO_number = MODEL["IO_number"] # Must be EVEN
        self.DCN_number = MODEL["DCN_number"]# Must be EVEN

    @staticmethod #we can call the method wihout initializing the class
    def read_two_column_file_and_order(self, file_name):
        with open(file_name, 'r') as data:
            population = []
            time = []
            for line in data:
                p = line.split()
                population.append(int(p[0]))
                time.append(float(p[1]))

            #Order both lists depending on the order of the time
            try:
                time, population = zip(*sorted(zip(time, population)))
                print(file_name + "\n")
            except Exception as e:
                print(Style.BRIGHT + Fore.RED + "ERROR: " + Style.RESET_ALL + 
                    Fore.YELLOW + "Error reading file: " + file_name + Style.RESET_ALL + " "+ 
                    Style.BRIGHT + Fore.RED + str(e) + Style.RESET_ALL + "\n")
                time = [1.0]
                population = [-1]

        return time, population

    def merge_threads(self):
        for module in range(self.modules):
            exec("populations = [" + "\"" + "MF_"+str(module)+"_"+"\""+","+
                    "\""+"PC_"+str(module)+"_"+"\"" +","+
                    "\"" +"IO_"+str(module)+"_"+"\"" +","
                    +"\"" +"DCN_"+str(module)+"_"+"\"" +"]")
            for joint in self.joints:
                #list of the main populations
                exec("populations_total_" + str(module) + "_" + joint + "={}")

                for key in populations:
                    exec("populations_total_"+str(module)+"_" + joint + "[" + "\"" + key + joint + "-\"" + "] = []")
                
                key = "GR"
                exec("populations_total_"+str(module)+"_" + joint + "[" + "\"" + key + "-\"" + "] = []")

                #list of groups in main populations
                exec("populations_groups_" + str(module) + "_" + joint + "={}")

                for key in populations:
                    if "MF" in key:
                        for i in ["_des_q", "_des_qd", "_cur_q"]:
                            exec("populations_groups_"+str(module)+"_" + joint + "[" + "\"" + key + joint + i+"-\"" + "] = []")
                    else:
                        for i in ["_pos", "_neg"]:
                            exec("populations_groups_"+str(module)+"_" + joint + "[" + "\"" + key + joint + i+"-\"" + "] = []")

        #Create directory to store the final files
        try:
            self.test_path_final = os.path.join(self.test_path_raw,"final_files")
            os.mkdir(self.test_path_final)
        except Exception as e:
            r = raw_input(Style.BRIGHT + Fore.RED + "WARNING: " + Style.RESET_ALL + Fore.YELLOW + "Erase current final_files? y/n: "+ Style.RESET_ALL)
            if r.lower() == "y":
                for file in os.listdir(self.test_path_final):
                    file_path = os.path.join(self.test_path_final, file)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                #erase files HERE
            else:
                return

        pop_types = ["_total_", "_groups_"]


        for pop_type in pop_types:
            for module in range(self.modules):
                for joint in self.joints:
                    exec("populations = "+"populations"+pop_type+str(module)+"_"+joint)
                    for root,dirs,files in os.walk(self.test_path_raw):
                        for name in files:
                            for pop in populations.keys():
                                if pop in name:
                                    populations[pop].append(os.path.join(root,name))

        #Merge the different files per thread: create one single .gdf file per population
        for pop_type in pop_types:
            for module in range(self.modules):
                for joint in self.joints:
                    exec("populations = "+"populations"+pop_type+str(module)+"_"+joint)
                    for pop in populations.keys():
                        filename = "Spike_Detector_"+pop[:-1]+".gdf"
                            
                        with open(os.path.join(self.test_path_final, filename), 'w') as outfile:
                            for fname in populations[pop]:
                                with open(fname) as infile:
                                    for line in infile:
                                        outfile.write(line)


        #Order each .gdf file per time
        print(Style.BRIGHT + Fore.BLUE + "\nModifying the final files\n" + Style.RESET_ALL)
        i = 1
        for root,dirs,files in os.walk(self.test_path_final):
            for name in files:
                time, population = self.read_two_column_file_and_order(self, os.path.join(root, name))
                with open(os.path.join(root, name), 'w') as final_file:
                    for line in range(0, len(time)):
                        final_file.write(str(population[line]) + "    " + str(time[line])+'\n')
    
    def check_rates(self, ALL_CONTRIBUTIONS=False):
        self.all_contributions = ALL_CONTRIBUTIONS
        print(Style.BRIGHT + Fore.RED + "WARNING: " + Style.RESET_ALL + Fore.YELLOW +"If you want to see the contributions of each group that is part of each population,"+
            " pass True as argument to this method")
        print("Looking into: " + self.test_path_final)
        merge = False
        for item in os.listdir(self.test_path_raw):
            item_path = os.path.join(self.test_path_raw, item)
            if os.path.isdir(item_path):
                if item in "final_files":
                    merge = True

        if not merge:
            print(Style.BRIGHT + Fore.RED + "ERROR: " + Style.RESET_ALL + Fore.YELLOW + "You need to merge_threads before check_rates" + Style.RESET_ALL )
            return
        #try:
        #    SpikeFile = h5py.File(PathDir + "Spikes.h5", "w-")
        #except IOError:
        #    print("Spikes.h5 already exists")

        for joint in self.joints:
            for cells in ["MF", "GR", "PC", "DCN", "IO"]:
                if self.all_contributions:
                    if cells == "MF":
                        contributions = ["", "_cur_q", "_des_q", "_des_qd"]
                    elif cells == "GR":
                        contributions = [""]
                    elif cells in ["PC", "DCN", "IO"]:
                        contributions = ["", "_pos", "_neg"]
                else:
                    contributions = [""]

                for contribution in contributions:
                    Spikes = 0
                    time = []

                    if cells == "GR":
                        Cellfiles = self.test_path_final + "/Spike_Detector_" + cells + ".gdf"
                    else:
                        Cellfiles = self.test_path_final + "/Spike_Detector_" + cells + "_0_" + joint + contribution + ".gdf"

                    exec("CellNumber=" + "self."+cells + "_number")
                    exec("SpikeMatrix_" + cells + "={}")
                    #for cell_n in range(CellNumber):
                    #     exec("SpikeMatrix_" + cells + ".append([])")

                    OpenFiles = glob.glob(Cellfiles)
                    OpenFiles.sort()

                    for count, fl in enumerate(OpenFiles):
                        #if count%10 == 0:
                        #    aux.progressbar(float(count)/float(len(OpenFiles)), 60)
                        with open(fl) as openfileobject:
                            for line in openfileobject:
                                if len(line.split())>1:
                                    exec("list_of_cells = SpikeMatrix_" + cells + ".keys()")
                                    #if the neuron ID is valid
                                    if int(line.split()[0]) >= 0:
                                        if line.split()[0] not in list_of_cells:
                                            exec("SpikeMatrix_" + cells + "[line.split()[0]]=[]")
                                            exec("SpikeMatrix_" + cells + "[line.split()[0]].append(float(line.split()[1]))")
                                        else:
                                            exec("SpikeMatrix_" + cells + "[line.split()[0]].append(float(line.split()[1]))")
                                        Spikes+=1
                                        time.append(float(line.split()[1]))

                            exec("list_of_cells = SpikeMatrix_" + cells + ".keys()")
                            TestTime = float(line.split()[1])

                            max_n_spikes = 0

                            for pop in list_of_cells:
                                exec("Active = len(SpikeMatrix_" + cells + "[pop])")
                                if Active > max_n_spikes:
                                    max_n_spikes=Active

                    s = 0
                    f = 0
                    for i in range(0,len(time)):
                        if time[i] <= (time[-1]*0.25):
                            s+=1
                        if time[i] <= (time[-1]*0.75):
                            f+=1

                    
                    #print("FiringRate_i:" + str(FiringRate_i))
                    #print("FiringRate_f:" + str(FiringRate_f))
                    print("--------------------------")
                    print(Fore.GREEN+Cellfiles)
                    
                    FiringRate   = 1000*Spikes/(TestTime*CellNumber)
                    FiringRate_i = 1000*s/((time[-1]*0.25)*CellNumber)
                    FiringRate_f = 1000*f/((time[-1]*0.75)*CellNumber)

                    print(Style.RESET_ALL)
                    print("Simulation time: " + str(TestTime/1000) + "seconds")
                    print(Style.BRIGHT + Fore.BLUE+ cells + contribution + " Mean Firing Rate START= " + "{0:0.2f}".format(FiringRate_i))
                    print(Style.BRIGHT + Fore.BLUE+ cells + contribution + " Mean Firing Rate END= " + "{0:0.2f}".format(FiringRate_f))
                    print(Style.BRIGHT + Fore.BLUE+ cells + contribution + " Mean Firing Rate TOTAL= " + "{0:0.2f}".format(FiringRate))
                    exec("N_active_cells = len(SpikeMatrix_" + cells + ".keys())")
                    print(Style.BRIGHT + Fore.YELLOW+str(N_active_cells) + " " + cells + contribution + " were active, out of a total number of " + str(CellNumber) + " " + cells)
                    print(Style.RESET_ALL)

    def plot_activity(self, path=None):
        if path is None:
            path=self.test_path_final

        i = 1
        for root,dirs,files in os.walk(path):
            for name in files:
                if "DCN_0_0" in name:
                    print(os.path.join(root, name)+'\n')
                    time, population = self.read_two_column_file_and_order(self, os.path.join(root, name))

                    print(name)
                    print(len(list(set(population))))

                    with open(os.path.join(root, name), 'w') as final_file:
                        for line in range(0, len(time)):
                            final_file.write(str(population[line]) + "    " + str(time[line])+'\n')
                    
                    plt.figure(i)
                    plt.title(name)
                    plt.scatter(time, population)
                    plt.xlabel('Time (milliiseconds)', fontsize=12, color='black')
                    plt.ylabel('Neuron ID', fontsize=12, color='black')
                    plt.grid(True)
                    i += 1
                    plt.show()

    @staticmethod
    def read_datafile(self, file_name):
        # the skiprows keyword is for heading, but I don't know if trailing lines
        # can be specified
        data = np.loadtxt(file_name, delimiter=',', skiprows=1, dtype=float)
        title = np.loadtxt(file_name, delimiter=',', dtype='str')
        return data,title[0,:]
    
    #@staticmethod
    def plot_joint(self, path=None):
        if path is None:
            path = self.test_path_raw
        joint_data, title = self.read_datafile(self, os.path.join(path, 'joint_info.csv'))
        fntsz = 14
        y_range_limit = [-1.5, 1.5]
        # ** Plotting position **
        fig = 1

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
        #print i
        plt.subplot(sbplt+i)
        plt.title('$ error joint_1 $ ')
        plt.plot(joint_data[:,0], joint_data[:,13], 'b',label = title[13])
        #plt.title('$ MSE joint_1 $ ')
        #plt.plot(joint_data[:,0], joint_data[:,13]**2, 'b',label = title[13])
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
        #plt.title('$ MSE joint_2 $ ')
        #plt.plot(joint_data[:,0], joint_data[:,14]**2, 'b',label = title[14])
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

        #mse1 = np.mean((joint_data[:,7] - joint_data[:,1])**2)
        #mse2 = np.mean((joint_data[:,8] - joint_data[:,2])**2)
        mse1_total = np.mean((joint_data[:,13])**2)
        mse1_start = np.mean((joint_data[1:int(len(joint_data[:,1])/4),13])**2)
        mse1_end   = np.mean((joint_data[int(len(joint_data[:,1])/4):,13])**2)
        
        mse2_total = np.mean((joint_data[:,14])**2)
        mse2_start = np.mean((joint_data[1:int(len(joint_data[:,1])/4),14])**2)
        mse2_end   = np.mean((joint_data[int(len(joint_data[:,1])/4):,14])**2)
        
        print "######### JOINT 1 #########"
        print "MSE START joint 1: " + str(mse1_start)
        print "MSE END   joint 1: " + str(mse1_end)
        print "MSE TOTAL joint 1: " + str(mse1_total)

        print "\n"
        print "######### JOINT 2 #########"
        print "MSE START joint 2: " + str(mse2_start)
        print "MSE END   joint 2: " + str(mse2_end)
        print "MSE TOTAL joint 2: " + str(mse2_total)
            
        plt.show()
        fig = fig +1

    def plot_control(self, path=None):
        if path is None:
            path = self.test_path_raw
        control_data, title_cmd = self.read_datafile(self, os.path.join(path, 'command.csv'))
        fntsz = 14
        y_range_limit = [-1.5, 1.5]
        # ** Plotting position **
        fig = 1
        plt.figure(fig)
        y_range_limit = []
        sbplt = 211
        i=0

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





    def get_gdf_freq_data(self, file_path, bucket_length = 100):
        '''Input:
            file_path: path to GDF file
            bucket_length: buckets in ms, that are analysed individually
        Output:
            data dict with keys
                "center_time": middle of bucket in time
                "bucket_start_time": start time of bucket
                "bucket_end_time": end time of bucket
                "avg_cell_spike_freq": average cell spike frequency during bucket
                "unique_active_cells": Number of unique cells that spiked in bucket
                "spike_count": how many spikes during the bucket
        '''

        spike_time_list, population_id = self.get_sorted_gdf_data(file_path)
        total_cells = len(list(set(population_id)))

        bucket_spike_count = [0]
        last_time = 0

        spike_data = {  "center_time":[],
                        "bucket_start_time": [],
                        "bucket_end_time":[],
                        "avg_cell_spike_freq": [],
                        "unique_active_cells":[],
                        "spike_count":[]
                     }

        until = spike_time_list[0] + bucket_length
        earlier_spikes = 0
        for spike_count, spike_time in enumerate(spike_time_list):

            if spike_time > until:
                spikes_in_current_bucket = spike_count - earlier_spikes

                spike_data["center_time"].append( spike_time - (bucket_length/2.) )
                spike_data["bucket_start_time"].append( until - bucket_length )
                spike_data["bucket_end_time"].append( until)

                # Count number of unique cells in bucket
                pop_list = population_id[ earlier_spikes : spike_count]
                total_active_cells = len(pop_list)
                unique_active_cells = len(list(set(pop_list)))

                spike_data["spike_count"].append( spikes_in_current_bucket )
                spike_data["avg_cell_spike_freq"].append( (1000. * spikes_in_current_bucket) / (bucket_length * total_cells) )
                spike_data["unique_active_cells"].append(unique_active_cells)

                #print "Spikes in bucket:", spikes_in_current_bucket
                #print "Unique active cells:", unique_active_cells
                #print "Avg spike freq:", spike_data["avg_cell_spike_freq"][-1]

                until += bucket_length
                earlier_spikes = spike_count

        return spike_data



    def get_sorted_gdf_data(self, gdf_filename):

        with open(gdf_filename, 'r') as data:
            population = []
            time = []
            for line in data:
                p = line.split()
                population.append(int(p[0]))
                time.append(float(p[1]))

            #Order both lists depending on the order of the time
            try:
                time, population = zip(*sorted(zip(time, population)))

            except Exception as e:
                print("ERROR: Error reading file: " + gdf_filename + " " + str(e) + "\n")
                time = [1.0]
                population = [-1]

        return time, population



    def plot_gdf_freq_evolution(self, population, bucket_length = 100):


        file_path = self.test_path_final + "/Spike_Detector_" + population + ".gdf"

        spike_data = self.get_gdf_freq_data(file_path, bucket_length)
        df = pd.DataFrame(data=spike_data)

        fig_size = (10,5)#(14, 8)
        fig_dpi = 100#300

        freq_max = df["avg_cell_spike_freq"].max()
        freq_min = df["avg_cell_spike_freq"].min()

        fig = plt.figure(figsize=fig_size)#(figsize=fig_size, dpi=fig_dpi)
        ax = fig.gca()

        df.plot(x="bucket_start_time",#x="center_time",
                y="avg_cell_spike_freq",
                label="Avg Cell Spike Freq",
                kind="line",#"bar",
                color="b",
                ax=ax)#, style='o', ms=3)

        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Frequency [Hz]")
        #ax.set_xlim(3,3.2)
        ax.set_ylim(freq_min * 0.999,freq_max * 1.001)
        plt.title(file_path.split("/")[-1] + " (bucket length: " + str(bucket_length) + "ms)")
        plt.grid()
        plt.show()
