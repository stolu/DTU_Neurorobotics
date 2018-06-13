#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""============================================================
* File Name     : fable_multiprocess.py
* Purpose 	    : Control several modules of Fable at the same
                  time, using one single dongle and from just
                  one script
* Creation Date : 06-12-2016
* Last Modified : Tue 13 Dec 2016 06:18:24 PM CET
__author__  	= "Ismael Baira Ojeda"
__credits__ 	= ["Ismael Baira Ojeda", "Silvia Tolu"]
__maintainer__ 	= "Ismael Baira Ojeda"
__email__ 	= ["i.bairao@gmail.com", "iboj@elektro.dtu.dk"]

============================================================"""

import sys, time, math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

sys.path.append("../../../Fable_api_PY_NEW/python/api")
sys.path.append("../../../Robot_toolboxPy/robot")
sys.path.append("../../fforward_macFableNew")

from lwpr import *
from AFEL_LWPRandC_class_2_modules import *
from Robot import *
from fableAPI import *
from simfable import *
from dynamics import *
from datetime import datetime, date
import serial, time, random, math, threading
from threading import Thread


def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper

class AFEL():

    def __init__(self, module_id):
        self.ModuleID = module_id
        self.n_iter = 20500
        self.njoints = 2

        # self.a = 1.0
        # self.b = 7.5
        # self.c = 6.4 #6.4
        # self.ki = [self.a, self.a]
        # self.kp = [self.b, self.b]
        # self.kv = [self.c, self.c]

        self.ki = [1.0, 1.0]
        self.kp = [7.5, 7.5]
        self.kv = [6.4, 6.4]
        self.grav = [0, 0, 9.81]

        # Variable definitions
        self.A1      = 400 #700 1100
        self.A2      = 400 #700
        self.phase   = math.pi / 2
        self.dt      = 0.01
        self.t0      = 0

        self.q1   = [0 for k in range(self.n_iter + 1)]
        self.q2   = [0 for a in range(self.n_iter + 1)]
        self.q1d  = [0 for c in range(self.n_iter + 1)]
        self.q2d  = [0 for v in range(self.n_iter + 1)]
        self.q1dd = [0 for o in range(self.n_iter + 1)]
        self.q2dd = [0 for l in range(self.n_iter + 1)]
        self.x    = [0 for k in range(self.n_iter + 1)]
        self.y    = [0 for k in range(self.n_iter + 1)]

        self.posr = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.velr = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.accr = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.D    = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.erra = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.errp = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.errv = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.ea   = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.ep   = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.ev   = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)

        self.torquesLF   = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.torqLWPR    = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.Ctorques    = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.DCNtorques    = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.torquestot  = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)

        self.posC    = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.velC    = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.pLWPR   = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)
        self.vLWPR   = np.zeros((self.njoints, self.n_iter + 1), dtype=np.double)

    def __del__(self):
        print("Class object for Module_{} - Destroyed".format(self.ModuleID))

    # Circular trajectory
    def calc_trajectory_circle(self, fab):
        for i in range(self.n_iter):
            self.q1dd[i] = self.A1 * math.sin(2 * math.pi * self.t0)
            self.q1d[i]  = -1/(2*math.pi) * self.A1 * math.cos(2 * math.pi * self.t0)
            self.q1[i]   = -math.pow((1/(2*math.pi)),2) * self.A1 * math.sin(2 * math.pi * self.t0)

            self.q2dd[i] = self.A2 * math.sin(2 * math.pi * self.t0 + self.phase)
            self.q2d[i]  = -1/(2*math.pi) * self.A2 * math.cos(2 * math.pi * self.t0 + self.phase)
            self.q2[i]   = -math.pow((1/(2*math.pi)),2) * self.A2 * math.sin(2 * math.pi * self.t0 + self.phase)

            self.t0 += self.dt
            # Cartesian coordinates
            # self.x[i]    = fab.l1 * math.cos(self.q1[i]) + fab.l2 * math.cos((self.q1[i] + self.q2[i]))
            # self.y[i]    = fab.l1 * math.sin(self.q1[i]) + fab.l2 * math.sin((self.q1[i] + self.q2[i]))

        return self.q1, self.q2, self.q1d, self.q2d, self.q1dd, self.q2dd#, self.x, self.y

    # Eight figure trajectory
    def calc_trajectory_8(self, fab):
        self.A1    = 5
        self.phase = math.pi / 2
        for i in range(self.n_iter):
            self.q1dd[i] = self.A1 * math.sin(2 * math.pi * self.t0)
            self.q1d[i]  = ((-1 / 2) * math.pi) * self.A1 * math.cos(2 * math.pi * self.t0)
            self.q1[i]   = (-math.pow(((1 / 2) * math.pi), 2)) * self.A1 * math.sin(2 * math.pi * self.t0)

            self.q2dd[i] = self.A1 * math.cos(4 * math.pi * self.t0 + math.pi / 2)
            self.q2d[i]  =((1 / 2) * math.pi)  * self.A1 * math.sin(4 * math.pi * self.t0 + math.pi / 2)
            self.q2[i]   = (-math.pow(((1 / 2) * math.pi), 2)) * self.A1 * math.cos(4 * math.pi * self.t0 + math.pi / 2)

            self.t0 += self.dt
            # Cartesian coordinates

        return self.q1, self.q2, self.q1d, self.q2d, self.q1dd, self.q2dd

    # Circular trajectory
    def calc_trajectory_crawl(self, fab):
        self.phase = math.pi / 2
        for i in range(self.n_iter):
            self.q1dd[i] = -self.A1 * math.sin(2 * math.pi * self.t0)
            self.q1d[i]  = 1/(2*math.pi) * self.A1 * math.cos(2 * math.pi * self.t0)
            self.q1[i]   = math.pow((1/(2*math.pi)),2) * self.A1 * math.sin(2 * math.pi * self.t0)

            self.q2dd[i] = -self.A2 * math.sin(2 * math.pi * self.t0 + self.phase)
            self.q2d[i]  = 1/(2*math.pi) * self.A2 * math.cos(2 * math.pi * self.t0 + self.phase)
            self.q2[i]   = math.pow((1/(2*math.pi)),2) * self.A2 * math.sin(2 * math.pi * self.t0 + self.phase)

            self.t0 += self.dt
            # Cartesian coordinates
            # self.x[i]    = fab.l1 * math.cos(self.q1[i]) + fab.l2 * math.cos((self.q1[i] + self.q2[i]))
            # self.y[i]    = fab.l1 * math.sin(self.q1[i]) + fab.l2 * math.sin((self.q1[i] + self.q2[i]))

        return self.q1, self.q2, self.q1d, self.q2d, self.q1dd, self.q2dd#, self.x, self.y


    @threaded
    def run_test(self, mlcj, api, fab, fab17):
        print('Test-thread for Module_{}: starting...'.format(self.ModuleID))
        api.setModuleMotorPosition(self.ModuleID, 0, self.q1[0])
        api.setModuleMotorPosition(self.ModuleID, 1, self.q2[0])
        time.sleep(1)
        end_time = time.time()

        for j in range(self.n_iter):
            for i in range(self.njoints):
                # Feedback error learning
                if j > 1:
                    self.D[i, j] = self.D[i, j - 1] + (self.erra[i, j]) * self.ki[i] + (self.errp[i, j] * (self.kp[i])) + (self.errv[i, j] * (self.kv[i]))
                else:
                    self.D[i, j] = (self.erra[i, j]) * self.ki[i] + (self.errp[i, j] * (self.kp[i])) + (self.errv[i, j] * (self.kv[i]))

            tau = rne(fab17, [self.q1[j], self.q2[j]], [0, 0], [1, 1], [0, 0, 0])

            self.torquesLF[0, j] = tau[0, 0] * self.D[0, j]
            self.torquesLF[1, j] = tau[0, 1] * self.D[1, j]

            # Predictions
            (self.torqLWPR[:, j], self.Ctorques[:, j], self.DCNtorques[:, j]) = mlcj.ML_prediction(np.array([self.q1[j],            self.q2[j],
                                                                                                             self.q1d[j],           self.q2d[j],
                                                                                                             self.posr[0, j],       self.posr[1, j]]), self.torquesLF[:, j])

            self.torquestot[0, j] = self.torquesLF[0, j] + self.torqLWPR[0, j] + self.Ctorques[0, j]
            self.torquestot[1, j] = self.torquesLF[1, j] + self.torqLWPR[1, j] + self.Ctorques[1, j]

            # self.torquestot[0, j] = self.torquesLF[0, j] + self.DCNtorques[0, j]
            # self.torquestot[1, j] = self.torquesLF[1, j] + self.DCNtorques[1, j]

            # Avoid torques higher than 100
            if self.torquestot[0, j] > 100.0:
                self.torquestot[0, j] = 100.0
            if self.torquestot[1, j] > 100.0:
                self.torquestot[1, j] = 100.0
            # Avoid torques smaller than -100
            if self.torquestot[0, j] < -100.0:
                self.torquestot[0, j] = -100.0
            if self.torquestot[1, j] < -100.0:
                self.torquestot[1, j] = -100.0

            #print("j1:            ", j)
            # print("torquesLF:    ", self.torquesLF[:, j])
            # print("torques LWPR: ", self.torqLWPR[:, j])
            # print("torquestot:   ", self.torquestot[:, j])
            # print("C torques:    ", self.Ctorques[:, j])

            # Control in motor torques
            api.setModuleMotorTorque(self.ModuleID, 0, -self.torquestot[0, j], 1, -self.torquestot[1, j], ack=False)

            t = (time.time() - end_time)

            # Receive feedback positions from motors
            self.posr[0, j+1] =  api.getModuleMotorPosition(self.ModuleID, 0)       # degrees
            self.posr[1, j+1] =  api.getModuleMotorPosition(self.ModuleID, 1)
            # self.velr[0, j+1] = -api.getModuleMotorSpeed(self.ModuleID, 0)        # r.p.m     # (posr[0,j+1] - posr[0,j] / 2*t)  #
            # self.velr[1, j+1] = -api.getModuleMotorSpeed(self.ModuleID, 1)        # (posr[1,j+1] - posr[1,j] / 2*t)  #
            self.velr[0, j+1] = -6 * api.getModuleMotorSpeed(self.ModuleID, 0)      # grad/s
            self.velr[1, j+1] = -6 * api.getModuleMotorSpeed(self.ModuleID, 1)
            # Cambio a rad/s: r.p.m * (2pi / 1rev) * (1min / 60s) => 1 r.p.m = 0.104 rad/s => 1 rad/s = 57.295779 grad/s
            # 1 r.p.m = 6 grad/s

            print("Module_{} ".format(self.ModuleID), "- j: ", j, "- t: ", t)

            # Compute errors
            self.errp[0, j + 1] = (self.q1[j] - self.posr[0, j+1])
            self.errp[1, j + 1] = (self.q2[j] - self.posr[1, j+1])
            self.errv[0, j + 1] = (self.errp[0, j+1] - self.errp[0, j]) / t         # grad/s        # (q1d[j] - velr[0, j+1]) #
            self.errv[1, j + 1] = (self.errp[1, j+1] - self.errp[1, j]) / t         # (q2d[j] - velr[1, j+1]) #
            self.erra[0, j + 1] = 0.22                                              # q1dd[j] #erra[0, j] + (errp[0, j+1] * dt)
            self.erra[1, j + 1] = 0.22                                              # q2dd[j]

            # print("errp: ", self.errp[:, j + 1])
            # print("errv: ", self.errv[:, j + 1])
            # print("erra: ", self.erra[:, j + 1])
            # print("veld: ", [self.q1d[j], self.q2d[j]])
            # print("velr: ", self.velr[:, j+1])
            # print("posd: ", [self.q1[j], self.q2[j]])
            # print("posr: ", self.posr[:, j+1])

            # Update models
            mlcj.ML_update(np.array([self.q1[j],      self.q2[j],      self.q1d[j],     self.q2d[j],
                                     self.posr[0, j], self.posr[1, j]
                                     ]), (self.torquestot[:, j]))

            mlcj.ML_rfs()
            print("\n\n")
            end_time = time.time()

        # Save with Matlab compatibility
        now = datetime.now()
        scipy.io.savemat('TestAFEL_multithread_fab{0}_{1}_{2}rfs.mat'.format(self.ModuleID, now.strftime('%d-%m-%Y_%H:%M'), mlcj.model[0].num_rfs[0]),
                          dict(q1 = self.q1, q2 = self.q2, posr1 = self.posr[0], posr2 = self.posr[1], errp1 = self.errp[0], errp2 = self.errp[1],
                               torquesLF1 = self.torquesLF[0], torquesLF2 = self.torquesLF[1], torqLWPR1 = self.torqLWPR[0], torqLWPR2 = self.torqLWPR[1],
                               Ctorques1 = self.Ctorques[0], Ctorques2 = self.Ctorques[1], DCNtorques1 = self.DCNtorques[0, j], DCNtorques2 = self.DCNtorques[1, j], torquestot1 = self.torquestot[0], torquestot2 = self.torquestot[1],
                               weight_mod = mlcj.weights_mod))

        print('Thread-test for Module_{}: finishing'.format(self.ModuleID))
        return True



if __name__ == '__main__':
    njoints = 2                     # 2-DoF Fable modules
    module_id_2 = 80
    module_id_1 = 74
    # module_id_3 = 80

    # Class MLandC object for every joint link of each module
    mlcj_1  = MLandC(6, njoints)
    mlcj_2  = MLandC(6, njoints)
    # mlcj_3  = MLandC(10, njoints)

    # Fable api
    grav    = [0, 0, 9.81]
    api     = FableAPI()
    fab     = moduleFable()
    fab17   = fab.Fab()
    api.setup(1)

    # AFEL objects
    afel_1 = AFEL(module_id_1)
    afel_2 = AFEL(module_id_2)
    # afel_3 = AFEL(module_id_3)

    # Generation of Trajectories
    afel_1.calc_trajectory_8(fab)
    afel_2.calc_trajectory_8(fab)  # crawl(fab)
    # afel_3.calc_trajectory_crawl(fab)

    # afel_1.calc_trajectory_crawl(fab)
    # afel_2.calc_trajectory_crawl(fab)
    # # afel_3.calc_trajectory_crawl(fab)

    # Plot desired trajectories
    # plt.figure(1)
    # plt.plot(afel_1.q1, afel_1.q2, label = "Module 1")
    # plt.plot(afel_2.q1, afel_2.q2, label = "Module 2")
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    # # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # plt.title('Desired trajectories')
    # plt.ylabel('y (cm)')
    # plt.xlabel('x (cm)')
    # plt.show()

    # Multithread handling
    handle_1 = afel_1.run_test(mlcj_1, api, fab, fab17)
    handle_2 = afel_2.run_test(mlcj_2, api, fab, fab17)
    # handle_3 = afel_3.run_test(mlcj_3, api, fab, fab17)

    handle_1.join()
    handle_2.join()
    # handle_3.join()

    # # Plot desired trajectories
    # plt.figure(1)
    # plt.plot(afel_1.q1, afel_1.q2, label = "Module 1")
    # plt.plot(afel_2.q1, afel_2.q2, label = "Module 2")
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    # # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # plt.title('Desired trajectories')
    # plt.ylabel('y (cm)')
    # plt.xlabel('x (cm)')
    # plt.show()

    # Termination of api and class usage
    del afel_1, afel_2
    time.sleep(1)
    print("Test - Done...")
    api.sleep(1)
    api.terminate()
