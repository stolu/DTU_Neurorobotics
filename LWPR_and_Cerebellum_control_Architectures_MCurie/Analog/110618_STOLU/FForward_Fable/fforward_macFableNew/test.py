"""
Created on Feb 11 2015
__author__ = 'silvia'
Test the function setModuleMotorTorque with new firmware

"""
import sys, time, random
import numpy as np
import random
sys.path.append("../../Fable_api_PY_NEW/python/api")

from fableAPI import FableAPI

print("go")

ModuleID = 21

api = FableAPI()
api.setup(1)

# Variable definitions
dt = 0.5
n = 100
vel = np.zeros((2), dtype=np.double)

for j in range(n):
  #pos = random.randint(0, 90)
  torque = random.randint(-80, 80)
  # Control in motor torques 
  api.setModuleMotorTorque(ModuleID, 0, -torque, 1, -torque, ack=False)
  #api.setModuleMotorPosition(ModuleID, 1, pos)
  vel[0] = api.getModuleMotorSpeed(ModuleID, 0)
  
  #api.setModuleMotorPosition(ModuleID, 0, pos)
  vel[1] = api.getModuleMotorSpeed(ModuleID, 1)
  
  #print("pos:", pos) 
  print("vel:", vel)
  
  print(j)
  
  #print("torques: ", torque)
  #time.sleep(dt)
