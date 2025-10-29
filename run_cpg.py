# SPDX-FileCopyrightText: Copyright (c) 2022 Guillaume Bellegarda. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2022 EPFL, Guillaume Bellegarda

""" Run CPG """

import time
import numpy as np
import matplotlib

# adapt as needed for your system
# from sys import platform
# if platform =="darwin":
#   matplotlib.use("Qt5Agg")
# else:
#   matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
from env.hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv

ADD_CARTESIAN_PD = False
TIME_STEP = 0.001
foot_y = 0.0838 # this is the hip length 
sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

env = QuadrupedGymEnv(render=True,              # visualize
                    on_rack=True,              # useful for debugging! 
                    isRLGymInterface=False,     # not using RL
                    time_step=TIME_STEP,
                    action_repeat=1,
                    motor_control_mode="TORQUE",
                    add_noise=False,    # start in ideal conditions
                    # record_video=True
                    )

# initialize Hopf Network, supply gait
cpg = HopfNetwork(time_step=TIME_STEP)

TEST_STEPS = int(10 / (TIME_STEP))
t = np.arange(TEST_STEPS)*TIME_STEP

# [TODO] initialize data structures to save CPG and robot states

############## Sample Gains
# joint PD gains
kp=np.array([100,100,100])
kd=np.array([2,2,2])

# Cartesian PD gains
kpCartesian = np.diag([500]*3)
kdCartesian = np.diag([20]*3)

for j in range(TEST_STEPS):
  # initialize torque array to send to motors
  action = np.zeros(12) 

  # get desired foot positions from CPG 
  xs,zs = cpg.update()

  # [TODO] get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
  q = env.robot.GetMotorAngles()
  dq = env.robot.GetMotorVelocities() # [FR_HIP, FR_THIGH, FR_CALF, FL_HIP, FR_THIGH, FL_CALF, FL_HIP, RL_THIGH, RL_CALF, RR_HIP, RR_THIGH, RR_CALF,]

  # loop through desired foot positions and calculate torques
  for current_leg in range(4):
    # initialize torques for legi
    tau = np.zeros(3)

    # get desired foot i pos (xi, yi, zi) in leg frame
    leg_xyz = np.array([xs[current_leg], sideSign[current_leg] * foot_y, zs[current_leg]])

    # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
    # [TODO] MATAS done
    leg_q = env.robot.ComputeInverseKinematics(legID=current_leg, xyz_coord=leg_xyz)

    # Add joint PD contribution to tau for leg i (Equation 4)
     # [TODO] MATAS done
    # τ_joint = Kp_joint(qd − q) + Kd_joint(dqd − dq) 
    dq_i = dq[current_leg*3 : current_leg*3+3]
    q_i = q[current_leg*3 : current_leg*3+3]
    tau += kp @ (leg_q - q_i) + kd @ (-dq_i)

    # add Cartesian PD contribution
    if ADD_CARTESIAN_PD:
      # Get desired xyz position in leg frame (use ComputeJacobianAndPosition with the joint angles you just found above)
      # [TODO] MATAS done
      _, leg_pd = env.robot.ComputeJacobianAndPosition(current_leg, specific_q=leg_q)

      # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
      # [TODO] Matas done
      J, leg_p = env.robot.ComputeJacobianAndPosition(current_leg)

      # Get current foot velocity in leg frame (Equation 2)
      # [TODO] MATAS done
      leg_dp = J @ dq_i

      # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
       # [TODO] MATAS done
      tau += kpCartesian @ (leg_pd - leg_p) + kdCartesian @ (-leg_dp)

    # Set tau for legi in action vector
    action[3*current_leg:3*current_leg+3] = tau

  # send torques to robot and simulate TIME_STEP seconds 
  env.step(action) 

  # [TODO] save any CPG or robot states

##################################################### 
# PLOTS
#####################################################
# [TODO] Create your plots

# example
# fig = plt.figure()
# plt.plot(t,joint_pos[1,:], label='FR thigh')
# plt.legend()
# plt.show()