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
from matplotlib.widgets import Slider

from matplotlib.animation import FuncAnimation

ADD_CARTESIAN_PD = False
TIME_STEP = 0.001
foot_y = 0.0838 # this is the hip length 
sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

env = QuadrupedGymEnv(render=True,              # visualize
                    on_rack=False,              # useful for debugging! 
                    isRLGymInterface=False,     # not using RL
                    time_step=TIME_STEP,
                    action_repeat=1,
                    motor_control_mode="TORQUE",
                    add_noise=False,    # start in ideal conditions
                    # record_video=True
                    )

# initialize Hopf Network, supply gait
mu = 1.7 ** 2
omega_swing = 9 * 2 * np.pi
omega_stance = 10 * 2 * np.pi
gait = "TROT"
alpha = 45
coupling_strength = 20
ground_clearance = 0.05   # foot swing height 
ground_penetration = 0.05 # foot stance penetration into ground 
robot_height = 0.25        # in nominal case (standing) 
des_step_len = 0.05

best_vel = 0

# During your simulation loop:
total_energy = 0

if gait == "BOUND":
  omega_swing = 10 * 2 * np.pi
  omega_stance = 18 * 2 * np.pi
  ground_clearance = 0.2   # foot swing height 
  ground_penetration = 0.08
  des_step_len = 0.09

if gait == "WALK":
  omega_swing = 12 * 2 * np.pi
  omega_stance = 10 * 2 * np.pi
  ground_clearance = 0.2   # foot swing height 
  ground_penetration = 0.01
  des_step_len = 0.07

cpg = HopfNetwork(time_step=TIME_STEP, mu=mu, omega_swing=omega_swing, omega_stance=omega_stance, gait=gait, alpha=alpha, coupling_strength=coupling_strength, ground_clearance=ground_clearance, ground_penetration=ground_penetration, robot_height=robot_height, des_step_len=des_step_len)

TEST_STEPS = int(5 / (TIME_STEP))
t = np.arange(TEST_STEPS)*TIME_STEP

# [TODO] initialize data structures to save CPG and robot state
# [TODO] Matas done
# Reminder: r = current amplitude of oscillator and theta = oscilator phase
CPG_r = []
CPG_theta = []
CPG_r_dot = []
CPG_theta_dot = []
leg_desired = []
leg_current = []
angle_desired = []
angle_current = []

############## Sample Gains
# joint PD gains
kp=np.array([150,150,150])
kd=np.array([1,2,2])

# Cartesian PD gains
kpCartesian = np.diag([100]*3)
kdCartesian = np.diag([3]*3)

if gait == "BOUND":
  kp=np.array([100,100,100])
  kd=np.array([5,5,5])

if gait == "WALK":
  kp=np.array([80,80,80])
  kd=np.array([4,4,4])






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

    tau += kp * (leg_q - q_i) + kd * (-dq_i)

    # For plotting
    _, leg_p = env.robot.ComputeJacobianAndPosition(current_leg)

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

  E_mech_step = np.sum(np.abs(action * dq)) * TIME_STEP
  total_energy += E_mech_step

  vel = env.robot.GetBaseLinearVelocity()
  abs_vel = np.sqrt(vel[0]**2 + vel[1]**2)

  if abs_vel > best_vel:
      best_vel = abs_vel

  # [TODO] save any CPG or robot states
  CPG_r.append(cpg.get_r().copy())
  CPG_r_dot.append(cpg.get_dr().copy())
  CPG_theta.append(cpg.get_theta().copy())
  CPG_theta_dot.append(cpg.get_dtheta().copy())
  leg_desired.append(leg_xyz.copy())
  leg_current.append(leg_p.copy())
  angle_desired.append(leg_q.copy())
  angle_current.append(q_i.copy())


##################################################### 
# PLOTS
#####################################################
# [TODO] Create your plots
print(f"best velocity: {best_vel}")
distance_traveled = np.sqrt(env.robot.GetBasePosition()[0]**2 + env.robot.GetBasePosition()[1]**2)

COT = total_energy / (9.81 * np.sum(env.robot._total_mass_urdf) * distance_traveled)
print(f"COT = {COT}")
# Plot can be set to:
# 1) "CPG_states_1"
# 2) "CPG_states_combined"
# 3) "leg_positions"
# 4) "leg_angles"
PLOT = "CPG_states_1"

if PLOT == "CPG_states_1":
    CPG_r = np.array(CPG_r) 
    CPG_r_dot = np.array(CPG_r_dot)
    CPG_theta = np.array(CPG_theta)
    CPG_theta_dot = np.array(CPG_theta_dot)
    
    time_vector = np.arange(len(CPG_r)) * TIME_STEP
    
    # Parameters
    window_size = 600  # Only plot first 600 points
    window_size_seconds = window_size * TIME_STEP
    
    leg_names = ['FR', 'FL', 'RL', 'RR']
    labels = ['r [-]', 'r_dot [-/s]', 'θ [rad]', 'θ_dot [rad/s]']
    data_sources = [CPG_r, CPG_r_dot, CPG_theta, CPG_theta_dot]
    
    leg_colors = ['b', 'r', 'g', 'm']
    
    # SPACING CONTROLS
    SUBPLOT_WIDTH = 0.38
    SUBPLOT_HEIGHT = 0.10
    H_SPACING = 0.44
    V_SPACING = 0.49
    SIGNAL_SPACING = 0.110
    LEFT_MARGIN = 0.08
    BOTTOM_MARGIN = 0.05
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    
    leg_axes = []
    leg_lines = []
    
    for leg_idx in range(4):
        inner_axes = []
        inner_lines = []
        
        for signal_idx, (label, data) in enumerate(zip(labels, data_sources)):
            left = LEFT_MARGIN + (leg_idx % 2) * H_SPACING
            bottom = BOTTOM_MARGIN + (1 - leg_idx // 2) * V_SPACING + (3 - signal_idx) * SIGNAL_SPACING
            
            ax = fig.add_axes([left, bottom, SUBPLOT_WIDTH, SUBPLOT_HEIGHT])
            
            # Create line with label only for first subplot (r) of each leg
            if signal_idx == 0:
                line, = ax.plot([], [], lw=1.5, color=leg_colors[leg_idx], 
                              label=leg_names[leg_idx])
                # Add legend to first subplot only
                ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
            else:
                line, = ax.plot([], [], lw=1.5, color=leg_colors[leg_idx])
            
            inner_lines.append(line)
            inner_axes.append(ax)
            
            ax.set_ylabel(label, fontsize=12, labelpad=0, fontweight='bold') 
            ax.yaxis.set_label_coords(-0.10, 0.5) 
            
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=12)
            
            if signal_idx == 3:
                ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
            else:
                ax.set_xticklabels([])
        
        leg_axes.append(inner_axes)
        leg_lines.append(inner_lines)
    
    # Plot only first 600 cycles
    start_idx = 0
    end_idx = min(600, len(CPG_r))  # First 600 points or all data if less
    
    for leg_idx in range(4):
        for signal_idx, data in enumerate(data_sources):
            x_data = time_vector[start_idx:end_idx]
            y_data = data[start_idx:end_idx, leg_idx]
            
            line = leg_lines[leg_idx][signal_idx]
            ax = leg_axes[leg_idx][signal_idx]
            
            line.set_data(x_data, y_data)
            
            if len(y_data) > 0:
                y_min, y_max = np.min(y_data), np.max(y_data)
                margin = (y_max - y_min) * 0.1 if y_max != y_min else 1
                ax.set_ylim(y_min - margin, y_max + margin)
            
            ax.set_xlim(x_data[0], x_data[-1])
    
    plt.show()

if PLOT == "leg_positions":
    # Convert to numpy arrays
    leg_desired = np.array(leg_desired)  # shape: (time_steps, 3) - xyz for one leg
    leg_current = np.array(leg_current)  # shape: (time_steps, 3) - xyz for one leg
    time_vector = np.arange(len(leg_desired)) * TIME_STEP
    
    from matplotlib.widgets import Slider
    
    # Parameters
    window_size = 300
    coord_labels = ['x [m]', 'y [m]', 'z [m]']
    
    # Create figure with 3 subplots (one for each coordinate)
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    fig.suptitle('Leg Position: Desired vs Current with Cartesian PD', fontsize=14, fontweight='bold', y=0.96)
    plt.subplots_adjust(bottom=0.12, hspace=0.3, top=0.93)
    
    # Initialize lines
    lines_desired = []
    lines_current = []
    
    for coord_idx, (ax, coord_label) in enumerate(zip(axes, coord_labels)):
        # Plot desired (solid line) and current (dashed line)
        line_desired, = ax.plot([], [], lw=2, color='b', label='Desired', linestyle='-')
        line_current, = ax.plot([], [], lw=2, color='r', label='Current', linestyle='--')
        
        lines_desired.append(line_desired)
        lines_current.append(line_current)
        
        ax.set_ylabel(coord_label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        if coord_idx == 2:  # Bottom subplot
            ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    
    # Create slider
    slider_ax = plt.axes([0.15, 0.03, 0.7, 0.025])
    slider = Slider(
        ax=slider_ax,
        label='Window Position (s)',
        valmin=0,
        valmax=max(0, (len(leg_desired) - window_size) * TIME_STEP),
        valinit=0,
        valstep=TIME_STEP
    )
    
    def update(val):
        start_time = slider.val
        start_idx = int(start_time / TIME_STEP)
        end_idx = start_idx + window_size
        
        if end_idx <= len(leg_desired):
            x_data = time_vector[start_idx:end_idx]
            
            for coord_idx in range(3):
                # Get data for this coordinate
                y_desired = leg_desired[start_idx:end_idx, coord_idx]
                y_current = leg_current[start_idx:end_idx, coord_idx]
                
                # Update lines
                lines_desired[coord_idx].set_data(x_data, y_desired)
                lines_current[coord_idx].set_data(x_data, y_current)
                
                # Auto-scale y-axis
                ax = axes[coord_idx]
                all_y = np.concatenate([y_desired, y_current])
                if len(all_y) > 0:
                    y_min, y_max = np.min(all_y), np.max(all_y)
                    margin = (y_max - y_min) * 0.1 if y_max != y_min else 0.01
                    ax.set_ylim(y_min - margin, y_max + margin)
                
                # Set x-axis limits
                ax.set_xlim(x_data[0], x_data[-1])
        
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    update(0)
    plt.show()


if PLOT == "leg_angles":
  # Convert to numpy arrays
  angle_desired = np.array(angle_desired)  # shape: (time_steps, 3) - xyz for one leg
  angle_current = np.array(angle_current)  # shape: (time_steps, 3) - xyz for one leg
  time_vector = np.arange(len(angle_desired)) * TIME_STEP
  
  from matplotlib.widgets import Slider
  
  # Parameters
  window_size = 500
  coord_labels = ['Hip angle [rad]', 'Calf angle [rad]', 'Foot angle [rad]']
  
  # Create figure with 3 subplots (one for each coordinate)
  fig, axes = plt.subplots(3, 1, figsize=(12, 12))
  fig.suptitle('Leg Angle: Desired vs Current with Cartesian PD', fontsize=14, fontweight='bold', y=0.96)
  plt.subplots_adjust(bottom=0.12, hspace=0.3, top=0.93)
  
  # Initialize lines
  lines_desired = []
  lines_current = []
  
  for coord_idx, (ax, coord_label) in enumerate(zip(axes, coord_labels)):
      # Plot desired (solid line) and current (dashed line)
      line_desired, = ax.plot([], [], lw=2, color='b', label='Desired', linestyle='-')
      line_current, = ax.plot([], [], lw=2, color='r', label='Current', linestyle='--')
      
      lines_desired.append(line_desired)
      lines_current.append(line_current)
      
      ax.set_ylabel(coord_label, fontsize=12, fontweight='bold')
      ax.grid(True, alpha=0.3)
      ax.tick_params(labelsize=10)
      ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
      
      if coord_idx == 2:  # Bottom subplot
          ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
  
  # Create slider
  slider_ax = plt.axes([0.15, 0.03, 0.7, 0.025])
  slider = Slider(
      ax=slider_ax,
      label='Window Position (s)',
      valmin=0,
      valmax=max(0, (len(leg_desired) - window_size) * TIME_STEP),
      valinit=0,
      valstep=TIME_STEP
  )
  
  def update(val):
      start_time = slider.val
      start_idx = int(start_time / TIME_STEP)
      end_idx = start_idx + window_size
      
      if end_idx <= len(leg_desired):
          x_data = time_vector[start_idx:end_idx]
          
          for coord_idx in range(3):
              # Get data for this coordinate
              y_desired = angle_desired[start_idx:end_idx, coord_idx]
              y_current = angle_current[start_idx:end_idx, coord_idx]
              
              # Update lines
              lines_desired[coord_idx].set_data(x_data, y_desired)
              lines_current[coord_idx].set_data(x_data, y_current)
              
              # Auto-scale y-axis
              ax = axes[coord_idx]
              all_y = np.concatenate([y_desired, y_current])
              if len(all_y) > 0:
                  y_min, y_max = np.min(all_y), np.max(all_y)
                  margin = (y_max - y_min) * 0.1 if y_max != y_min else 0.01
                  ax.set_ylim(y_min - margin, y_max + margin)
              
              # Set x-axis limits
              ax.set_xlim(x_data[0], x_data[-1])
      
      fig.canvas.draw_idle()
  
  slider.on_changed(update)
  update(0)
  plt.show()