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

import os, sys
import gymnasium as gym
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sys import platform
# may be helpful depending on your system
# if platform =="darwin": # mac
#   import PyQt5
#   matplotlib.use("Qt5Agg")
# else: # linux
#   matplotlib.use('TkAgg')

# stable-baselines3
from stable_baselines3.common.monitor import load_results 
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env # fix for newer versions of stable-baselines3

# utils
from env.quadruped_gym_env import QuadrupedGymEnv
from utils.utils import plot_results
from utils.file_utils import get_latest_model, load_all_results, get_sorted_dirs, read_env_config

LEARNING_ALG = "PPO" #"SAC"
interm_dir = "./logs/intermediate_models/"

# If you want a specific run, set this to the subdirectory name
# e.g. RUN_SUBDIR = "velocity_102824115106"
RUN_SUBDIR = None

if RUN_SUBDIR is None:
    # automatically pick the most recent run
    run_dirs = get_sorted_dirs(interm_dir)
    if len(run_dirs) == 0:
        raise RuntimeError("No run directories found in {}".format(interm_dir))
    log_dir = run_dirs[-1]
else:
    log_dir = os.path.join(interm_dir, RUN_SUBDIR)

print("Using run directory:", log_dir)

# initialize env configs (render at test time)
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
monitor_results = load_results(log_dir)
print(monitor_results)
plot_results([log_dir] , 10e10, 'timesteps', LEARNING_ALG + ' ')
plt.show() 

# Read env config saved during training and override a few test-time flags
env_config = read_env_config(log_dir)
env_config['render'] = True
env_config['record_video'] = False
# usually you want deterministic test without extra noise
env_config['add_noise'] = False 

# reconstruct env 
make_env_fn = lambda: QuadrupedGymEnv(**env_config)
env = make_vec_env(make_env_fn, n_envs=1)
env = VecNormalize.load(stats_path, env)
env.training = False    # do not update stats at test time
env.norm_reward = False # reward normalization is not needed at test time

# load model
if LEARNING_ALG == "PPO":
    model = PPO.load(model_name, env)
elif LEARNING_ALG == "SAC":
    model = SAC.load(model_name, env)
print("\nLoaded model", model_name, "\n")

# reset env
obs = env.reset()
episode_reward = 0

# access underlying QuadrupedGymEnv
quad_env = env.venv.envs[0].env

# ----- Logging buffers -----
NUM_STEPS = 2000
time_log      = np.zeros(NUM_STEPS)
base_pos_log  = np.zeros((NUM_STEPS, 3))
base_vel_log  = np.zeros((NUM_STEPS, 3))
base_rpy_log  = np.zeros((NUM_STEPS, 3))
reward_log    = np.zeros(NUM_STEPS)

for i in range(NUM_STEPS):
    action, _states = model.predict(obs, deterministic=False) # or True for deterministic
    obs, rewards, dones, infos = env.step(action)
    r = rewards[0]
    episode_reward += r

    # Save logs
    time_log[i] = quad_env.get_sim_time()
    base_pos_log[i,:] = quad_env.robot.GetBasePosition()
    base_vel_log[i,:] = quad_env.robot.GetBaseLinearVelocity()
    base_rpy_log[i,:] = quad_env.robot.GetBaseOrientationRollPitchYaw()
    reward_log[i] = r

    if dones[0]:
        print('episode_reward', episode_reward)
        print('Final base position', infos[0]['base_pos'])
        episode_reward = 0

# ----- Basic plots -----
plt.figure()
plt.plot(time_log, base_pos_log[:,0], label='x')
plt.plot(time_log, base_pos_log[:,1], label='y')
plt.plot(time_log, base_pos_log[:,2], label='z')
plt.xlabel('time [s]')
plt.ylabel('base position [m]')
plt.title('Base position over time')
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(time_log, base_vel_log[:,0], label='vx')
plt.plot(time_log, base_vel_log[:,1], label='vy')
plt.plot(time_log, base_vel_log[:,2], label='vz')
plt.xlabel('time [s]')
plt.ylabel('base linear velocity [m/s]')
plt.title('Base linear velocity over time')
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(time_log, base_rpy_log[:,0], label='roll')
plt.plot(time_log, base_rpy_log[:,1], label='pitch')
plt.plot(time_log, base_rpy_log[:,2], label='yaw')
plt.xlabel('time [s]')
plt.ylabel('angles [rad]')
plt.title('Base orientation (roll/pitch/yaw)')
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(time_log, reward_log)
plt.xlabel('time [s]')
plt.ylabel('instant reward')
plt.title('Instant reward over time')
plt.tight_layout()

plt.show()