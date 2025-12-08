# SPDX-FileCopyrightText: Copyright (c) 2022 Guillaume Bellegarda. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ... (Copyright headers implied) ...

"""
Run stable baselines 3 on quadruped env 
"""

import os
import multiprocessing 
from datetime import datetime
import torch # Needed for thread optimization

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env

from utils.utils import CheckpointCallback
from utils.file_utils import get_latest_model, write_env_config
from env.quadruped_gym_env import QuadrupedGymEnv

# --- GLOBAL CONFIG ---
LEARNING_ALG = "PPO" #Works
LOAD_NN = False      
USE_GPU = True       

# -----------------------------------------------------------
# OPTIMIZATION: CPU CORES
# Colab 12 threads = ~6 Physical Cores.
# We set this to 6 to give PyBullet dedicated physical cores.
# -----------------------------------------------------------
NUM_ENVS = 6

TRAINING_TASK = "VELOCITY" 
TARGET_VELOCITY = 0.1      

if TRAINING_TASK == "VELOCITY":
    env_configs = {
        "motor_control_mode": "PD",
        "task_env": "FWD_LOCOMOTION",
        "observation_space_mode": "LR_COURSE_OBS",
        "terrain": None,
        "add_noise": False,
        "des_vel_x": TARGET_VELOCITY,
    }
elif TRAINING_TASK == "SLOPES":
    env_configs = {
        "motor_control_mode": "PD",
        "task_env": "LR_COURSE_TASK",
        "observation_space_mode": "LR_COURSE_OBS",
        "terrain": "SLOPES",
        "add_noise": True,
        "des_vel_x": TARGET_VELOCITY,
    }
else:
    raise ValueError(TRAINING_TASK + " not implemented")

# -----------------------------------------------------------
# CRITICAL FIX: ENABLE GPU FOR PPO
# -----------------------------------------------------------
if USE_GPU:
    gpu_arg = "auto"  # This will detect CUDA for both PPO and SAC
else:
    gpu_arg = "cpu"

print(f"--- Training {LEARNING_ALG} on device: {gpu_arg} ---")


if __name__ == '__main__':
    # OPTIMIZATION: Restrict PyTorch CPU usage so PyBullet can breathe
    torch.set_num_threads(1)

    if LOAD_NN:
        interm_dir = "./logs/intermediate_models/"
        log_dir = interm_dir + ''  
        stats_path = os.path.join(log_dir, "vec_normalize.pkl")
        model_name = get_latest_model(log_dir)

    ROOT_SAVE_PATH = '/content/drive/MyDrive/QuadrupedRL_Logs/' #For Colab
    SAVE_PATH = ROOT_SAVE_PATH + '{}_{}'.format(
        TRAINING_TASK.lower(), datetime.now().strftime("%m%d%y%H%M%S")
    ) + '/'
    os.makedirs(SAVE_PATH, exist_ok=True)

    checkpoint_callback = CheckpointCallback(save_freq=30000, save_path=SAVE_PATH,
                                             name_prefix='rl_model', verbose=2)

    make_env_fn = lambda: QuadrupedGymEnv(**env_configs)
    
    # Use SubprocVecEnv for parallel execution
    vec_env_cls = SubprocVecEnv 

    if LOAD_NN:
        env = make_vec_env(make_env_fn, monitor_dir=SAVE_PATH, n_envs=NUM_ENVS, vec_env_cls=vec_env_cls)
        env = VecNormalize.load(stats_path, env)
        env.training = True
        env.norm_reward = False
    else:
        env = make_vec_env(make_env_fn, monitor_dir=SAVE_PATH, n_envs=NUM_ENVS, vec_env_cls=vec_env_cls)
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=100.)

    write_env_config(SAVE_PATH, env, updated_config=env_configs)

    # -----------------------------------------------------------
    # OPTIMIZATION: HYPERPARAMETERS
    # Large batches to saturate the GPU
    # -----------------------------------------------------------
    policy_kwargs = dict(net_arch=[256,256]) 

    # With 6 envs, 2048 steps each = 12,288 steps per update.
    # This keeps the GPU busy and reduces CPU interruption.
    n_steps_per_env = 2048 
    
    ppo_config = {  
        "gamma": 0.995, 
        "n_steps": n_steps_per_env, 
        "ent_coef": 0.0, 
        "learning_rate": 1e-4, 
        "vf_coef": 0.5,
        "max_grad_norm": 0.5, 
        "gae_lambda": 0.95, 
        "batch_size": 512,  # Large batch for GPU
        "n_epochs": 15, 
        "clip_range": 0.2, 
        "clip_range_vf": 1,
        "verbose": 1, 
        "tensorboard_log": None, 
        "_init_setup_model": True, 
        "policy_kwargs": policy_kwargs,
        "device": gpu_arg # Now correctly passes "auto" (GPU)
    }

    sac_config={
        "learning_rate":1e-4,
        "buffer_size":1000000,
        "batch_size":4096,
        "ent_coef":'auto', 
        "gamma":0.99, 
        "tau":0.005,
        "train_freq":100, 
        "gradient_steps":100,
        "learning_starts": 2000,
        "verbose":1, 
        "tensorboard_log":None,
        "policy_kwargs": policy_kwargs,
        "seed":None, 
        "device": gpu_arg
    }

    if LEARNING_ALG == "PPO":
        model = PPO('MlpPolicy', env, **ppo_config)
    elif LEARNING_ALG == "SAC":
        model = SAC('MlpPolicy', env, **sac_config)
    else:
        raise ValueError(LEARNING_ALG + ' not implemented')

    if LOAD_NN:
        if LEARNING_ALG == "PPO":
            model = PPO.load(model_name, env)
        elif LEARNING_ALG == "SAC":
            model = SAC.load(model_name, env)
        print("\nLoaded model", model_name, "\n")

    model.learn(total_timesteps=1000000, log_interval=1, callback=checkpoint_callback)

    model.save( os.path.join(SAVE_PATH, "rl_model" ) ) 
    env.save(os.path.join(SAVE_PATH, "vec_normalize.pkl" )) 

    if LEARNING_ALG == "SAC": 
        model.save_replay_buffer(os.path.join(SAVE_PATH,"off_policy_replay_buffer"))