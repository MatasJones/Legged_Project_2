import os
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import pybullet as p

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.quadruped_gym_env import QuadrupedGymEnv
from utils.file_utils import get_latest_model

###############################################################################
# USER SETTINGS
###############################################################################
LEARNING_ALG = "PPO" 
EVAL_POLICY = "TASK_STAIRS"  # Toggle: "TASK_SLOPES" or "TASK_STAIRS"

interm_dir = "./logs/intermediate_models/"
log_dir = os.path.join(interm_dir, EVAL_POLICY)
stats_path = os.path.join(log_dir, "vec_normalize.pkl")

# Experiment Flags
MAX_STAIR_HEIGHT = (EVAL_POLICY == "TASK_STAIRS")
MAX_SLOPE = (EVAL_POLICY == "TASK_SLOPES")

MIN_GOOD_TIME_S = 8.0 

# Base configuration
env_config = {
    "motor_control_mode": "CPG",
    "observation_space_mode": "LR_COURSE_OBS",
    "on_rack": False,
    "render": False,
    "record_video": False,
    "add_noise": False,
    "task_env": "LR_COURSE_TASK",
    "cpg_gait": "TROT",
}

###############################################################################
# ISOLATED NORMALIZATION
###############################################################################
def load_obs_normalizer(stats_file):
    if not os.path.exists(stats_file):
        print(f"Stats file not found at {stats_file}")
        return None
    
    # Create a minimal dummy env just to load the pickle
    dummy_cfg = env_config.copy()
    dummy_cfg["terrain"] = "SLOPES" if MAX_SLOPE else "STAIRS"
    
    dummy = DummyVecEnv([lambda: QuadrupedGymEnv(**dummy_cfg)])
    v = VecNormalize.load(stats_file, dummy)
    data = (v.obs_rms, v.clip_obs, v.epsilon)
    v.close()
    return data

norm_data = load_obs_normalizer(stats_path)

def normalize_obs(obs):
    if norm_data is None: return obs
    rms, clip, eps = norm_data
    obs = np.array(obs, dtype=np.float32)
    obs = (obs - rms.mean) / np.sqrt(rms.var + eps)
    return np.clip(obs, -clip, clip)

###############################################################################
# EPISODE RUNNER (Strict Logic)
###############################################################################
def run_eval_episode(env, model, target_time):
    obs = env.reset()
    if isinstance(obs, tuple): obs = obs[0]
    
    while True:
        obs_norm = normalize_obs(obs)
        action, _ = model.predict(obs_norm, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        
        curr_t = env.get_sim_time()

        # 1. Immediate failure if fallen
        if terminated:
            return False, curr_t
        
        # 2. Success if time limit reached
        if curr_t >= target_time:
            return True, curr_t
            
        # 3. Handle gym internal truncation
        if truncated:
            return (curr_t >= target_time), curr_t

###############################################################################
# MAIN SWEEP
###############################################################################
def main():
    m_path = get_latest_model(log_dir)
    Algo = PPO if LEARNING_ALG == "PPO" else SAC
    model = Algo.load(m_path)
    
    # Define range and terrain-specific settings
    if MAX_SLOPE:
        test_range = np.arange(0.25, 0.46, 0.01) # 0.01 rad precision
        label = "Slope Angle (rad)"
        terrain_type = "SLOPES"
    else:
        test_range = np.arange(0.03, 0.1, 0.01) # 1cm precision
        label = "Stair Height (m)"
        terrain_type = "STAIRS"

    res_rate = []
    res_time = []
    vals = []

    print(f"\n--- Starting {EVAL_POLICY} Sweep ---")
    print(f"Model: {m_path}")

    for val in test_range:
        successes = 0
        times = []
        n_trials = 5
        
        for _ in range(n_trials):
            cfg = env_config.copy()
            cfg["terrain"] = terrain_type
            
            if MAX_SLOPE:
                cfg["slope_pitch"] = float(val)
            else:
                cfg["stair_height"] = float(val)

            env = QuadrupedGymEnv(**cfg)
            
            # Bullet silence
            if hasattr(env, '_pybullet_client'):
                env._pybullet_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
                env._pybullet_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            
            is_ok, t_survived = run_eval_episode(env, model, MIN_GOOD_TIME_S)
            
            successes += int(is_ok)
            times.append(t_survived)
            env.close()

        rate = successes / n_trials
        avg_t = np.mean(times)
        
        res_rate.append(rate)
        res_time.append(avg_t)
        vals.append(val)
        
        print(f"{label}: {val:.3f} | Success: {rate*100:>3.0f}% | Avg Time: {avg_t:.2f}s")
        
        # Stop early if the robot is failing completely
        if rate == 0 and avg_t < 1.0:
            print("Total failure detected. Ending sweep.")
            break

    # Final Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color_rate = 'tab:blue'
    ax1.set_xlabel(label)
    ax1.set_ylabel('Success Rate', color=color_rate)
    ax1.plot(vals, res_rate, color=color_rate, marker='o', linewidth=2, label='Success Rate')
    ax1.tick_params(axis='y', labelcolor=color_rate)
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color_time = 'tab:red'
    ax2.set_ylabel('Avg Survival Time (s)', color=color_time)
    ax2.plot(vals, res_time, color=color_time, linestyle='--', marker='x', label='Survival Time')
    ax2.tick_params(axis='y', labelcolor=color_time)
    ax2.set_ylim(0, MIN_GOOD_TIME_S + 1)

    plt.title(f"Detailed Robustness Analysis: {EVAL_POLICY}")
    fig.tight_layout()
    plt.savefig(f"robustness_{EVAL_POLICY}.png")
    print(f"\nExperiment complete. Results saved to robustness_{EVAL_POLICY}.png")

if __name__ == "__main__":
    main()