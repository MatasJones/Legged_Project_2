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
EVAL_POLICY = "TASK_STAIRS" # Options: TASK_STAIRS, TASK_SLOPES
N_RUNS = 10  # Number of full sweep rounds to average

# Formal Plotting Constants
TITLE_FS = 28
LABEL_FS = 22
TICK_FS  = 20
LINE_WIDTH = 4
PRIMARY_COLOR = '#2C3E50'
SECONDARY_COLOR = '#E74C3C'

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
    "add_noise": True,
    "task_env": "LR_COURSE_TASK",
    "cpg_gait": "TROT",
}

###############################################################################
# ISOLATED NORMALIZATION
###############################################################################
def load_obs_normalizer(stats_file):
    if not os.path.exists(stats_file):
        return None
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
# EPISODE RUNNER
###############################################################################
def run_eval_episode(env, model, target_time):
    obs = env.reset()
    if isinstance(obs, tuple): obs = obs[0]
    
    while True:
        obs_norm = normalize_obs(obs)
        action, _ = model.predict(obs_norm, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        curr_t = env.get_sim_time()

        if terminated: return False, curr_t
        if curr_t >= target_time: return True, curr_t
        if truncated: return (curr_t >= target_time), curr_t

###############################################################################
# MAIN SWEEP
###############################################################################
def main():
    m_path = get_latest_model(log_dir)
    Algo = PPO if LEARNING_ALG == "PPO" else SAC
    model = Algo.load(m_path)
    
    if MAX_SLOPE:
        test_range = np.arange(0.25, 0.46, 0.01)
        label = "Slope Angle (rad)"
        terrain_type = "SLOPES"
    else:
        test_range = np.arange(0.03, 0.1, 0.01)
        label = "Stair Height (m)"
        terrain_type = "STAIRS"

    res_rate = []
    res_time = []
    vals = []

    output_dir = "extra plots"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n--- Starting Robust {EVAL_POLICY} Sweep (Noise: ON, Runs: {N_RUNS}) ---")

    for val in test_range:
        all_run_successes = []
        all_run_times = []
        
        # We repeat the trials N_RUNS times to average out the stochasticity
        for r in range(N_RUNS):
            run_successes = 0
            run_times = []
            n_trials = 5
            
            for _ in range(n_trials):
                cfg = env_config.copy()
                cfg["terrain"] = terrain_type
                if MAX_SLOPE: cfg["slope_pitch"] = float(val)
                else: cfg["stair_height"] = float(val)

                env = QuadrupedGymEnv(**cfg)
                if hasattr(env, '_pybullet_client'):
                    env._pybullet_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
                
                is_ok, t_survived = run_eval_episode(env, model, MIN_GOOD_TIME_S)
                run_successes += int(is_ok)
                run_times.append(t_survived)
                env.close()
            
            # Record results for this specific run
            all_run_successes.append(run_successes / n_trials)
            all_run_times.append(np.mean(run_times))

        # Average across all N_RUNS
        avg_rate = np.mean(all_run_successes)
        avg_time = np.mean(all_run_times)
        
        res_rate.append(avg_rate)
        res_time.append(avg_time)
        vals.append(val)
        
        print(f"{label}: {val:.3f} | Avg Success: {avg_rate*100:>3.0f}% | Avg Time: {avg_time:.2f}s")
        
        # Early stop if the robot is failing completely even after averaging
        if avg_rate == 0 and avg_time < 1.0:
            break

    # --- FORMAL PLOTTING ---
    fig, ax1 = plt.subplots(figsize=(14, 10))
    
    # Success Rate (Primary Axis)
    ax1.set_xlabel(label, fontsize=LABEL_FS, fontweight='bold')
    ax1.set_ylabel('Success Rate (Avg)', color=PRIMARY_COLOR, fontsize=LABEL_FS, fontweight='bold')
    ax1.plot(vals, res_rate, color=PRIMARY_COLOR, marker='o', markersize=10, 
             linewidth=LINE_WIDTH, label='Success Rate')
    ax1.tick_params(axis='y', labelcolor=PRIMARY_COLOR, labelsize=TICK_FS)
    ax1.tick_params(axis='x', labelsize=TICK_FS)
    ax1.set_ylim(-0.05, 1.1)
    ax1.grid(True, alpha=0.2, linestyle='--')

    # Survival Time (Secondary Axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Avg Survival Time (s)', color=SECONDARY_COLOR, fontsize=LABEL_FS, fontweight='bold')
    ax2.plot(vals, res_time, color=SECONDARY_COLOR, linestyle='--', marker='x', 
             markersize=10, linewidth=LINE_WIDTH, label='Survival Time')
    ax2.tick_params(axis='y', labelcolor=SECONDARY_COLOR, labelsize=TICK_FS)
    ax2.set_ylim(0, MIN_GOOD_TIME_S + 1)

    plt.title(f"Averaged Robustness Analysis: {EVAL_POLICY} (n={N_RUNS})", 
              fontsize=TITLE_FS, fontweight='bold', pad=30)
    
    # Border Styling
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_linewidth(LINE_WIDTH/2)
            spine.set_color('#1B2631')

    fig.tight_layout()
    save_path = os.path.join(output_dir, f"robustness_{EVAL_POLICY}_noisy_avg.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"\nExperiment complete. Results saved to: {save_path}")

if __name__ == "__main__":
    main()