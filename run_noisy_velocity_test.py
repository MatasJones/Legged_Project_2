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
EVAL_POLICY  = "VEL_TROT"  # "VEL_TROT" or "VEL_WALK"
N_TRIALS     = 10          # Number of trials to average
MIN_GOOD_TIME_S = 10.0     # Duration per trial

# Formal Plotting Constants
TITLE_FS, LABEL_FS, TICK_FS = 34, 28, 26
PRIMARY_COLOR   = '#2C3E50'   # Formal Navy
SECONDARY_COLOR = '#E74C3C'   # Formal Red

interm_dir = "./logs/intermediate_models/"
log_dir    = os.path.join(interm_dir, EVAL_POLICY)
output_dir = "extra plots"
os.makedirs(output_dir, exist_ok=True)

###############################################################################
# ENV CONFIGURATION
###############################################################################
env_config = {
    "motor_control_mode": "CPG",
    "observation_space_mode": "LR_COURSE_OBS",
    "on_rack": False,
    "render": False,
    "record_video": False,
    "add_noise": True,         # Internal noise enabled
    "task_env": "FWD_LOCOMOTION",
    "terrain": None,
    "cpg_gait": "WALK" if "WALK" in EVAL_POLICY else "TROT",
}

###############################################################################
# NORMALIZATION
###############################################################################
def load_obs_normalizer(stats_file):
    if not os.path.exists(stats_file):
        print(f"Stats file not found: {stats_file}")
        return None
    dummy = DummyVecEnv([lambda: QuadrupedGymEnv(**env_config)])
    v = VecNormalize.load(stats_file, dummy)
    data = (v.obs_rms, v.clip_obs, v.epsilon)
    v.close()
    return data

stats_path = os.path.join(log_dir, "vec_normalize.pkl")
norm_data  = load_obs_normalizer(stats_path)

def normalize_obs(obs):
    if norm_data is None: return obs
    rms, clip, eps = norm_data
    obs = np.asarray(obs, dtype=np.float32)
    obs = (obs - rms.mean) / np.sqrt(rms.var + eps)
    return np.clip(obs, -clip, clip)

###############################################################################
# DATA COLLECTION (MULTI-TRIAL)
###############################################################################
def main():
    m_path = get_latest_model(log_dir)
    print(f"Loading model: {m_path}")
    model = (PPO if LEARNING_ALG == "PPO" else SAC).load(m_path)
    
    # To store time-series from all trials
    # We will interpolate to a common time grid to ensure alignment
    common_t = np.linspace(0, MIN_GOOD_TIME_S, 500) 
    all_v_interp = []

    print(f"--- Starting {N_TRIALS} Averaged Stability Trials for {EVAL_POLICY} ---")

    for i in range(N_TRIALS):
        env = QuadrupedGymEnv(**env_config)
        obs, _ = env.reset()
        trial_t, trial_v = [], []

        while env.get_sim_time() < MIN_GOOD_TIME_S:
            obs_n = normalize_obs(obs)
            action, _ = model.predict(obs_n, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            
            trial_t.append(env.get_sim_time())
            trial_v.append(env.robot.GetBaseLinearVelocity()[0])
            
            if terminated or truncated: break
        
        # Interpolate results to common time grid for averaging
        v_interp = np.interp(common_t, trial_t, trial_v)
        all_v_interp.append(v_interp)
        
        env.close()
        print(f"Trial {i+1}/{N_TRIALS} complete.")

    # Convert to numpy for stats
    all_v_interp = np.array(all_v_interp)
    mean_v = np.mean(all_v_interp, axis=0)
    std_v  = np.std(all_v_interp, axis=0)

    # Calculate overall metrics for the steady-state period (t > 2.0s)
    steady_mask = common_t > 2.0
    final_avg = np.mean(mean_v[steady_mask])
    final_std = np.mean(std_v[steady_mask]) # Mean deviation across trials

    ###############################################################################
    # PROFESSIONAL PLOTTING
    ###############################################################################
    
    plt.figure(figsize=(16, 9))
    
    # Plot Shaded Variance (Standard Deviation across trials)
    plt.fill_between(common_t, mean_v - std_v, mean_v + std_v, 
                     color=SECONDARY_COLOR, alpha=0.2, label=f'Inter-trial Deviation ($\pm 1 \sigma$)')
    
    # Plot Mean Trajectory
    plt.plot(common_t, mean_v, color=PRIMARY_COLOR, lw=4, label='Mean Velocity ($v_x$)')
    
    # Plot horizontal reference for cruising speed
    plt.axhline(y=final_avg, color=SECONDARY_COLOR, ls='--', lw=3, alpha=0.8,
                label=f'Avg Cruising Speed: {final_avg:.2f} m/s')

    # Formatting
    plt.title(f"Averaged Velocity Stability: {EVAL_POLICY} ($n={N_TRIALS}$)", 
              fontsize=TITLE_FS, fontweight='bold', pad=30)
    plt.xlabel("Time (s)", fontsize=LABEL_FS)
    plt.ylabel("Forward Velocity (m/s)", fontsize=LABEL_FS)
    plt.xticks(fontsize=TICK_FS)
    plt.yticks(fontsize=TICK_FS)
    plt.grid(True, alpha=0.2, ls='--')
    plt.legend(fontsize=22, loc='lower right', frameon=True, shadow=True)
    
    # Layout adjustment
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"avg_stability_{EVAL_POLICY}.png")
    plt.savefig(save_path, dpi=300)
    print(f"\nAnalysis complete. Save path: {save_path}")
    print(f"Results: Mean={final_avg:.3f} m/s, Average Noise Deviation={final_std:.3f}")

if __name__ == "__main__":
    main()