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
EVAL_POLICY  = "VEL_TROT"  # Options: "VEL_TROT" or "VEL_WALK"
MIN_GOOD_TIME_S = 10.0      # Duration of the stability test

# Formal Plotting Constants
TITLE_FS, LABEL_FS, TICK_FS = 28, 22, 20
PRIMARY_COLOR   = '#2C3E50'   # Formal Navy
SECONDARY_COLOR = '#E74C3C'   # Formal Red

interm_dir = "./logs/intermediate_models/"
log_dir    = os.path.join(interm_dir, EVAL_POLICY)
output_dir = "extra plots"
os.makedirs(output_dir, exist_ok=True)

###############################################################################
# ENV CONFIGURATION (Strictly Velocity Focused)
###############################################################################
env_config = {
    "motor_control_mode": "CPG",
    "observation_space_mode": "LR_COURSE_OBS",
    "on_rack": False,
    "render": False,
    "record_video": False,
    "add_noise": True,         # Using environment's internal noise
    "task_env": "FWD_LOCOMOTION",
    "terrain": None,
    "cpg_gait": "WALK" if "WALK" in EVAL_POLICY else "TROT",
}

###############################################################################
# NORMALIZATION LOGIC
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
# MAIN EXECUTION
###############################################################################
def main():
    # 1. Load Model
    m_path = get_latest_model(log_dir)
    print(f"Loading model: {m_path}")
    model = (PPO if LEARNING_ALG == "PPO" else SAC).load(m_path)
    
    # 2. Setup Env
    env = QuadrupedGymEnv(**env_config)
    obs, _ = env.reset()
    
    history = {"t": [], "vx": []}
    
    # 3. Data Collection Loop
    print(f"Running {MIN_GOOD_TIME_S}s stability rollout (Noise: ON)...")
    while env.get_sim_time() < MIN_GOOD_TIME_S:
        obs_n = normalize_obs(obs)
        action, _ = model.predict(obs_n, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        
        # Get physics-engine ground truth velocity
        base_vel = env.robot.GetBaseLinearVelocity()
        history["t"].append(env.get_sim_time())
        history["vx"].append(base_vel[0])
        
        if terminated or truncated:
            print("Warning: Episode ended prematurely.")
            break

    env.close()

    # 4. Processing Results
    t = np.array(history["t"])
    vx = np.array(history["vx"])
    
    # Filter for steady-state (ignore 0s-2s acceleration)
    steady_mask = (t > 2.0)
    if np.any(steady_mask):
        avg_v = np.mean(vx[steady_mask])
        std_v = np.std(vx[steady_mask])
    else:
        avg_v, std_v = 0, 0

    # 5. Professional Plotting
    
    plt.figure(figsize=(14, 8))
    
    # Raw velocity line
    plt.plot(t, vx, color=PRIMARY_COLOR, lw=3, label='Forward Velocity ($v_x$)')
    
    # Mean and Standard Deviation (Stability Area)
    plt.axhline(y=avg_v, color=SECONDARY_COLOR, ls='--', lw=3, 
                label=f'Mean Cruising Speed: {avg_v:.2f} m/s')
    plt.fill_between(t, avg_v - std_v, avg_v + std_v, color=SECONDARY_COLOR, 
                    alpha=0.15, label=f'Velocity Jitter ($\pm 1 \sigma = {std_v:.3f}$)')

    # Formatting
    plt.title(f"Velocity Stability Profile: {EVAL_POLICY} (Noisy)", fontsize=TITLE_FS, fontweight='bold', pad=20)
    plt.xlabel("Time (s)", fontsize=LABEL_FS)
    plt.ylabel("Velocity (m/s)", fontsize=LABEL_FS)
    plt.xticks(fontsize=TICK_FS)
    plt.yticks(fontsize=TICK_FS)
    plt.grid(True, alpha=0.3, ls='--')
    plt.legend(fontsize=16, loc='lower right')
    
    # Clean spines
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"stability_{EVAL_POLICY}.png")
    plt.savefig(save_path, dpi=300)
    print(f"Showcase plot saved to: {save_path}")

if __name__ == "__main__":
    main()