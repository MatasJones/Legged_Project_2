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

import os
import glob
import csv
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.quadruped_gym_env import QuadrupedGymEnv
from utils.file_utils import get_latest_model

###############################################################################
# USER SETTINGS
###############################################################################
LEARNING_ALG = "PPO"  # "PPO" or "SAC"

# --------------------------------------------------------------------------
# Toggle which policy to evaluate :
#   "VEL_TROT"    -> flat ground velocity using TROT gait  
#   "VEL_WALK"    -> flat ground velocity using WALK gait  
#   "TASK_SLOPES" -> slopes task-specific policy           
#   "TASK_STAIRS" -> stairs task-specific policy           
# --------------------------------------------------------------------------
EVAL_POLICY = "TASK_STAIRS"

interm_dir = "./logs/intermediate_models/"
log_dir = os.path.join(interm_dir, "TASK_STAIRS")  # set your run folder here
print(f"Using log_dir: {log_dir}")

# --- Multi-try evaluation (in case the robot falls at the beginning of the simulation) ---
MAX_TRIES = 3
MIN_GOOD_TIME_S = 8.0
WARMUP_PER_TRY_S = 1.0
STOP_AT_FIRST_GOOD = True

# --- Training curves ---
MA_WINDOW = 50  # moving average window in episodes

###############################################################################
# BUILD EVAL ENV CONFIG (do not rely on env_configs.json)
###############################################################################
env_config = {
    "motor_control_mode": "CPG",
    "observation_space_mode": "LR_COURSE_OBS",
    "on_rack": False,
    "render": True,
    "record_video": False,
    "add_noise": False,
    "terrain": None,
    "task_env": "FWD_LOCOMOTION",
    "test_flagrun": False,
    "cpg_gait": "TROT",
    "slope_pitch": 0.2,
}

if EVAL_POLICY == "VEL_TROT":
    env_config["task_env"] = "FWD_LOCOMOTION"
    env_config["terrain"] = None
    env_config["cpg_gait"] = "TROT"

elif EVAL_POLICY == "VEL_WALK":
    env_config["task_env"] = "FWD_LOCOMOTION"
    env_config["terrain"] = None
    env_config["cpg_gait"] = "WALK"

elif EVAL_POLICY == "TASK_SLOPES":
    env_config["task_env"] = "LR_COURSE_TASK"
    env_config["terrain"] = "SLOPES"
    env_config["cpg_gait"] = "TROT"

elif EVAL_POLICY == "TASK_STAIRS":
    env_config["task_env"] = "LR_COURSE_TASK"
    env_config["terrain"] = "STAIRS"
    env_config["cpg_gait"] = "TROT"

else:
    raise ValueError(f"Unknown EVAL_POLICY: {EVAL_POLICY}")


###############################################################################
# PATHS
###############################################################################
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_path = get_latest_model(log_dir)
print("Latest model file:", model_path)

TITLE_FS = 30
LABEL_FS = 26
TICK_FS  = 24
LINE_WIDTH = 4

TITLE_FS_EP = 18
LABEL_FS_EP = 16
TICK_FS_EP  = 14


###############################################################################
# TRAINING CURVES (from the monitor CSV, we plot the ep length + ep return)
###############################################################################
def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return np.array([])
    w = np.ones(window, dtype=float) / float(window)
    return np.convolve(y, w, mode="valid")

def load_monitor_csvs(run_dir: str):
    """
    Loads SB3 Monitor CSV(s) from a run directory, including multi-env monitor files.
    Uses absolute wall-time = t_start + t to merge multi-env logs properly.
    Returns arrays: timesteps (cumulative), ep_returns, ep_lengths.
    """
    patterns = [
        os.path.join(run_dir, "monitor*.csv"),
        os.path.join(run_dir, "*.monitor.csv"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(set(files))

    if not files:
        print(f"[WARN] No monitor CSV files found in {run_dir}. Training curves will be skipped.")
        return None

    rows = []
    for fpath in files:
        t_start = None
        header_found = False

        with open(fpath, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue

                # comment/header lines
                if row[0].startswith("#"):
                    # try parse t_start from JSON in the first line
                    if t_start is None:
                        try:
                            meta = json.loads(row[0][1:])
                            t_start = float(meta.get("t_start", 0.0))
                        except Exception:
                            t_start = 0.0
                    continue

                # header row: r,l,t
                if not header_found:
                    if row[0].strip() == "r" and len(row) >= 3 and row[1].strip() == "l" and row[2].strip() == "t":
                        header_found = True
                    continue

                # data row
                try:
                    r = float(row[0])
                    l = int(float(row[1]))
                    t = float(row[2])
                    abs_t = (t_start if t_start is not None else 0.0) + t
                    rows.append((abs_t, r, l))
                except Exception:
                    continue

    if not rows:
        print(f"[WARN] Monitor CSV files found but no data rows parsed. Training curves will be skipped.")
        return None

    # Sort chronologically by absolute wall time, then compute cumulative timesteps
    rows.sort(key=lambda x: x[0])
    ep_returns = np.array([x[1] for x in rows], dtype=float)
    ep_lengths = np.array([x[2] for x in rows], dtype=int)
    timesteps = np.cumsum(ep_lengths)

    return timesteps, ep_returns, ep_lengths

def plot_training_curves(run_dir: str, ma_window: int = 50):
    data = load_monitor_csvs(run_dir)
    if data is None:
        return

    timesteps, ep_returns, ep_lengths = data

    # Episode length
    plt.figure(figsize=(8, 2.6))
    plt.scatter(timesteps, ep_lengths, s=6)
    ma_len = _moving_average(ep_lengths.astype(float), ma_window)
    if len(ma_len) > 0:
        x_ma = timesteps[ma_window - 1 :]
        plt.plot(x_ma, ma_len)
    plt.title(f"{EVAL_POLICY} {LEARNING_ALG} Ep Len", fontsize=TITLE_FS_EP)
    plt.xlabel("timesteps", fontsize=LABEL_FS_EP)
    plt.ylabel("Episode Length", fontsize=LABEL_FS_EP)
    plt.tick_params(axis="both", labelsize=TICK_FS_EP)
    plt.tight_layout()
    out_len = os.path.join(run_dir, "training_ep_len.png")
    plt.savefig(out_len, dpi=300)
    plt.close()

    # Episode return
    plt.figure(figsize=(8, 2.6))
    plt.scatter(timesteps, ep_returns, s=6)
    ma_ret = _moving_average(ep_returns.astype(float), ma_window)
    if len(ma_ret) > 0:
        x_ma = timesteps[ma_window - 1 :]
        plt.plot(x_ma, ma_ret)
    plt.title(f"{EVAL_POLICY} {LEARNING_ALG} Ep Return", fontsize=TITLE_FS_EP)
    plt.xlabel("timesteps", fontsize=LABEL_FS_EP)
    plt.ylabel("Episode Return", fontsize=LABEL_FS_EP)
    plt.tick_params(axis="both", labelsize=TICK_FS_EP)
    plt.tight_layout()
    out_ret = os.path.join(run_dir, "training_ep_return.png")
    plt.savefig(out_ret, dpi=300)
    plt.close()

    print("Training curves saved:")
    print(" -", out_len)
    print(" -", out_ret)

plot_training_curves(log_dir, ma_window=MA_WINDOW)

###############################################################################
# LOAD MODEL (suppress schedule-deserialization warnings via custom_objects)
###############################################################################
custom_objects = None
if LEARNING_ALG == "PPO":
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.2,
        "clip_range_vf": lambda _: 1.0,
    }

if LEARNING_ALG == "PPO":
    model = PPO.load(model_path, custom_objects=custom_objects)
elif LEARNING_ALG == "SAC":
    model = SAC.load(model_path)
else:
    raise ValueError(f"{LEARNING_ALG} not implemented")

print("\nLoaded model", model_path, "\n")

###############################################################################
# LOAD VecNormalize STATS -> extract obs_rms for manual normalization
###############################################################################
def load_obs_normalizer(stats_file, cfg):
    if not os.path.exists(stats_file):
        return None

    cfg2 = dict(cfg)
    cfg2["render"] = False
    cfg2["record_video"] = False

    dummy = DummyVecEnv([lambda: QuadrupedGymEnv(**cfg2)])
    v = VecNormalize.load(stats_file, dummy)
    v.training = False
    v.norm_reward = False

    obs_rms = v.obs_rms
    clip_obs = float(v.clip_obs)
    epsilon = float(v.epsilon)

    try:
        v.close()
    except Exception:
        pass

    return (obs_rms, clip_obs, epsilon)

normalizer = load_obs_normalizer(stats_path, env_config)
if normalizer is None:
    print("WARNING: vec_normalize.pkl not found -> running WITHOUT obs normalization.")

def normalize_obs(obs):
    if normalizer is None:
        return obs
    obs_rms, clip_obs, epsilon = normalizer
    obs = np.asarray(obs, dtype=np.float32)
    obs = (obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon)
    obs = np.clip(obs, -clip_obs, clip_obs)
    return obs

###############################################################################
# CREATE RAW ENV (no VecEnv, no auto-reset)
###############################################################################
env = QuadrupedGymEnv(**env_config)

# dt per env.step()
dt_per_sim = float(env._time_step)
dt_per_step = float(env._time_step * env._action_repeat)

###############################################################################
# HELPERS: mass, foot links, contacts, gait metrics, CoT
###############################################################################
def get_feet_reorder_idx_FRFLRRRL(pybullet_client, robot_id: int, foot_link_ids):
    """
    Returns:
      reorder_idx: list of 4 ints such that:
        feet_FRFLRRRL = feet_in_contact_raw[reorder_idx]
      names_in_foot_id_order: list[str] link/joint names corresponding to foot_link_ids order.
    """
    names_in_foot_id_order = []
    for link_id in foot_link_ids:
        try:
            info = pybullet_client.getJointInfo(robot_id, int(link_id))
            joint_name = info[1].decode("utf-8", errors="ignore")
            link_name  = info[12].decode("utf-8", errors="ignore")
            nm = (link_name or joint_name or str(link_id)).lower()
        except Exception:
            nm = str(link_id).lower()
        names_in_foot_id_order.append(nm)

    label_to_i = {}
    for i, nm in enumerate(names_in_foot_id_order):
        # common A1 patterns: "fr_foot", "fl_foot", "rr_foot", "rl_foot"
        if ("fr" in nm) and ("foot" in nm or "toe" in nm):
            label_to_i["FR"] = i
        elif ("fl" in nm) and ("foot" in nm or "toe" in nm):
            label_to_i["FL"] = i
        elif ("rr" in nm) and ("foot" in nm or "toe" in nm):
            label_to_i["RR"] = i
        elif ("rl" in nm) and ("foot" in nm or "toe" in nm):
            label_to_i["RL"] = i

    if len(label_to_i) == 4:
        reorder_idx = [label_to_i["FR"], label_to_i["FL"], label_to_i["RR"], label_to_i["RL"]]
        return reorder_idx, names_in_foot_id_order

    print("[WARN] Could not infer FR/FL/RR/RL from foot link names. Using raw order.")
    print("       foot_link_ids:", list(foot_link_ids))
    print("       names:", names_in_foot_id_order)
    return [0, 1, 2, 3], names_in_foot_id_order

def get_total_robot_mass(pybullet_client, robot_id: int) -> float:
    try:
        n = pybullet_client.getNumJoints(robot_id)
        m_total = 0.0
        for link_idx in range(-1, n):
            dinfo = pybullet_client.getDynamicsInfo(robot_id, link_idx)
            m_total += float(dinfo[0])
        return m_total
    except Exception:
        return float("nan")

def extract_stance_swing_durations(t_arr: np.ndarray, stance_bool: np.ndarray):
    """
    Given stance_bool over time, extract stance and swing durations (seconds).
    stance duration: True segment length
    swing duration : False segment length
    step_period    : touchdown->next touchdown (if possible)
    """
    if len(t_arr) < 2 or len(stance_bool) != len(t_arr):
        return [], [], []

    # transitions
    b = stance_bool.astype(int)
    db = np.diff(b)

    touchdown_idx = list(np.where(db == 1)[0] + 1)  # False -> True
    liftoff_idx = list(np.where(db == -1)[0] + 1)   # True -> False

    stance_durs = []
    swing_durs = []
    step_periods = []

    # stance segments: from touchdown to next liftoff
    for td in touchdown_idx:
        lo_candidates = [lo for lo in liftoff_idx if lo > td]
        if not lo_candidates:
            continue
        lo = lo_candidates[0]
        stance_durs.append(float(t_arr[lo] - t_arr[td]))

    # swing segments: from liftoff to next touchdown
    for lo in liftoff_idx:
        td_candidates = [td for td in touchdown_idx if td > lo]
        if not td_candidates:
            continue
        td = td_candidates[0]
        swing_durs.append(float(t_arr[td] - t_arr[lo]))

    # step period: touchdown to next touchdown
    for i in range(len(touchdown_idx) - 1):
        td0 = touchdown_idx[i]
        td1 = touchdown_idx[i + 1]
        step_periods.append(float(t_arr[td1] - t_arr[td0]))

    return stance_durs, swing_durs, step_periods

####################################################################################
# RUNNING ONE EPISODE (with warmup), RECORDING EVERYTHING NEEDED FOR PLOTS + METRICS
####################################################################################
def run_one_episode():
    obs, info = env.reset()
    terminated = False
    truncated = False
    ep_return = 0.0

    # hide rendering during warmup (optional)
    warmup_steps = int(WARMUP_PER_TRY_S / dt_per_step)
    if warmup_steps > 0 and env._is_render:
        import pybullet as pybullet
        env._pybullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

    for _ in range(warmup_steps):
        obs_in = normalize_obs(obs)
        action, _ = model.predict(obs_in, deterministic=True)
        obs, r, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            # failed during warmup
            if warmup_steps > 0 and env._is_render:
                import pybullet as pybullet
                env._pybullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
            return {
                "ok": False,
                "time": float(env.get_sim_time()),
                "return": float(ep_return),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "data": None,
                "final_info": info,
            }

    if warmup_steps > 0 and env._is_render:
        import pybullet as pybullet
        env._pybullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

    # rollout
    max_steps = int((env._MAX_EP_LEN / dt_per_step) + 5)

    t_list = []
    base_pos_list = []
    base_lin_vel_list = []
    base_rpy_list = []
    reward_list = []
    contact_list = []      # shape [T, n_feet]
    power_list = []        # avg power during this env.step (W)
    energy_total = 0.0     # J

    start_pos = np.array(env.robot.GetBasePosition(), dtype=float)

    rollout_t0 = float(env.get_sim_time())

    # --- robust foot ordering (FR, FL, RR, RL) based on URDF foot link names ---
    foot_link_ids = list(getattr(env.robot, "_foot_link_ids", []))
    reorder_idx_FRFLRRRL, foot_link_names = get_feet_reorder_idx_FRFLRRRL(
        env._pybullet_client, int(env.robot.quadruped), foot_link_ids
    )

    for _ in range(max_steps):
        t_now = float(env.get_sim_time())
        t_list.append(t_now)
        base_pos_list.append(env.robot.GetBasePosition())
        base_lin_vel_list.append(env.robot.GetBaseLinearVelocity())
        base_rpy_list.append(env.robot.GetBaseOrientationRollPitchYaw())

        # contacts at this decision step (robust: uses robot's internal foot link ids)
        try:
            _, _, _, feet_in_contact = env.robot.GetContactInfo()  # raw order = env.robot._foot_link_ids
            feet = np.asarray(feet_in_contact, dtype=int)
            feet = feet[reorder_idx_FRFLRRRL]  # now FR,FL,RR,RL
            contact_list.append(feet.tolist())
        except Exception:
            contact_list.append([0, 0, 0, 0])


        obs_in = normalize_obs(obs)
        action, _ = model.predict(obs_in, deterministic=True)
        obs, r, terminated, truncated, info = env.step(action)

        # accumulate energy from inner sim steps
        step_energy = 0.0
        try:
            for tau, vel in zip(env._dt_motor_torques, env._dt_motor_velocities):
                tau = np.asarray(tau, dtype=float)
                vel = np.asarray(vel, dtype=float)
                step_energy += float(np.sum(np.abs(tau * vel)) * dt_per_sim)
        except Exception:
            step_energy += 0.0

        energy_total += step_energy
        avg_power = step_energy / (dt_per_sim * float(env._action_repeat)) if env._action_repeat > 0 else 0.0
        power_list.append(float(avg_power))

        r = float(r)
        reward_list.append(r)
        ep_return += r

        if terminated or truncated:
            break

    rollout_t1 = float(env.get_sim_time())

    end_pos = np.array(env.robot.GetBasePosition(), dtype=float)
    sim_time = float(env.get_sim_time())

    data = {
        "t": np.asarray(t_list, dtype=float),
        "pos": np.asarray(base_pos_list, dtype=float),
        "vel": np.asarray(base_lin_vel_list, dtype=float),
        "rpy": np.asarray(base_rpy_list, dtype=float),
        "rew": np.asarray(reward_list, dtype=float),
        "contacts": np.asarray(contact_list, dtype=int) if len(contact_list) > 0 else None,
        "power": np.asarray(power_list, dtype=float),
        "energy_total_J": float(energy_total),
        "start_pos": start_pos,
        "end_pos": end_pos,
        "rollout_t0": rollout_t0,
        "rollout_t1": rollout_t1,
        "foot_link_ids_raw": foot_link_ids,
        "foot_link_names_raw": foot_link_names,
        "reorder_idx_FRFLRRRL": reorder_idx_FRFLRRRL,
    }

    ok = (sim_time >= MIN_GOOD_TIME_S) and (not terminated)
    return {
        "ok": bool(ok),
        "time": float(sim_time),
        "return": float(ep_return),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "data": data,
        "final_info": info,
    }


###############################################################################
# TRYING MULTIPLE EPISODES, KEEP FIRST GOOD OR BEST RETURN
###############################################################################
best = None

for i in range(1, MAX_TRIES + 1):
    res = run_one_episode()

    print(
        f"Try {i}/{MAX_TRIES} | time={res['time']:.2f}s | return={res['return']:.2f} "
        f"| terminated={res['terminated']} truncated={res['truncated']} | ok={res['ok']}"
    )

    if res["data"] is not None:
        if best is None or res["return"] > best["return"]:
            best = res

    if res["ok"] and STOP_AT_FIRST_GOOD:
        best = res
        break

if best is None or best.get("data") is None:
    raise RuntimeError("Could not generate any episode data to plot/analyze.")

print("Selected episode:")
print("  return:", best["return"])
print("  time:", best["time"])
print("  terminated:", best["terminated"], "truncated:", best["truncated"])
print("  Final base position:", best["final_info"].get("base_pos", None))

###############################################################################
# ROLLOUT PLOTS (ONLY SELECTED EPISODE)
###############################################################################
data = best["data"]
t_arr = data["t"]
base_pos_arr = data["pos"]
base_lin_vel_arr = data["vel"]
base_rpy_arr = data["rpy"]
reward_arr = data["rew"]
power_arr = data["power"]

plt.figure(figsize=(18, 4))
plt.plot(t_arr, base_pos_arr[:, 0], label="x", linewidth=LINE_WIDTH)
plt.plot(t_arr, base_pos_arr[:, 1], label="y", linewidth=LINE_WIDTH)
plt.plot(t_arr, base_pos_arr[:, 2], label="z", linewidth=LINE_WIDTH)
plt.xlabel("time [s]", fontsize=LABEL_FS)
plt.ylabel("base position [m]", fontsize=LABEL_FS) 
plt.tick_params(axis="both", labelsize=TICK_FS)
plt.legend(fontsize=LABEL_FS)
plt.title(f"{EVAL_POLICY} Base position over time over episode", fontsize=TITLE_FS)
plt.tight_layout()
plt.savefig(os.path.join(log_dir, "rollout_base_position.png"),dpi=300)
plt.close()

plt.figure(figsize=(18, 4))
plt.plot(t_arr, base_lin_vel_arr[:, 0], label="vx", linewidth=LINE_WIDTH)
plt.plot(t_arr, base_lin_vel_arr[:, 1], label="vy", linewidth=LINE_WIDTH)
plt.plot(t_arr, base_lin_vel_arr[:, 2], label="vz", linewidth=LINE_WIDTH)
plt.xlabel("time [s]", fontsize=LABEL_FS)
plt.ylabel("base linear velocity [m/s]", fontsize=LABEL_FS)
plt.tick_params(axis="both", labelsize=TICK_FS)
plt.legend(fontsize=LABEL_FS)
plt.title(f"{EVAL_POLICY} Base linear velocity over time over episode", fontsize=TITLE_FS)
plt.tight_layout()
plt.savefig(os.path.join(log_dir, "rollout_base_lin_vel.png"),dpi=300)
plt.close()

plt.figure(figsize=(18, 4))
plt.plot(t_arr, base_rpy_arr[:, 0], label="roll", linewidth=LINE_WIDTH)
plt.plot(t_arr, base_rpy_arr[:, 1], label="pitch", linewidth=LINE_WIDTH)
plt.plot(t_arr, base_rpy_arr[:, 2], label="yaw", linewidth=LINE_WIDTH)
plt.xlabel("time [s]", fontsize=LABEL_FS)
plt.ylabel("base RPY [rad]", fontsize=LABEL_FS)
plt.tick_params(axis="both", labelsize=TICK_FS)
plt.legend(fontsize=LABEL_FS)
plt.title(f"{EVAL_POLICY} Base orientation (RPY) over time over episode", fontsize=TITLE_FS)
plt.tight_layout()
plt.savefig(os.path.join(log_dir, "rollout_base_rpy.png"),dpi=300)
plt.close()

plt.figure(figsize=(18, 4))
plt.plot(t_arr[: len(reward_arr)], reward_arr, linewidth=LINE_WIDTH)
plt.xlabel("time [s]",fontsize=LABEL_FS)
plt.ylabel("reward", fontsize=LABEL_FS)
plt.tick_params(axis="both", labelsize=TICK_FS)
plt.title("Instantaneous reward over time (selected episode)", fontsize=TITLE_FS)
plt.tight_layout()
plt.savefig(os.path.join(log_dir, "rollout_reward.png"),dpi=300)
plt.close()

plt.figure(figsize=(18, 4))
plt.plot(t_arr[: len(power_arr)], power_arr, linewidth=LINE_WIDTH)
plt.xlabel("time [s]", fontsize=LABEL_FS)
plt.ylabel("power [W]", fontsize=LABEL_FS)
plt.tick_params(axis="both", labelsize=TICK_FS)
plt.title("Estimated mechanical power over time (selected episode)", fontsize=TITLE_FS)
plt.tight_layout()
plt.savefig(os.path.join(log_dir, "rollout_power.png"),dpi=300)
plt.close()

###############################################################################
# GAIT METRICS + CoT
###############################################################################
# Velocities: forward vx and speed magnitude in XY
vx = base_lin_vel_arr[:, 0]
vxy = np.linalg.norm(base_lin_vel_arr[:, 0:2], axis=1)

vx_min = float(np.min(vx)) if len(vx) else float("nan")
vx_max = float(np.max(vx)) if len(vx) else float("nan")
vxy_min = float(np.min(vxy)) if len(vxy) else float("nan")
vxy_max = float(np.max(vxy)) if len(vxy) else float("nan")

# Distance traveled (XY)
start_xy = np.asarray(data["start_pos"][0:2], dtype=float)
end_xy = np.asarray(data["end_pos"][0:2], dtype=float)
dist_xy = float(np.linalg.norm(end_xy - start_xy))
T = float(data["rollout_t1"] - data["rollout_t0"])
v_avg = dist_xy / T if T > 1e-9 else float("nan")

# CoT
robot_id = int(env.robot.quadruped)
m_total = get_total_robot_mass(env._pybullet_client, robot_id)
g = 9.8
E = float(data["energy_total_J"])
P_avg = E / T if T > 1e-9 else float("nan")
cot = (E / (m_total * g * dist_xy)) if (m_total > 0 and dist_xy > 1e-9) else float("nan")

# Contacts / duty / stance-swing / footfall
contacts = data.get("contacts", None)
leg_labels = ["FR", "FL", "RR", "RL"]

duty = {}
stance_swing = {}
footfall_events = {}

if contacts is None or contacts.shape[1] < 4:
    print("[WARN] Could not compute per-leg gait metrics (contacts missing or wrong shape).")
else:
    # contacts: shape [N,4] (best-effort order assumed FR,FL,RR,RL)
    contacts = contacts[:, :4].astype(bool)

    # duty cycle per leg
    for k in range(4):
        duty[leg_labels[k]] = float(np.mean(contacts[:, k]))

    # stance/swing durations per leg
    for k in range(4):
        stance_durs, swing_durs, step_periods = extract_stance_swing_durations(t_arr, contacts[:, k])
        stance_swing[leg_labels[k]] = {
            "stance_durations_s": stance_durs,
            "swing_durations_s": swing_durs,
            "step_periods_s": step_periods,
            "stance_mean_s": float(np.mean(stance_durs)) if len(stance_durs) else None,
            "swing_mean_s": float(np.mean(swing_durs)) if len(swing_durs) else None,
            "step_period_mean_s": float(np.mean(step_periods)) if len(step_periods) else None,
        }

        # touchdown times = False->True
        b = contacts[:, k].astype(int)
        td_idx = list(np.where(np.diff(b) == 1)[0] + 1)
        footfall_events[leg_labels[k]] = {
            "touchdown_times_s": [float(t_arr[i]) for i in td_idx]
        }

    # footfall pattern plot (per-leg colors)
    plt.figure(figsize=(12, 2.6))

    # contacts: shape (T, 4), True/1 = stance, False/0 = swing
    # we want an RGBA image of shape (4, T, 4)
    img_bool = contacts.T  # shape (4, T)

    # background (swing) color
    SWING_COLOR = "#ffffff"  
    # stance colors per leg
    stance_colors = {
        "FR": "green",
        "FL": "blue",
        "RR": "orange",
        "RL": "red",
    }

    # initialize all cells to swing color
    rgba = np.zeros((4, img_bool.shape[1], 4), dtype=float)
    rgba[:, :, :] = mcolors.to_rgba(SWING_COLOR)

    # set stance cells per leg to that leg's color
    for row, leg in enumerate(leg_labels):
        stance_mask = img_bool[row, :]  # True where stance
        rgba[row, stance_mask, :] = mcolors.to_rgba(stance_colors[leg])

    plt.imshow(
        rgba,
        aspect="auto",
        interpolation="nearest",
    )

    plt.yticks(range(4), leg_labels)
    plt.tick_params(axis="both", labelsize=TITLE_FS_EP)
    plt.xlabel("time index", fontsize=LABEL_FS_EP)
    plt.ylabel("leg", fontsize=LABEL_FS_EP)
    plt.title(f"{EVAL_POLICY} footfall pattern (colored stance per leg, white = swing)", fontsize=TITLE_FS_EP)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "rollout_footfall_pattern.png"), dpi=300)
    plt.close()



###############################################################################
# SAVE METRICS TO JSON + PRINT SUMMARY
###############################################################################
metrics = {
    "eval_task": EVAL_POLICY,
    "learning_alg": LEARNING_ALG,
    "selected_episode": {
        "return": float(best["return"]),
        "time_s": float(best["time"]),
        "terminated": bool(best["terminated"]),
        "truncated": bool(best["truncated"]),
    },
    "velocity": {
        "vx_min": vx_min,
        "vx_max": vx_max,
        "vxy_min": vxy_min,
        "vxy_max": vxy_max,
        "v_avg_xy": float(v_avg),
    },
    "distance": {
        "dist_xy_m": float(dist_xy),
        "start_xy": [float(start_xy[0]), float(start_xy[1])],
        "end_xy": [float(end_xy[0]), float(end_xy[1])],
    },
    "energy": {
        "energy_total_J": float(E),
        "power_avg_W": float(P_avg),
    },
    "robot": {
        "mass_total_kg": float(m_total),
        "g_m_s2": float(g),
    },
    "cot": {
        "cot_E_over_mgd": float(cot),
        "definition": "COT = E / (m*g*d), with d = XY distance traveled",
    },
    "gait": {
        "foot_link_ids_raw": data.get("foot_link_ids_raw", []),
        "foot_link_names_raw": data.get("foot_link_names_raw", []),
        "reorder_idx_FRFLRRRL": data.get("reorder_idx_FRFLRRRL", []),
        "duty_cycle": duty,
        "stance_swing": stance_swing,
        "footfall_events": footfall_events,
    },
}

metrics_path = os.path.join(log_dir, "rollout_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print("----- GAIT / ENERGY SUMMARY (selected episode) -----")
print(f"Distance XY: {dist_xy:.3f} m | Time: {T:.3f} s | v_avg(XY): {v_avg:.3f} m/s")
print(f"vx min/max: {vx_min:.3f} / {vx_max:.3f} m/s | vxy min/max: {vxy_min:.3f} / {vxy_max:.3f} m/s")
print(f"Energy: {E:.3f} J | Avg Power: {P_avg:.3f} W | Mass: {m_total:.3f} kg")
print(f"COT (E/(m g d)): {cot:.6f}")
if duty:
    print("Duty cycle:", {k: round(v, 3) for k, v in duty.items()})
print("Metrics saved:", metrics_path)

print("Rollout plots saved in:", log_dir)
print("Press Enter to close the simulator window.")

input()
env.close()

