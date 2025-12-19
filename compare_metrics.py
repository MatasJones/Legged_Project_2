import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Adjusted Formatting Flags for better fit
TITLE_FS = 24
LABEL_FS = 22
TICK_FS  = 22
LINE_WIDTH = 4

def load_metrics(run_name):
    """Loads JSON metrics for a specific policy/task folder."""
    path = f"./logs/intermediate_models/{run_name}/rollout_metrics.json"
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)

def extract_summary(data):
    """Extracts relevant comparison metrics from the JSON structure."""
    # Average velocity across the run
    v_avg = data.get("velocity", {}).get("v_avg_xy", 0)
    # Total distance traveled
    dist = data.get("distance", {}).get("dist_xy_m", 0)
    # Cost of Transport (Efficiency)
    cot = data.get("cot", {}).get("cot_E_over_mgd", 0)
    # Average power consumption in Watts
    pwr = data.get("energy", {}).get("power_avg_W", 0)
    
    gait_data = data.get("gait", {}).get("stance_swing", {})
    step_periods = [leg.get("step_period_mean_s", 0) for leg in gait_data.values()]
    avg_step_period = np.mean(step_periods) if step_periods else 0
    
    return {
        "Velocity (m/s)": v_avg,
        "Distance (m)": dist,
        "COT": cot,
        "Avg Power (W)": pwr,
        "Step Period (s)": avg_step_period
    }

def plot_comparisons():
    runs = ["TASK_STAIRS", "TASK_SLOPES", "VEL_TROT", "VEL_WALK"]
    all_results = {}
    
    for run in runs:
        data = load_metrics(run)
        if data:
            all_results[run] = extract_summary(data)
    
    if not all_results:
        print("No data found.")
        return

    output_dir = "extra plots"
    os.makedirs(output_dir, exist_ok=True)

    metrics_to_plot = list(next(iter(all_results.values())).keys())
    colors = ['#2C3E50', '#34495E', '#5D6D7E', '#7F8C8D']

    for metric in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        available_runs = [run for run in runs if run in all_results]
        values = [all_results[run][metric] for run in available_runs]
        
        bars = ax.bar(available_runs, values, color=colors[:len(available_runs)], 
                      edgecolor='#1B2631', linewidth=2, width=0.5)
        
        if values and max(values) > 0:
            ax.set_ylim(0, max(values) * 1.2)
        
        ax.set_title(f"{metric} Comparison", fontsize=TITLE_FS, fontweight='bold', pad=30)
        ax.set_ylabel(metric, fontsize=LABEL_FS)
        ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (max(values)*0.02),
                    f'{height:.2f}', ha='center', va='bottom', 
                    fontsize=TICK_FS, fontweight='bold', clip_on=False)

        for spine in ax.spines.values():
            spine.set_linewidth(LINE_WIDTH/2)

        fig.tight_layout()
        
        safe_name = metric.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "").lower()
        save_path = os.path.join(output_dir, f"formal_compare_{safe_name}.png")
        
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Saved (Fixed Labels): {save_path}")

if __name__ == "__main__":
    plot_comparisons()