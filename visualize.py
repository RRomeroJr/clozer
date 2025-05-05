import sys
import time
from typing import Any, Dict, List, Tuple

from matplotlib import ticker
import us_helpers
import pandas as pd
import matplotlib.pyplot as plt
import json
import pprint
import os
def g_lastest_train_state(target_dir):
    try:
        lastest_chpnt = [dir for dir in us_helpers.g_checkpoint_dirs_sorted(desc=True, chpnts_dir=target_dir)][0]
        with open(lastest_chpnt + '/trainer_state.json', 'r', encoding='UTF-8') as f:
            return json.loads(f.read())
    except:
        return None
def g_log_history_objs(target_dir):
    # target_dir = "outputs/note_maker"
    # chpnt_dirs = [dirs[0] for dirs in us_helpers.g_checkpoint_dirs_sorted(desc=True, chpnts_dir=target_dir)]
    # res = []
    # for d in chpnt_dirs:
    #     with open(d + '\\trainer_state.json', 'r', encoding='UTF-8') as f:
    #         res.extend(json.loads(f.read())['log_history'])
    # return res
    chpnt_dirs = [dir for dir in us_helpers.g_checkpoint_dirs_sorted(desc=True, chpnts_dir=target_dir)]
    res = []
    if len(chpnt_dirs) > 0:
        with open(chpnt_dirs[0] + '\\trainer_state.json', 'r', encoding='UTF-8') as f:
            res.extend(json.loads(f.read())['log_history'])
    return res
def g_checkpoint_loss(log_history_objs: dict[str, Any]) -> Dict[str, List[Tuple]]:
    res_train: List[Tuple] = []
    res_eval: List[Tuple] = []

    for o in log_history_objs:
        if "loss" in o and "step" in o:
            res_train.append((o['step'], o['loss']))
        if "eval_loss" in o and "step" in o:
            res_eval.append((o['step'], o['eval_loss']))
    return {"train": res_train, "eval": res_eval}

def get_plot_data(target_dir="outputs/note_maker"):
    initial_eval_path = os.path.join(target_dir, "initial_eval.json")
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    if os.path.exists(initial_eval_path):
        with open(initial_eval_path, 'r', encoding='UTF-8') as f:
            initial_eval = json.loads(f.read())
            if "eval_loss" in initial_eval:
                x2.append(0)
                y2.append(initial_eval["eval_loss"])
    """Gather and prepare the data for plotting."""
    data = g_checkpoint_loss(g_log_history_objs(target_dir))
    x1.extend(x[0] for x in data["train"])
    y1.extend(y[1] for y in data['train'])
    x2.extend(x[0] for x in data["eval"])
    y2.extend(y[1] for y in data["eval"])
    return x1, y1, x2, y2
def g_learning_rate_data(target_dir="outputs/note_maker"):
    """Gather and prepare the data for plotting."""
    x = []
    y = []
    initial_eval_path = os.path.join(target_dir, "initial_eval.json")
    if os.path.exists(initial_eval_path):
        with open(initial_eval_path, 'r', encoding='UTF-8') as f:
            initial_eval = json.loads(f.read())
            if "initial_learning_rate" in initial_eval:
                x.append(0)
                y.append(initial_eval["initial_learning_rate"])
    objs = g_log_history_objs(target_dir)
    for o in objs:
        if "learning_rate" in o and "step" in o:
            x.append(o["step"])
            y.append(o["learning_rate"])
    return x, y

def monitor_training(target_dir="outputs/note_maker", refresh_interval=5):
    print(f"monitoring {target_dir}")
    # Initial setup
    plt.ion()
    fig, ax1 = plt.subplots()
    plt.subplots_adjust(right=0.84)
    
    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()
    
    # Set the window title to the target directory
    fig.canvas.manager.set_window_title(f"Training Loss: {target_dir}")
    
    # Set up the primary axes (for loss values)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax1.grid(True)
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss")
    
    # Secondary axis for learning rate
    ax2.set_ylabel("Learning Rate")
    
    # Get initial data
    lastest_trainer_state = g_lastest_train_state(target_dir)
    x1, y1, x2, y2 = get_plot_data(target_dir)
    lr_x, lr_y = g_learning_rate_data(target_dir)
    # ax2.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    def get_title_str():
        return f"No source, Gradient 8, Inverse Sqrt, Start 1e-4, Multiple Regens:\n{os.path.basename(target_dir)} (current step: {lastest_trainer_state['global_step'] if lastest_trainer_state != None else 0})"
    ax1.set_title(get_title_str())

    # Create initial plots
    train_line, = ax1.plot(x1, y1, marker='o', markersize=4, label='Training Loss')
    eval_line, = ax1.plot(x2, y2, marker='o', markersize=4, label='Evaluation Loss')
    
    # Add learning rate line on the secondary axis with transparency
    lr_line, = ax2.plot(lr_x, lr_y, 'g-', alpha=0.35, label='Learning Rate')
    
    # Create a combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    fig.canvas.draw()

    # Use timer instead of sleep
    last_time = time.time()
    last_refresh = time.time()

    # Main loop
    try:
        while plt.fignum_exists(fig.number):
            current_time = time.time()
            
            # Check if it's time to refresh
            if current_time - last_refresh >= refresh_interval:
                # Get new data using the same function
                try:
                    lastest_trainer_state = g_lastest_train_state(target_dir)
                    x1, y1, x2, y2 = get_plot_data(target_dir)
                    lr_x, lr_y = g_learning_rate_data(target_dir)
                    # ax2.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
                    
                    ax1.set_title(get_title_str())

                    # Update existing line data
                    train_line.set_data(x1, y1)
                    eval_line.set_data(x2, y2)
                    lr_line.set_data(lr_x, lr_y)
                    
                    # Update axis limits to fit the new data
                    ax1.relim()
                    ax1.autoscale_view()
                    ax2.relim()
                    ax2.autoscale_view()
                    
                    # Redraw the canvas
                    fig.canvas.draw_idle()
                except Exception as e:
                    print(f"Error updating plot: {e}")
                    
                last_refresh = current_time
            
            # Print countdown
            if current_time - last_time >= 1:
                seconds_since_refresh = int(current_time - last_refresh)
                last_time = current_time
            
            # Process GUI events without blocking
            plt.pause(0.01)
            
    except Exception as e:
        print(f"Error or window closed: {e}")
        print(sys.exc_info())

    print("Window was closed or program terminated")

if __name__ == "__main__":
    print("visualize main")
    if len(sys.argv) >= 2:
        monitor_training(sys.argv[1])
    else:
        monitor_training()
    # monitor_training("outputs/objs_to_src")