from typing import Any, Dict, List, Tuple
import us_helpers
import pandas as pd
import matplotlib.pyplot as plt
import json
import pprint

def g_log_history_objs():
    chpnt_dirs = [us_helpers.g_checkpoint_dirs_sorted(desc=True)[0]]
    res = []
    for d in chpnt_dirs:
        with open(d + '\\trainer_state.json', 'r', encoding='UTF-8') as f:
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

def test():

    data = g_checkpoint_loss(g_log_history_objs())
    # pprint.pprint(data[:2])
    xl = [x[0] for x in data["train"]]
    x2 = [x[0] for x in data["eval"]]
    yl = [y[1] for y in data['train']]
    y2 = [y[1] for y in data["eval"]]
    # pprint.pprint(data)
    plt.plot(xl, yl)
    plt.plot(x2, y2)
    plt.show()
    # input("Continue")
    # plt.close()

if __name__ == "__main__":
    from sys import argv
    from typing import Callable
    
    # Get all callable functions defined in this module
    callables: dict[str,Callable] = {name: obj for name, obj in globals().items() if callable(obj)}
    
    if len(argv) >= 2:
        func_name = argv[1]
        if func_name in callables:
            # Call the function with any remaining arguments
            args = argv[2:] if len(argv) > 2 else []
            callables[func_name](*args)
        else:
            print(f"Error: Function '{func_name}' not found.")
    else:
        # Display available commands
        commands = list(callables.keys())
        print(f"Available commands: {', '.join(commands)}")