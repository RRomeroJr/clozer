import os
import re
import shutil
import glob
import sys
from typing import Callable
# chpnt = os.path.normpath("outputs/checkpoint")
def g_checkpoint_dirs(chpnts_dir="./outputs"):
    chpnts_str = f"{chpnts_dir}/checkpoint-*"
    res = glob.glob(chpnts_str)
    for i in range(len(res)):
        res[i] = res[i].replace("\\", "/")
    return res
def g_checkpoint_dirs_sorted(desc = False, chpnts_dir="./outputs"):
    dirs = g_checkpoint_dirs(chpnts_dir=chpnts_dir)
    print(dirs)
    re_str = "{}/checkpoint".format(chpnts_dir.replace('\\', '/'))
    re_str += r"-(\d+)"
    print(re_str)
    pattern = re.compile(re_str)
    return sorted(dirs, key=lambda x : int(pattern.match(x)[1]), reverse=desc)

def delete_specific_directories():
    """
    Looks for specific directories in the current working directory
    and deletes them if they exist.
    """
    # List of directories to delete
    directories_to_delete = [
        "unsloth_compiled_cache",
        "_unsloth_sentencepiece_temp"
    ]
    
    # Get current working directory
    cwd = os.getcwd()
    
    for dir_name in directories_to_delete:
        dir_path = os.path.join(cwd, dir_name)
        
        # Check if directory exists
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            try:
                # Remove directory and all its contents
                shutil.rmtree(dir_path)
                print(f"Successfully deleted: {dir_name}")
            except Exception as e:
                print(f"Error deleting {dir_name}: {e}")
        else:
            print(f"Directory not found: {dir_name}")
def find_checkpoint(chpnt_num: str| int = None, silent = False, chpnts_dir = "./outputs") -> dict[str, str | int] | None:
    if chpnt_num == None:
        return find_latest_checkpoint(silent = silent, chpnts_dir=chpnts_dir)
    else:
        return find_checkpoint_num(chpnt_num, silent = silent, chpnts_dir=chpnts_dir)
def find_checkpoint_num(chpnt_num: str | int, silent = False, chpnts_dir = "./outputs") -> dict[str, str | int]:
    if isinstance(chpnt_num, int):
        chpnt_num = chpnt_num
    else:
        try:
            chpnt_num = int(chpnt_num)
        except Exception as e:
            raise Exception(f"couldn't cast {inp} to int") from e
    
    chpnt_path = f"{chpnts_dir}/checkpoint-{chpnt_num}"
    if os.path.exists(chpnt_path):
        chpnt = {"path": chpnt_path, "num": chpnt_num}
        if not silent: print(f"Returning checkpoint {chpnt['num']} at\n", chpnt["path"])
        return(chpnt)
    return None
def find_latest_checkpoint(silent = False, chpnts_dir= "./outputs") -> dict[str, str | int]:
    """
    Searches for checkpoint directories in ./outputs and returns the path to the latest one.
    
    Returns:
        str: Relative path to the latest checkpoint directory, or None if no checkpoints found.
    """
    base_dir = chpnts_dir
    latest_num = -1
    latest_checkpoint_path = None
    
    # Check if the base directory exists
    if not os.path.exists(base_dir) or not os.path.isdir(base_dir):
        print(f"Directory {base_dir} does not exist or is not a directory.")
        return None
    
    # Pattern to match "checkpoints-X" where X is a number
    pattern = re.compile(r"checkpoint-(\d+)")
    
    # Walk through all directories in the base directory
    for root, dirs, _ in os.walk(base_dir):
        for dir_name in dirs:
            # print(root, dir_name)

            match = pattern.match(dir_name)
            if match:
                checkpoint_num = int(match.group(1))
                # print(checkpoint_num)
                if checkpoint_num > latest_num:
                    latest_num = checkpoint_num
                    latest_checkpoint_path = os.path.join(root, dir_name)
    res = None
    # If we found a checkpoint, return its relative path
    if latest_checkpoint_path:
        # Convert to relative path if needed
        if latest_checkpoint_path.startswith("./"):
            res =  {"path": latest_checkpoint_path,"num": latest_num}
        else:
            res =  {"path": os.path.relpath(latest_checkpoint_path), "num": latest_num}
        if not silent: print(f"Returning checkpoint {res['num']} at\n", res["path"])
    else:
        print("No checkpoint directories found.")
    return res
def test():
    print('dirs', g_checkpoint_dirs_sorted())

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