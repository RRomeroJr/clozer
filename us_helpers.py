import os
import re
import shutil
import glob
import sys
from typing import Callable
chpnt = os.path.normpath("outputs/checkpoint")
def g_checkpoint_dirs():
    global chpnt
    return glob.glob(chpnt + "-*")
def g_checkpoint_dirs_sorted(desc = False):
    global chpnt
    dirs = g_checkpoint_dirs()
    re_str = chpnt.replace('\\', '\\\\') + r"-(\d+)"
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
def find_checkpoint(inp: str| int = None, silent = False) -> dict[str, str | int] | None:
    if inp == None:
        return find_latest_checkpoint(silent = silent)
    else:
        return find_checkpoint_num(inp, silent = silent)
def find_checkpoint_num(inp: str | int, silent = False) -> dict[str, str | int]:
    if isinstance(inp, int):
        chpnt_num = inp
    else:
        try:
            chpnt_num = int(inp)
        except Exception as e:
            raise Exception(f"couldn't cast {inp} to int") from e
    
    chpnt_path = f"outputs/checkpoint-{chpnt_num}"
    if os.path.exists(chpnt_path):
        chpnt = {"path": chpnt_path, "num": chpnt_num}
        if not silent: print(f"Returning checkpoint {chpnt['num']} at\n", chpnt["path"])
        return(chpnt)
    return None
def find_latest_checkpoint(silent = False) -> dict[str, str | int]:
    """
    Searches for checkpoint directories in ./outputs and returns the path to the latest one.
    
    Returns:
        str: Relative path to the latest checkpoint directory, or None if no checkpoints found.
    """
    base_dir = "./outputs"
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