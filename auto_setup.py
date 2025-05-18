import os
import subprocess
"""1st you would set up your own virtual environment. Then.."""
def show_and_run(cmd: str):
    print(">", cmd)
    subprocess.run(cmd, shell=True, check=True)
# Install the static version requiremnts. == not >=
reqs_filename = "requirements_static.txt"
try:
    show_and_run(f"pip install -r {reqs_filename}")
except Exception as e:
    raise Exception(f"An error occurred while installing {reqs_filename}: {e}") from e

# Install the torch version.
torch_wheel_name = "https://download.pytorch.org/whl/cu124"
try:
    show_and_run(f"pip install torch torchvision torchaudio --index-url {torch_wheel_name}")
except Exception as e:
    raise Exception(f"An error installing torch. url, {torch_wheel_name}: {e}") from e

# Run the clone_repos script. Grabs deck assigner and note make datasets and LoRAs
try:
    show_and_run("python clone_repos.py")
except Exception as e:
    raise Exception(f"An error occurred while cloning repos: {e}") from e

# Install rrjr_py as editiable install
try:
    show_and_run("pip install -e rrjr_py_pkg --config-settings editable_mode=strict")
except Exception as e:
    raise Exception(f"Error occurred while installing rrjr_py: {e}") from e

