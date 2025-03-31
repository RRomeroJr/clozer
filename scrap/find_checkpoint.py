import json

from us_helpers import find_checkpoint

path = find_checkpoint()["path"] + "/trainer_state.json"
print(path)
with open(path, "r", encoding="UTF-8") as f:
    last_trainer_state = json.loads(f.read())
print("final eval..", last_trainer_state["log_history"][-1])