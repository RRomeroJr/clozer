import os
import json
from pathlib import Path

# Default value for the Anki collection path
anki_col_path = None

# Try to load settings from settings.json
settings_path = Path("settings/settings.json")

try:
    if settings_path.exists():
        with open(settings_path, "r") as f:
            settings: dict = json.load(f)
        
        # Extract Anki collection paths
        anki_paths = settings.get("anki_collection_paths", [])
        
        # Try each path until finding a valid one
        for path in anki_paths:
            # Expand environment variables
            expanded_path = os.path.expandvars(path)
            
            # Check if the path exists
            if os.path.exists(expanded_path):
                anki_col_path = expanded_path
                # print(f"Found valid Anki collection path: {expanded_path}")
                break
        
        if anki_col_path is None and anki_paths:
            print("None of the specified Anki collection paths were valid.")
    else:
        print(f"Settings file not found at {settings_path}")
except Exception as e:
    print(f"Error loading settings: {e}")

# Example usage when the module is run directly
if __name__ == "__main__":
    print(f"Anki collection path: {anki_col_path}")