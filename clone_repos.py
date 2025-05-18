import os
import subprocess

# Define the repositories and their destination paths
repos_to_clone = [
    {
        "url": "https://huggingface.co/RRomeroJr/clozer_deck_assigner_lora",
        "dest": "./deck_assigner_lora"
    },
    {
        "url": "https://huggingface.co/RRomeroJr/clozer_note_maker_lora",
        "dest": "./note_maker_lora"
    },
    {
        "url": "https://huggingface.co/datasets/RRomeroJr/clozer-deck-assigner",
        "dest": "./datasets/examples/deck-assigner"
    },
    {
        "url": "https://huggingface.co/datasets/RRomeroJr/clozer-note-maker",
        "dest": "./datasets/examples/note-maker"
    }
]

# Create any necessary parent directories
for repo in repos_to_clone:
    os.makedirs(os.path.dirname(repo["dest"]), exist_ok=True)

# Clone each repository
for repo in repos_to_clone:
    print(f"Cloning {repo['url']} to {repo['dest']}...")
    try:
        subprocess.run(["git", "clone", repo["url"], repo["dest"]], check=True)
        print(f"Successfully cloned {repo['url']}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone {repo['url']}: {e}")
    except Exception as e:
        print(f"An error occurred while cloning {repo['url']}: {e}")

print("Cloning complete!")