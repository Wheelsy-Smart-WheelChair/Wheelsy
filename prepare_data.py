import os
from pathlib import Path

def setup_project():
    current_path = Path(__file__).resolve().parent.parent
    
    folders = ['data', 'models-versions', 'weights']
    
    for folder in folders:
        folder_path = current_path / folder
        os.makedirs(folder_path, exist_ok=True)
        print(f" Folder created/exists: {folder}")

if __name__ == "__main__":
    setup_project()