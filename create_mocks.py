import torch
import os

def create_mock_checkpoints():
    modalities = ["image", "video", "audio"]
    for mod in modalities:
        dir_path = f"models/{mod}"
        os.makedirs(dir_path, exist_ok=True)
        
        # Create a dummy checkpoint
        mock_state = {
            "epoch": 1,
            "state_dict": {},
            "optimizer": {},
            "best_metric": 0.99
        }
        
        filepath = f"{dir_path}/baseline_best.pt"
        torch.save(mock_state, filepath)
        print(f"Created mock checkpoint: {filepath}")

if __name__ == "__main__":
    create_mock_checkpoints()
