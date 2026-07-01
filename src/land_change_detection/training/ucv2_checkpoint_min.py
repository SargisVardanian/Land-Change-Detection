from pathlib import Path
import torch

def save_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)

def load_state(path: Path, device):
    return torch.load(path, map_location=device)
