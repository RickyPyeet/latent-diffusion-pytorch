import datetime
from pathlib import Path
import torch

def save_checkpoint(name, 
                    checkpoint, 
                    checkpoint_path):
  checkpoint_path = Path(checkpoint_path)

  if not checkpoint_path.exists():
    checkpoint_path.mkdir(parents = True, exist_ok = True)

  date = datetime.datetime.now()
  date = "_".join(date.strftime("%c").strip().split())

  save_path = checkpoint_path / f"{date}_{name}"

  if save_path.suffix not in ['.pt', '.pth']:
    save_path = save_path.with_suffix('.pt')

  print(f"[INFO] Saving {save_path}...")
  torch.save(obj = checkpoint, f = save_path)
  print(f"[INFO] Checkpoint saved! :)")

  return save_path

def load_checkpoint(checkpoint_path, 
                    model,
                    optimizer = None,
                    ema = None,
                    device = 'cpu'):
  checkpoint = torch.load(checkpoint_path, map_location = device)
  model.load_state_dict(checkpoint['model_state_dict'])
  if optimizer is not None:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  if ema is not None:
    ema.load_state_dict(checkpoint['ema_state_dict'])

  return checkpoint
