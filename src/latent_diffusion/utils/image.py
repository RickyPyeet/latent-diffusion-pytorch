import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from PIL import Image
from typing import List, Tuple


def denormalize_samples(samples: torch.Tensor) -> torch.Tensor:
  """Denormalize the tensor samples - converting them from [-1, 1] to [0, 1]"""
  samples = samples.clamp(-1, 1)
  samples = (samples + 1)/2
  return samples

def tensor_to_numpy(samples: torch.Tensor) -> np.ndarray:
  """
  Converts a batch of tensors (B, C, H, W) [-1, 1]
  to NumPy arrays of shape (B, H, W, C) [0, 255] in uint8
  """
  samples = denormalize_samples(samples).detach().cpu()
  samples = samples.permute(0,2,3,1)
  samples = (samples * 255).round()
  samples = samples.numpy().astype(np.uint8)
  return samples

def tensor_to_pil(samples: torch.Tensor) -> List[Image.Image]:
  """
  Converts a batch of tensors (B, C, H, W) into a list of PIL Images
  """
  numpy_images = tensor_to_numpy(samples)
  pil_images = [Image.fromarray(img) for img in numpy_images]
  return pil_images


def plot_image_grid(samples: torch.Tensor, 
                    ncols: int = 4,
                    figsize: Tuple[int, int] = (8, 8),
                    title: str | None = None,
                    show_img: bool = True,
                    save_img: bool = False,
                    save_path: str | None = None):
  """
  Plots the batch of tensors into a grid of images and saves it if required
  """
  if len(samples.shape) != 4:
    raise ValueError(f"Expected images with shape (B, C, H, W), got {samples.shape} instead")
  
  if save_img and (save_path is None):
    raise ValueError(f"Required to save the image but no path was specified")

  if ncols == 0:
    raise ValueError(f"Ncols must be at least 1")

  np_images = tensor_to_numpy(samples)

  batch_size = len(np_images)
  ncols = min(ncols, batch_size)
  nrows = math.ceil(batch_size / ncols)

  fig, ax = plt.subplots(nrows, ncols, figsize = figsize, layout = 'constrained')
  ax = np.atleast_2d(ax)

  if title is not None:
    fig.suptitle(title, fontsize=16)

  i = 0
  for row in range(nrows):
    for col in range(ncols):
      if i < batch_size:
        ax[row, col].imshow(np_images[i])

      ax[row, col].axis(False)
      i += 1
  
  if show_img:
    plt.show()

  if save_img:
    save_path = os.path.join(save_path, "sample_grid.png")
    fig.savefig(fname = save_path)
