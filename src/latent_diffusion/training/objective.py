import torch

from src.latent_diffusion.diffusion.process import extract

def get_train_target(x_0: torch.Tensor, 
                     noise: torch.Tensor,
                     alpha_bar: torch.Tensor,
                     timestep: torch.Tensor,
                     pred_type: str = 'v'):
  """
  Compute the training target if the prediction type is 'v',
  otherwise returns target based on type [x_0, epsilon]
  """
  if pred_type == 'v':
    alpha_bars_batched = extract(alpha_bar, timestep, x_0.shape)
    target = torch.sqrt(alpha_bars_batched)*noise - torch.sqrt(1-alpha_bars_batched)*x_0
  elif pred_type == 'epsilon':
    target = noise
  elif pred_type == 'x_0':
    target = x_0
  else:
    raise SyntaxError(f"{pred_type} loss not supported, use: v, epsilon, x_0")
  return target
