
import torch
from torch import nn

class DiffusionSchedule(nn.Module):
  """
  Define a Diffusion schedule that returns the value of beta and alpha_bar at time t
  init:
    timesteps (int) = 1000 like in DDPM paper
    schedule (str) = "cosine" schedule as default. Accepts: ['linear', 'cosine', 'sigmoid', 'exponential']
    device = device to move tensors to
  args:
    t (int) = instance of time to extract values of beta and alpha_bar
  out:
    beta = tensor of all betas
    beta = tensor of all betas
    alpha_bar = list of all alpha cumprods
  """
  def __init__(self, timesteps = 1000, schedule = "cosine", device = 'cpu'):
    super().__init__()
    VALID_SCHEDULES = ['linear', 'cosine', 'sigmoid', 'exponential']

    self.timesteps= timesteps
    self.schedule = schedule


    if schedule not in VALID_SCHEDULES:
      raise ValueError(f"{self.schedule} is not an accepted scheduler... please use any of the following {VALID_SCHEDULES}")

    if self.schedule == 'linear':
      self.beta = self._linear_beta_schedule(self.timesteps)
    elif self.schedule == 'cosine':
      self.beta = self._cosine_beta_schedule(self.timesteps)
    elif self.schedule == 'sigmoid':
      self.beta = self._sigmoid_beta_schedule(self.timesteps)
    else:
      self.beta = self._exponential_beta_schedule(self.timesteps)

    # Closed form noising q(x_t | x_0)
    self.alpha = 1. - self.beta
    self.alpha_bar = torch.cumprod(self.alpha, dim = 0).requires_grad_(False)

    self.beta = self.beta.to(device)
    self.alpha = self.alpha.to(device)
    self.alpha_bar = self.alpha_bar.to(device)

  def _linear_beta_schedule(self, timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps, requires_grad = False)

  def _cosine_beta_schedule(self, timesteps):
    s = 0.008 # Improved denoising diffusion model paper
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, requires_grad = False)
    alpha_bar = torch.cos(((x / timesteps) + s)/(1+s)*(torch.pi/2))**2
    alpha_bar = alpha_bar / alpha_bar[0] # Normalize
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1]) # Turn cosine (alpha_bar) into a noise schedule beta
    betas = torch.clip(betas, 0.0001, 0.9999)
    return betas

  def _exponential_beta_schedule(self, timesteps):
    beta_min = 0.0001
    beta_max = 0.02
    x = torch.linspace(0, timesteps-1, timesteps, requires_grad = False)
    beta = beta_min * (beta_max / beta_min)**((x)/(timesteps - 1))
    return beta

  def _sigmoid_beta_schedule(self, timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    k = 10
    x = torch.linspace(0, timesteps-1, timesteps, requires_grad = False)
    tau = x / (timesteps-1)
    sigmoid = torch.sigmoid(k * (tau - 0.5))
    beta = beta_start + (beta_end - beta_start) * sigmoid
    return beta

  def forward(self):
    return self.beta, self.alpha, self.alpha_bar
