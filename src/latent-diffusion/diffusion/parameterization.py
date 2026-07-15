import torch

def compute_x0_from_epsilon(x_t, epsilon, alpha_bar_t):
  """Used if predicting noise epsilon"""
  x_0 = x_t - torch.sqrt(1-alpha_bar_t)*epsilon
  x_0 = x_0 / torch.sqrt(alpha_bar_t)
  return x_0

def compute_epsilon_from_x0(x_t, x_0, alpha_bar_t):
  """Used if predicting clean image x_0"""
  epsilon = x_0 * torch.sqrt(alpha_bar_t)
  epsilon = (x_t - epsilon) / torch.sqrt(1-alpha_bar_t)
  return epsilon

def compute_x0_epsilon_from_v(x_t, v_pred, alpha_bar_t):
  """Used if predicting velocity v"""
  x_0 = torch.sqrt(alpha_bar_t)*x_t - torch.sqrt(1-alpha_bar_t)*v_pred
  epsilon = torch.sqrt(alpha_bar_t)*v_pred + torch.sqrt(1-alpha_bar_t)*x_t
  return x_0, epsilon

def compute_sigma_t(alpha_bar_t, alpha_bar_t_previous, eta = 0.8):
  """Compute sigma for DDIM"""
  sigma_t = torch.sqrt((1-alpha_bar_t_previous)/(1-alpha_bar_t) * (1 - alpha_bar_t/alpha_bar_t_previous))
  sigma_t = eta * sigma_t
  return sigma_t
