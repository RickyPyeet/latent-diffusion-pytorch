import torch

def extract(data, t, x_shape):
  """
  Extracts values from specific timestep t and makes it broadcastable by reshaping it like x
  args:
    data = 1D tensor of shape (timesteps, ) such as betas, alpha_bar, etc.
    t = 1D tensor of timesteps of shape (batch_size, ). Each timestep corresponds to the timestep for that specific img in the batch
    x_shape = shape of the x_not img, used to reshape the output tensor for broadcasting
  out:
    out = tensor of shape (batch_size, *(1, )*(len(x_shape)-1)) --> if x_shape = (batch,3,224,224) --> (batch_size, 1, 1, 1)
  """
  batch_size = t.shape[0]
  data = data.gather(-1, t)
  data = data.reshape(batch_size, *((1,)*(len(x_shape)-1))) # reshape gathered values to match (batch_size, 1,1,1)
  return data.to(t.device)

def batched_diffusion_kernel(x_not, t, alpha_bars, noise = None):
  """
  Forward diffusion process q(x_t | x_0)
  args:
    x_not (tensor) = original clean image
    t (tensor) = timestep t
    alpha_bars (tensor) = alpha_bars extracted with e.g. 1000 timesteps
    noise (tensor) = random gaussian noise
  output:
    x_t (tensor) = noised image at timestep t
    noise = noise added to img
  """
  if noise is None:
    noise = torch.randn_like(x_not, requires_grad = False)
  alpha_bar_batched = extract(alpha_bars, t, x_shape = x_not.shape)
  x_t = torch.sqrt(alpha_bar_batched)*x_not + torch.sqrt(1-alpha_bar_batched)*noise
  return x_t, noise
