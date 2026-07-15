
import torch
from torch import nn

from src.ddpm.models.init import init_conv


def scale_shift_extract(scale_shift):
  """
  From a conditional embedded timestep extract its gamma (scale) and beta (shift) values for FiLM injection.
  args:
    - scale_shift (torch.Tensor) - output of the time_conditioning. Shape (batch_size, embedding_dim*2)
  output:
    - (scale, shift) (Tuple) = tuple containing `scale` and `shift` tensors to add to our FiLM conditioning
  """
  scale_shift = scale_shift[:,:,None, None]
  return scale_shift.chunk(2, dim = 1)

class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, groups = 8):
    """
    Conv2d block: conv2d -> groupnorm -> FiLM injection (if present) -> activation
    """
    super().__init__()
    self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, padding = 1)
    init_conv(self.conv)
    self.norm = nn.GroupNorm(groups, out_channels)
    self.activation = nn.SiLU()

  def forward(self, x, scale_shift = None):
    x = self.norm(self.conv(x))
    if scale_shift is not None:
      gamma, beta = scale_shift
      x = x * (gamma + 1) + beta

    return self.activation(x)

class ResnetBlock(nn.Module):
  """
  Resnet Block: convblock (with FiLM) -> convblock -> residual connection
  """
  def __init__(self, in_channels, out_channels, groups = 8, time_embedded_dim = None):
    super().__init__()
    self.conv1 = ConvBlock(in_channels, out_channels, groups = groups)
    self.conv2 = ConvBlock(out_channels, out_channels, groups = groups)

    # FiLM Conditioning - creating gamma (scale) and beta (shift)
    self.mlp = (nn.Sequential(nn.SiLU(), nn.Linear(time_embedded_dim, out_channels*2)) # time conditioning, to get gamma and beta (out_channels * 2)
                if time_embedded_dim is not None else None)

    self.residual = nn.Conv2d(in_channels, out_channels, kernel_size = 1) if in_channels != out_channels else nn.Identity()
    if isinstance(self.residual, nn.Conv2d):
      init_conv(self.residual)

  def forward(self, x, time_embedding = None):
    scale_shift = None
    if (self.mlp is not None) and (time_embedding is not None):
      scale_shift = scale_shift_extract(self.mlp(time_embedding)) # Extract gamma and beta (scale and shift)

    y = self.conv1(x, scale_shift) # Inject time conditioning to first convblock
    y = self.conv2(y)

    return  y + self.residual(x)
