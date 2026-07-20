import torch
import torch.nn.functional as F
from torch import nn

from src.latent_diffusion.models.init import init_conv


### U-Net Helper Blocks ###
class GroupNormBlock(nn.Module):
  def __init__(self, dim, fn, groups = 8):
    super().__init__()
    self.fn = fn
    self.norm = nn.GroupNorm(groups, dim)

  def forward(self, x, *args, **kwargs):
    return self.fn(self.norm(x),*args, **kwargs)

class Residual(nn.Module):
  """Apply a residual connection -> out = x + fn(x)"""
  def __init__(self, fn):
    super().__init__()
    self.fn = fn
  def forward(self, x, *args, **kwargs):
    return self.fn(x, *args, **kwargs) + x

class Upsample(nn.Module):
  """Upsampling block without transposed convolutions. Using F.interpolate instead"""
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)

    init_conv(self.conv)
  def forward(self, x):
    x = F.interpolate(x, scale_factor = 2, mode = 'nearest')
    return self.conv(x)

class Downsample(nn.Module):
  """Downsampling using the space-to-depth technique followed by 1x1 conv"""
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.Conv2d(in_channels*4, out_channels, kernel_size = 1)
    init_conv(self.conv)
  def forward(self, x):
    b, c, h, w = x.shape
    x = x.reshape(b, c, h//2, 2, w//2, 2)
    x = x.permute(0,1,3,5,2,4)
    x = x.reshape(b, c*4, h//2, w//2) # SPACE-TO-DEPTH downsampling
    return self.conv(x)