import torch
from torch import nn
from dataclasses import dataclass
from typing import Literal

from src.latent_diffusion.models.attention import AttentionStack
from src.latent_diffusion.models.resnet import ResnetBlock
from src.latent_diffusion.models.blocks import Upsample, Downsample
from src.latent_diffusion.models.init import init_conv

attn_type = Literal['none', 'self', 'cross', 'self_cross']

@dataclass(frozen = True)
class UNetConfig:
  in_channels: int
  out_channels: int
  attn: attn_type
  updownsample: bool

class DownStage(nn.Module):
  def __init__(self,
               in_channels,
               out_channels,
               time_emb_dim,
               groups = 32,
               attn_type = 'none',
               context_dim = None,
               heads = 8,
               head_dim = 64,
               downsample = True):
    """
    Creates a down path layer consisting of: Resnet block -> Resnet block -> Attention stack
    """
    super().__init__()

    self.block1 = ResnetBlock(in_channels = in_channels,
                              out_channels = in_channels,
                              groups = groups,
                              time_embedded_dim = time_emb_dim)

    self.block2 = ResnetBlock(in_channels = in_channels,
                              out_channels = in_channels,
                              groups = groups,
                              time_embedded_dim = time_emb_dim)

    self.attn = AttentionStack(channels = in_channels,
                               attn_type = attn_type,
                               context_dim = context_dim,
                               heads = heads,
                               head_dim = head_dim)
    self.downsample = (Downsample(in_channels = in_channels, out_channels = out_channels)
                      if downsample
                      else nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, padding = 1))
    if not downsample:
      init_conv(self.downsample)

  def forward(self, x, t, context = None):
    h = []
    x = self.block1(x, t)
    # print(f"Downstage after first resnet block:\t x:{x.shape}")
    h.append(x)
    x = self.block2(x, t)
    # print(f"Downstage after second resnet block:\t x:{x.shape}")
    h.append(x)
    x = self.attn(x, context)
    x = self.downsample(x)
    # print(f"Downstage after downsample:\t\t x:{x.shape}\n---------------------")
    return x, h

class UpStage(nn.Module):
  def __init__(self,
               in_channels, # e.g. in_ch = 512
               out_channels, # e.g. out_ch = 256
               time_emb_dim,
               groups,
               attn_type = 'none',
               context_dim = None,
               heads = 8,
               head_dim = 64,
               upsample = True):
    """
    Creates an Up path layer made up of: Resnet -> Resnet -> Attention stack and skip connections from down stage
    """
    super().__init__()

    self.block1 = ResnetBlock(in_channels = in_channels + out_channels, # in_ch + out_ch
                              out_channels = in_channels, # out_ch
                              groups = groups,
                              time_embedded_dim = time_emb_dim)

    self.block2 = ResnetBlock(in_channels = in_channels + out_channels, # out_ch + out_ch
                              out_channels = in_channels, # out_ch
                              groups = groups,
                              time_embedded_dim = time_emb_dim)

    self.attn = AttentionStack(channels = in_channels, # out_ch
                               attn_type = attn_type,
                               context_dim = context_dim,
                               heads = heads,
                               head_dim = head_dim)

    self.up = (Upsample(in_channels, out_channels) if upsample # out_ch out_ch
               else nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, padding = 1))

    if not upsample:
      init_conv(self.up)

  def forward(self, x, t, h, context = None):
    # print(f"Dimension of x before adding h:\t\t x:{x.shape}")
    x = torch.cat((x, h.pop()), dim = 1)
    # print(f"Dimension of x after adding h:\t\t x:{x.shape}")
    x = self.block1(x, t)
    # print(f"Dimension of x after first resnet:\t x:{x.shape}")
    x = torch.cat((x, h.pop()), dim = 1)
    # print(f"Dimension of x after adding h:\t\t x:{x.shape}")
    x = self.block2(x, t)
    # print(f"Dimension of x after second resnet:\t x:{x.shape}")
    x = self.up(self.attn(x, context))
    # print(f"Dimension of x after up block:\t\t x:{x.shape}\n--------------------")
    return x