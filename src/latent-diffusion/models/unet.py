
import torch
from torch import nn
from torch.nn import functional as F

from src.ddpm.models.attention import SelfAttention, LinearAttention
from src.ddpm.models.init import init_conv
from src.ddpm.models.embeddings import SinPositionalEmbedding
from src.ddpm.models.resnet import ResnetBlock

### U-Net Helper Blocks ###

class GroupNormBlock(nn.Module):
  def __init__(self, dim, fn):
    super().__init__()
    self.fn = fn
    self.norm = nn.GroupNorm(8, dim) # changed num_groups from 1 to 8

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
  """Downsampling using the space-to-depth technique followed by 1x1 conv -> better than transposed convolution"""
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


### UNET ###
class DenoisingUNet(nn.Module):
  def __init__(self,
               input_dim,
               output_channels = None,
               initial_dim = None,
               channels = 3,
               dimension_multiplier = (1,2,4,8),
               groupnorm_groups = 4,):
    super().__init__()
    self.output_channels = output_channels

    in_channels = channels

    initial_dim = initial_dim if initial_dim is not None else input_dim

    ### Creating Dimensions List
    dimensions = [initial_dim, *map(lambda m: m*input_dim, dimension_multiplier)] # Creates a list of dimensions used in the Unet
    input_output_dim_list = list(zip(dimensions[:-1], dimensions[1:])) # tuples of input-output dimensions for each layer of the unet e.g. [(3,64),(64,128), etc.]

    ### Channel Projection Conv
    self.initial_conv = nn.Conv2d(in_channels, initial_dim, kernel_size = 1, padding = 0) # Channel projection
    init_conv(self.initial_conv)

    ### Setup Time Embedding
    time_dim = input_dim * 4 # standard, the time embedding is made larger (4×) so it has enough expressive power to modulate all the feature maps in the network

    self.time_mlp = nn.Sequential(SinPositionalEmbedding(input_dim), # Shape (B, input_dim)
                                  nn.Linear(in_features = input_dim, out_features = time_dim),
                                  nn.GELU(), # Used to preserve smoothness of the sinusoidal signal + Linear / GELU / Linear comes from Transformers
                                  nn.Linear(in_features = time_dim, out_features = time_dim)) # Final shape (B, time_dim)

    ### Down path
    self.downs = nn.ModuleList([])
    downs_len = len(input_output_dim_list)

    for i, (in_dim, out_dim) in enumerate(input_output_dim_list):
      is_last = i >= (downs_len - 1)
      self.downs.append(
          nn.ModuleList([
              ResnetBlock(in_channels = in_dim, out_channels = in_dim, time_embedded_dim=time_dim, groups = groupnorm_groups),
              ResnetBlock(in_channels = in_dim, out_channels = in_dim, time_embedded_dim=time_dim, groups = groupnorm_groups),
              Residual(GroupNormBlock(in_dim, LinearAttention(in_channels = in_dim))),
              Downsample(in_dim, out_dim) if not is_last
              else nn.Conv2d(in_dim, out_dim, kernel_size = 3, padding = 1) # Stop downsampling to keeo a good dimension for self-attention (H*W x H*W)
          ]))
      if is_last:
        init_conv(self.downs[-1][-1])

    ### Bottleneck
    bottle_dim = dimensions[-1]
    self.bottleneck_1 = ResnetBlock(in_channels = bottle_dim, out_channels = bottle_dim, time_embedded_dim = time_dim, groups = groupnorm_groups)
    self.bottleneck_attention = Residual(GroupNormBlock(bottle_dim, SelfAttention(bottle_dim)))
    self.bottleneck_2 = ResnetBlock(in_channels = bottle_dim, out_channels = bottle_dim, time_embedded_dim = time_dim, groups = groupnorm_groups)
    self.bottleneck_attention_2 = Residual(GroupNormBlock(bottle_dim, SelfAttention(bottle_dim)))
    self.bottleneck_3 = ResnetBlock(in_channels = bottle_dim, out_channels = bottle_dim, time_embedded_dim = time_dim, groups = groupnorm_groups)

    ### Upwards path
    self.ups = nn.ModuleList([])

    for i, (in_dim, out_dim) in enumerate(reversed(input_output_dim_list)):
      is_last = (i == (downs_len - 1))
      self.ups.append(
          nn.ModuleList([
              ResnetBlock(in_dim + out_dim, out_dim, time_embedded_dim = time_dim, groups = groupnorm_groups),
              ResnetBlock(in_dim + out_dim, out_dim, time_embedded_dim = time_dim, groups = groupnorm_groups),
              Residual(GroupNormBlock(out_dim, LinearAttention(in_channels = out_dim))),
              Upsample(out_dim, in_dim) if not is_last
              else nn.Conv2d(in_channels = out_dim, out_channels = in_dim, kernel_size = 3, padding = 1)
      ]))
      if is_last:
        init_conv(self.ups[-1][-1])

    ### Output layer
    self.out_dim = output_channels if output_channels is not None else channels
    self.out_resblock = ResnetBlock(in_channels = input_dim * 2, out_channels = input_dim, time_embedded_dim=time_dim, groups = groupnorm_groups)
    self.out_conv = nn.Conv2d(input_dim, self.out_dim, 1)
    init_conv(self.out_conv)

  def forward(self, x, time, cond_emb = None):
    # Channel projection
    x = self.initial_conv(x)
    r = x.clone()

    # Time embedding
    t = self.time_mlp(time) #embed timestep with sinposembedding and then enrich it with nn.Linear with input_dim*4

    # Conditioning
    if cond_emb is not None:
      t = t + cond_emb

    # Down path
    h = [] # skip connections list
    for block1, block2, linear_attention, downsample in self.downs:
      x = block1(x, t)
      h.append(x)

      x = block2(x, t)
      x = linear_attention(x)

      h.append(x)

      x = downsample(x)

    # Bottleneck
    x = self.bottleneck_1(x, t)
    x = self.bottleneck_attention(x)
    x = self.bottleneck_2(x, t)
    x = self.bottleneck_attention_2(x)
    x = self.bottleneck_3(x, t)

    # Upward path
    for block1, block2, linear_attention, upsample in self.ups:
      x = torch.cat((x, h.pop()), dim = 1)
      x = block1(x, t)

      x = torch.cat((x, h.pop()), dim = 1)
      x = block2(x, t)

      x = linear_attention(x)

      x = upsample(x)

    # Output layer
    x = torch.cat((x, r), dim = 1)
    x = self.out_resblock(x, t)
    return self.out_conv(x)
