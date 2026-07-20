
import torch
from torch import nn

from src.latent_diffusion.models.attention import AttentionStack
from src.latent_diffusion.models.init import init_conv
from src.latent_diffusion.models.embeddings import SinPositionalEmbedding
from src.latent_diffusion.models.resnet import ResnetBlock
from src.latent_diffusion.models.blocks import GroupNormBlock, Residual
from src.latent_diffusion.models.stages import DownStage, UpStage, UNetConfig

### UNET ###
class UNet(nn.Module):
  def __init__(self,
               input_dim = 64, # base feature dimension
               output_channels = None, # output channels
               initial_dim = None, # first layer channel dim if different from input_dim
               channels = 4, # input img channels
               context_dim = 768,
               dimension_multiplier = (1,2,4,8),
               attn_config = ('none', 'cross', 'self_cross', 'self_cross'),
               groupnorm_groups = 32,
               self_condition = False): # if model receives a previous prediction as extra input

    super().__init__()
    self.self_condition = self_condition
    self.output_channels = output_channels

    in_channels = channels * (2 if self_condition else 1) # Doubles the channels if there is previous prediction as extra input

    initial_dim = initial_dim if initial_dim is not None else input_dim

    ### Creating Dimensions List
    dimensions = [initial_dim, *map(lambda m: m*input_dim, dimension_multiplier)] # Creates a list of dimensions used in the Unet
    input_output_dim_list = list(zip(dimensions[:-1], dimensions[1:])) # tuples of input-output dimensions for each layer of the unet e.g. [(3,64),(64,128), etc.]

    #---------------------------------------------------
    down_stages = [UNetConfig(in_channels = in_ch,
                              out_channels = out_ch,
                              attn = attn_config[i],
                              updownsample = i < len(input_output_dim_list)-1
                              ) for i, (in_ch, out_ch) in enumerate(input_output_dim_list)]

    up_stages = [UNetConfig(in_channels = out_ch,
                            out_channels = in_ch,
                            attn = list(reversed(attn_config))[i],
                            updownsample = i < len(input_output_dim_list)-1
                            ) for i, (in_ch, out_ch) in enumerate(reversed(input_output_dim_list))]

    #---------------------------------------------------
    ### Channel Projection Conv
    self.initial_conv = nn.Conv2d(in_channels, initial_dim, kernel_size = 1, padding = 0) # Channel projection
    init_conv(self.initial_conv)

    ### Setup Time Embedding
    time_dim = input_dim * 4 # standard, the time embedding is made larger (4×) so it has enough expressive power to modulate all the feature maps in the network

    self.time_mlp = nn.Sequential(SinPositionalEmbedding(input_dim), # Shape (B, input_dim)
                                  nn.Linear(in_features = input_dim, out_features = time_dim),
                                  nn.GELU(), # Used to preserve smoothness of the sinusoidal signal + Linear / GELU / Linear comes from Transformers
                                  nn.Linear(in_features = time_dim, out_features = time_dim)) # Final shape (B, time_dim)

    #----------------DOWN PATH----------------------

    self.downs = nn.ModuleList([DownStage(in_channels = stage.in_channels,
                                          out_channels = stage.out_channels,
                                          time_emb_dim = time_dim,
                                          groups = groupnorm_groups,
                                          attn_type = stage.attn,
                                          context_dim = context_dim,
                                          heads = 8,
                                          head_dim = 64,
                                          downsample = stage.updownsample) for stage in down_stages])

    #----------------BOTTLENECK----------------------
    bottle_dim = dimensions[-1]
    self.bottleneck_1 = ResnetBlock(in_channels = bottle_dim, out_channels = bottle_dim, time_embedded_dim = time_dim, groups = groupnorm_groups)
    self.bottleneck_attention_1 = AttentionStack(channels = bottle_dim,
                                                attn_type = 'self_cross',
                                                context_dim = context_dim,
                                                heads = 8,
                                                head_dim = 64)
    self.bottleneck_2 = ResnetBlock(in_channels = bottle_dim, out_channels = bottle_dim, time_embedded_dim = time_dim, groups = groupnorm_groups)
    self.bottleneck_attention_2 = AttentionStack(channels = bottle_dim,
                                                attn_type = 'self_cross',
                                                context_dim = context_dim,
                                                heads = 8,
                                                head_dim = 64)
    self.bottleneck_3 = ResnetBlock(in_channels = bottle_dim, out_channels = bottle_dim, time_embedded_dim = time_dim, groups = groupnorm_groups)

    #----------------UP PATH----------------------

    self.ups = nn.ModuleList([UpStage(in_channels = stage.in_channels,
                                  out_channels = stage.out_channels,
                                  time_emb_dim = time_dim,
                                  groups = groupnorm_groups,
                                  attn_type = stage.attn,
                                  context_dim = context_dim,
                                  heads = 8,
                                  head_dim = 64,
                                  upsample = stage.updownsample) for stage in up_stages])

    #----------------OUTPUT PATH----------------------
    self.out_dim = output_channels if output_channels is not None else channels
    self.out_resblock = ResnetBlock(in_channels = initial_dim * 2, out_channels = initial_dim, time_embedded_dim=time_dim, groups = groupnorm_groups)
    self.out_conv = nn.Conv2d(initial_dim, self.out_dim, 1)

  def forward(self, x, time, context, x_self_cond = None):
    if self.self_condition:
      if x_self_cond is None:
        x_self_cond = torch.zeros_like(x) # If we don't have a x_self_cond tensor, create one with 0s like x
      x = torch.cat((x_self_cond, x), dim = 1) # Concatenate two tensors along its channels (remember that we did channels*2)

    # Channel projection
    x = self.initial_conv(x)
    r = x.clone()

    # Time embedding
    t = self.time_mlp(time) #embed timestep with sinposembedding and then enrich it with nn.Linear with input_dim*4

    # Down path
    h = [] # skip connections list
    for down in self.downs:
      x, skip = down(x, t, context)
      h.append(skip)

    # Bottleneck
    x = self.bottleneck_1(x, t)
    x = self.bottleneck_attention_1(x, context)
    x = self.bottleneck_2(x, t)

    # Doubled bottleneck depth
    x = self.bottleneck_attention_2(x, context)
    x = self.bottleneck_3(x, t)

    # Upward path
    for up in self.ups:
      x = up(x, t, h.pop(), context) # h is a list of lists, so h = [[skip1, skip2], [skip3, skip4],...]

    # Output layer
    x = torch.cat((x, r), dim = 1)
    x = self.out_resblock(x, t)
    return self.out_conv(x)
