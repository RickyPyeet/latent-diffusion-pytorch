import math
import torch

from torch import nn

class SinPositionalEmbedding(nn.Module):
  """
  Sin step embedding used by original Transformer paper and DDPM
  """
  def __init__(self, embedding_dim):
    super().__init__()
    self.embedding_dim = embedding_dim

  def forward(self, timesteps):
    device = timesteps.device
    half_dim = self.embedding_dim // 2

    embeddings = math.log(10000) / (half_dim - 1)
    embeddings = torch.exp(torch.arange(half_dim, device = device) * -embeddings)
    embeddings = timesteps[:, None] * embeddings[None, :]
    embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim = -1)
    return embeddings
