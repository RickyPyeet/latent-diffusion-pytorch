import math
import torch
from torch import nn

from src.ddpm.models.init import init_conv

### Attention Helpers ###

# Function to reshape QKV values
def _qkv_reshape(qkv, heads = 4):
  qkv = list(qkv)
  for i, t in enumerate(qkv):
    b, c, h, w = t.shape
    head_dim = c // heads
    t = t.reshape(b, heads, head_dim, h*w) # (b, heads, head_dim, N)
    t = t.transpose(-1, -2) # (b, heads, N, head_dim)
    qkv[i] = t

  q, k, v = qkv
  return q, k, v

# Compute the similarity matrix
def _compute_similarity(q, k):
  similarity = torch.matmul(q, k.transpose(-1,-2)) # (batch, heads, N, head_dim) @ (batch, heads, head_dim, N)
  similarity = similarity - similarity.amax(dim = -1, keepdim = True) # Subtract max for numerical stability
  return similarity

# Compute the context matrix
def _compute_context(k, v):
  k = k.transpose(-1,-2) # (batch, head, head_dim, N) while v = (b, head, N, head_dim)
  context = torch.matmul(k, v)
  return context

### Attention Blocks ###

class SelfAttention(nn.Module):
  def __init__(self, in_channels, heads = 4, head_dim = 32):
    super().__init__()
    self.scale = 1/math.sqrt(head_dim)
    self.heads = heads
    self.hidden_channels = heads * head_dim
    self.to_qkv = nn.Conv2d(in_channels, self.hidden_channels * 3, kernel_size = 1, bias = False) # hidden_channels * 3 because we need Q, K, V which are 3 different values of size `hidden_channels`
    self.to_out = nn.Conv2d(self.hidden_channels, in_channels, kernel_size = 1)

    init_conv(self.to_qkv)
    init_conv(self.to_out)

  def forward(self, x):
    b, c, h, w = x.shape
    qkv = self.to_qkv(x) # (batch, hidden_channels*3, w, h)
    qkv = qkv.chunk(3, dim = 1) # Split QKV in Q, K, V along the 1 dimension
    q, k, v = _qkv_reshape(qkv, heads = self.heads) # (batch, heads, N, head_dim)
    q = q * self.scale # scale q to have more stable gradients
    similarity = _compute_similarity(q, k)
    attention = torch.softmax(similarity, dim = -1)
    attention = torch.matmul(attention, v) # Shape is (b, heads, N, head_dim)
    attention = attention.transpose(-1, -2) # needed to return to original shape before splitting up (batch, heads, head_dim, height*width)
    attention = attention.reshape((b, self.hidden_channels, h, w)) # Shape (b, heads*heads_dim, height, width)
    out = self.to_out(attention)
    return out

class LinearAttention(nn.Module):
  def __init__(self, in_channels, heads = 4, head_dim = 32):
    super().__init__()
    self.heads = heads
    self.in_channels = in_channels
    self.head_dim = head_dim

    self.hidden_channels = self.heads * self.head_dim
    self.scale = 1/math.sqrt(self.head_dim)

    self.to_qkv = nn.Conv2d(in_channels = in_channels, out_channels = 3*self.hidden_channels, kernel_size = 1, bias = False)

    self.to_out = nn.Sequential(nn.Conv2d(self.hidden_channels, in_channels, kernel_size = 1),
                                nn.GroupNorm(1, in_channels))
    init_conv(self.to_qkv)
    init_conv(self.to_out[0])

  def forward(self, x):
    b, c, h, w = x.shape
    qkv = self.to_qkv(x).chunk(3, dim = 1)
    q, k, v = _qkv_reshape(qkv, heads = self.heads) # (batch, head, N, head_dim)

    q = torch.softmax(q, dim = -1) # Computes softmax over the head_dim to normalize it
    k = torch.softmax(k, dim = -2) # Computes softmax over the h*w to normalize it

    q = q * self.scale # Scale q for gradients stability

    context = _compute_context(k, v)
    out = torch.matmul(q, context)
    out = out.transpose(-1, -2)
    out = out.reshape((b, self.hidden_channels, h, w))
    out = self.to_out(out)

    return out

class CrossAttention(nn.Module):
  def __init__(self, in_channels, context_dim, heads = 4, head_dim = 32):
    super().__init__()
    self.heads = heads
    self.head_dim = head_dim
    self.hidden_channels = heads * head_dim
    self.scale = 1/math.sqrt(head_dim)

    self.query = nn.Conv2d(in_channels = in_channels,
                           out_channels = self.hidden_channels,
                           kernel_size = 1,
                           bias = False)

    self.key = nn.Linear(in_features = context_dim,
                         out_features = self.hidden_channels)

    self.value = nn.Linear(in_features = context_dim,
                           out_features = self.hidden_channels)

    self.to_out = nn.Sequential(
        nn.Conv2d(in_channels = self.hidden_channels,
                  out_channels = in_channels,
                  kernel_size = 1),
        nn.GroupNorm(num_groups = 1, num_channels = in_channels))
    
    init_conv(self.query)
    init_conv(self.to_out[0])

    # Create QKV
  def forward(self, x, cond):
    # Add dimension 1 if class conditioning
    if len(cond.shape) == 2:
      cond = cond.unsqueeze(1) # Add a token dimension in position 1 - (batch, embedding_dim) -> (batch, 1, embedding_dim)

    # Extract dimensions
    B, C, H, W = x.shape
    _, T, _ = cond.shape

    # Q from image
    Q = self.query(x) # (batch, heads*head_dim, H, W)
    #print(f"Q shape: {Q.shape}")

    # K and V from conditioning
    K = self.key(cond) # (batch, T, heads*head_dim)
    V = self.value(cond) # (batch, T, heads*head_dim)
    #print(f"K and V shape: {K.shape}")

    # Reshape Q
    Q = Q.reshape(B, self.heads, self.head_dim, H*W) # (batch, heads, head_dim, H*W)
    Q = Q.transpose(-1,-2) #(batch, heads, N, head_dim)
    Q = Q * self.scale

    # Reshape K
    K = K.reshape(B, T, self.heads, self.head_dim) # (batch, T, heads, head_dim)
    K = K.transpose(1, 2) # (batch, heads, T, head_dim)

    # Reshape V
    V = V.reshape(B, T, self.heads, self.head_dim) # (batch, T, heads, head_dim)
    V = V.transpose(1, 2) # (batch, heads, T, head_dim)

    # Compute similarity matrix
    similarity = _compute_similarity(q = Q, k = K) # (batch, heads, N, T)

    # Softmax
    similarity = torch.softmax(similarity, dim = -1)

    # Compute attention
    similarity = torch.matmul(similarity, V) # (batch, heads, N, head_dim)
    similarity = similarity.transpose(-1, -2) # (batch, heads, head_dim, N)
    similarity = similarity.reshape(B, self.hidden_channels, H, W)

    # Output
    out = self.to_out(similarity) # (batch, C, H, W)

    return out
