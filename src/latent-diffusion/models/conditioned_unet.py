import torch
from torch import nn

from src.ddpm.models.unet import DenoisingUNet

class ClassConditionedUNet(nn.Module):
  def __init__(self, num_classes = 10, input_dim = 64, channels = 3, groupnorm_groups = 8, dimension_multiplier = (1,2,4,8)):
    super().__init__()
    self.num_classes = num_classes

    self.embedding_dim = input_dim * 4
    self.class_embedding = nn.Embedding(self.num_classes, self.embedding_dim)
    self.class_mlp = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim),
                                   nn.GELU(),
                                   nn.Linear(self.embedding_dim, self.embedding_dim))

    self.null_emb_token = nn.Parameter(torch.randn(self.embedding_dim)) # To handle the null_token - no class - learnable parameter

    self.unet = DenoisingUNet(input_dim = input_dim,
                              channels = channels,
                              output_channels = channels,
                              dimension_multiplier = dimension_multiplier,
                              groupnorm_groups = groupnorm_groups)

  def forward(self, x, time, c):
    B = c.shape[0] # batch
    cloned_labels = c.clone()

    # Crate mask and apply it to cloned vector - set null labels to 0 to pass it through embedding
    null_mask = c == -1
    cloned_labels[null_mask] = 0

    # Embed the classes
    class_emb = self.class_mlp(self.class_embedding(cloned_labels))

    # Set the null_emb_token to the null position
    null_emb = self.null_emb_token.unsqueeze(0).expand(B, -1) # Change it to shape (B, embedding_dim)
    class_emb = torch.where(null_mask[:, None], null_emb, class_emb)

    return self.unet(x = x, time = time, cond_emb = class_emb)
