import torch
from torch import nn
# Helper function to initialize weights and bias in a new conv block
def init_conv(m, mode = 'fan_in', nonlinearity = 'relu'):
  if isinstance(m, nn.Conv2d):
    nn.init.kaiming_normal_(m.weight, mode = mode, nonlinearity = nonlinearity)
    if m.bias is not None:
      nn.init.zeros_(m.bias)
