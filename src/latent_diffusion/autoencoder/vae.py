import torch
from torch import nn
from diffusers import AutoencoderKL

class FrozenVAE(nn.Module):
  def __init__(self, vae_name: str):
    """
    Instantiate a pretrained VAE from vae_name, freezes it and encodes images or decodes latents applying the vae's scaling factor
    args:
      vae_name = name of pretrained vae - "stabilityai/sd-vae-ft-mse"
    """
    super().__init__()
    self.vae = AutoencoderKL.from_pretrained(vae_name)
    for param in self.vae.parameters():
      if param.requires_grad:
        param.requires_grad = False
    # Alternative is `self.vae.requires_grad_(False)`
    
    self.vae.eval()

  @torch.inference_mode()
  def encode(self, image: torch.Tensor):
    """
    Encodes an image tensor into a latent representation automatically scaled by the VAE's scaling factor
    args:
      image (torch.Tensor)
    out:
      latents = latent scaled representation
    """
    posterior = self.vae.encode(image).latent_dist
    latents = posterior.sample()
    latents = latents * self.vae.config.scaling_factor
    return latents

  @torch.inference_mode()
  def decode(self, latent: torch.Tensor):
    """
    Decodes a latent into an image tensor automatically de-scaled by the VAE's scaling factor
    args:
      latents = latent scaled representation
    out:
      image (torch.Tensor)
    """
    latent = latent / self.vae.config.scaling_factor
    image = self.vae.decode(latent).sample
    return image
  