import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from src.latent_diffusion.diffusion.process import extract
from src.latent_diffusion.diffusion.schedules import DiffusionSchedule

def p_sample(model,
            x,
            t,
            betas,
            alphas,
            alpha_bars,
            t_index,
            cond_embedding,
            uncond_embedding,
            pred_type = 'epsilon',
            guidance_scale = 2.5):
  '''Generate images, starting from pure noise, and slowly removing it'''

  # Extend alpha_bars to shape (batch, 1,1,1)
  beta_t = extract(betas, t, x.shape)
  alpha_t = extract(alphas, t, x.shape)
  alpha_bar_t = extract(alpha_bars, t, x.shape)
  alpha_bar_previous = F.pad(alpha_bars[:-1], (1,0), value = 1.0) # alpha at time step t-1
  alpha_bar_previous_t = extract(alpha_bar_previous, t, x.shape)

  # Compute the posterior variance
  posterior_variance = betas * (1. - alpha_bar_previous) / (1 - alpha_bars)

  # Make prediction
  pred_cond = model(x, t, context = cond_embedding)
  pred_uncond = model(x, t, context = uncond_embedding)
  
  pred_guided = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

  # Compute x_0 based on the type of prediction we make
  if pred_type == 'epsilon':
    eps = pred_guided
    x_0 = (x - torch.sqrt(1-alpha_bar_t)*eps) / (torch.sqrt(alpha_bar_t))
  elif pred_type == 'x_0':
    x_0 = pred_guided
    eps = (x - torch.sqrt(alpha_bar_t)*x_0) / torch.sqrt(1-alpha_bar_t)
  elif pred_type == 'v':
    v = pred_guided
    eps = torch.sqrt(alpha_bar_t)*v + torch.sqrt(1-alpha_bar_t)*x
    x_0 = torch.sqrt(alpha_bar_t)*x - torch.sqrt(1-alpha_bar_t)*v
  else:
    raise ValueError(f"Unknown pred type: {pred_type}")

  # Compute mean
  coeff1 = beta_t * torch.sqrt(alpha_bar_previous_t)/(1-alpha_bar_t)
  coeff2 = torch.sqrt(alpha_t) * (1-alpha_bar_previous_t) / (1-alpha_bar_t)
  mean = x_0*coeff1 + x*coeff2

  # Check if noise needs to be added or not
  if t_index == 0: # Final denoised image
    return mean
  else:
    posterior_variance_t = extract(posterior_variance, t, x.shape)
    noise = torch.randn_like(x) # z tilde (0, I)
    return mean + torch.sqrt(posterior_variance_t)*noise

@torch.inference_mode()
def sample_ddpm(model,
                timesteps,
                img_shape,
                prompt,
                clip,
                pred_type = 'epsilon',
                save_steps = False,
                guidance_scale = 2.5,
                vae = None):
  device = next(model.parameters()).device
  model.eval()

  # If using CLIP, move it to device
  if clip is not None:
    clip = clip.to(device)

  # If using VAE move it to device
  if vae is not None:
    vae = vae.to(device)


  schedule = DiffusionSchedule(timesteps = timesteps, schedule = 'cosine', device = device)
  betas, alphas, alpha_bars = schedule()

  x = torch.randn(img_shape, device = device)
  latent_steps = []

  if len(img_shape) == 3:
    batch = 1
    x = x.unsqueeze(0)
  else:
    batch = img_shape[0]
  
  # Adjust cond prompt if it's just a string or raise error if len of list does not match batch size
  if isinstance(prompt, str):
    prompt = [prompt] * batch
  elif len(prompt) != batch:
    raise ValueError(f"Expected {batch} prompts but received {len(prompt)} instead")

  # Define uncond_prompt
  uncond_prompt = [""] * batch

  cond_embedding = clip.encode(prompt)
  uncond_embedding = clip.encode(uncond_prompt)

  for t in tqdm(reversed(range(timesteps)), desc = f"Generating sample in a total of {timesteps} timesteps", total = timesteps): # Start from timesteps-1 and reach 0
    x = p_sample(model = model,
                 x = x,
                 t = torch.full((batch, ), t, device = device, dtype = torch.long),
                 betas = betas,
                 alphas = alphas,
                 alpha_bars = alpha_bars,
                 t_index = t,
                 cond_embedding = cond_embedding,
                 uncond_embedding = uncond_embedding,
                 pred_type = pred_type,
                 guidance_scale = guidance_scale)

    if save_steps == True and ((t%100 == 0) or t == timesteps-1):
      latent_steps.append(x.detach().cpu())

  if vae is not None:
    x = vae.decode(x)
    x = x.clamp(-1, 1)

  return x, latent_steps
