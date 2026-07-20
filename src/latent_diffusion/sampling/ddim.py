import torch
from tqdm.auto import tqdm

from src.latent_diffusion.diffusion.process import extract
from src.latent_diffusion.diffusion.schedules import DiffusionSchedule
from src.latent_diffusion.diffusion.parameterization import compute_epsilon_from_x0, compute_x0_from_epsilon, compute_x0_epsilon_from_v, compute_sigma_t

# Compute the sampling step x_t-1
def ddim_step(model, 
              x_t, 
              t, 
              t_prev, 
              cond_embedding,
              uncond_embedding, 
              alpha_bar_t, 
              alpha_bar_t_prev, 
              pred_type = 'epsilon', 
              guidance_scale = 2.5, 
              eta = 0.8):

  # Calculate random noise if sampling is stochastic or semi-deterministic
  if eta > 0.0:
    z = torch.randn_like(x_t)
  else:
    z = 0.0

  # Prediciton
  pred_cond = model(x = x_t, time = t, context = cond_embedding)
  pred_uncond = model(x = x_t, time = t, context = uncond_embedding)

  # Classifier free guidance prediction
  pred_guided = pred_uncond + guidance_scale*(pred_cond - pred_uncond)

  # Compute x0 and epsilon based on pred_type
  if pred_type == 'epsilon':
    epsilon = pred_guided
    x_0 = compute_x0_from_epsilon(x_t, epsilon, alpha_bar_t)

  elif pred_type == 'x_0':
    x_0 = pred_guided
    epsilon = compute_epsilon_from_x0(x_t, x_0, alpha_bar_t)

  elif pred_type == 'v':
    v = pred_guided
    x_0, epsilon = compute_x0_epsilon_from_v(x_t, v, alpha_bar_t)

  else:
    raise ValueError(f"Pred_type {pred_type} unsupported")

  # Sample DDIM
  if t_prev >= 0:
    sigma_t = compute_sigma_t(alpha_bar_t, alpha_bar_t_prev, eta)
    x = x_0 * torch.sqrt(alpha_bar_t_prev) + epsilon * torch.sqrt(1 - alpha_bar_t_prev - sigma_t**2) + sigma_t * z
  else:
    x = x_0
  return x

@torch.inference_mode()
def sample_ddim(model,
                timesteps,
                sampling_timesteps,
                img_shape,
                prompt,
                clip,
                pred_type = 'epsilon',
                save_steps = False,
                guidance_scale = 2.5,
                eta = 0.8,
                vae = None):

  device = next(model.parameters()).device
  model.eval()

  # Instantiate CLIP encoder
  if clip is not None:
    clip = clip.to(device)

  # If using VAE, instantiate VAE decoder
  if vae is not None:
    vae = vae.to(device)

  # Create noise schedule and extract alpha_bar
  schedule = DiffusionSchedule(timesteps = timesteps, schedule = 'cosine', device = device)
  _, _, alpha_bar = schedule()

  # Initialize starting noised img
  x = torch.randn(img_shape).to(device)
  latent_steps = []

  # Extract batch size
  if len(img_shape) == 4:
    batch_size = img_shape[0]
  else:
    batch_size = 1
    x = x.unsqueeze(0)

  # Adjust cond prompt if it's just a string or raise error if len of list does not match batch size
  if isinstance(prompt, str):
    prompt = [prompt] * batch_size
  elif len(prompt) != batch_size:
    raise ValueError(f"Expected {batch} prompts but received {len(prompt)} instead")

  # Define uncond_prompt
  uncond_prompt = [""] * batch
  
  # Check if c is tensor, if not make it so
  cond_embedding = clip.encode(prompt)
  uncond_embedding = clip.encode(uncond_prompt)

  # Define time pairs used to denoise (999, 949), (949, 799), ..., (49, 0), (0, -1)
  times = torch.linspace(0, timesteps - 1, sampling_timesteps, dtype=torch.long).tolist()
  times = [int(t) for t in reversed(times)]
  next_times = times[1:] + [-1]                    
  time_pairs = list(zip(times, next_times))

  # Start loop between
  for step, step_prev in tqdm(time_pairs, 
                              desc = f'DDIM Reverse Process for {sampling_timesteps} timesteps', 
                              total = sampling_timesteps):
    # Create a batched size t for model prediction
    t = torch.full((batch_size, ), step, device = device, dtype = torch.long)

    # Check if step_prev != -1
    if step_prev >= 0:
      t_prev = torch.full((batch_size, ), step_prev, device = device, dtype = torch.long)
    else:
      t_prev = None

    # Extract alpha, alpha bar, alpha bar previous at timestep t
    alpha_bar_t = extract(alpha_bar, t, x.shape)
    alpha_bar_t_prev = extract(alpha_bar, t_prev, x.shape) if t_prev is not None else None

    # Predict denoised image
    x = ddim_step(model = model,
                  x_t = x,
                  t = t,
                  t_prev = step_prev,
                  cond_embedding = cond_embedding,
                  uncond_embedding = uncond_embedding,
                  alpha_bar_t = alpha_bar_t,
                  alpha_bar_t_prev = alpha_bar_t_prev,
                  pred_type = pred_type,
                  guidance_scale = guidance_scale,
                  eta = eta)

    # Save steps
    if save_steps == True and ((step%50 == 0) or (step == 0)):
      latent_steps.append(x.detach().cpu())

  if vae is not None:
    x = vae.decode(x)
    # Clamp tensor
    x = x.clamp(-1, 1)

  return x, latent_steps
