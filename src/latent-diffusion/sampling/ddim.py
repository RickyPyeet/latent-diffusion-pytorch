import torch
from tqdm.auto import tqdm

from src.ddpm.diffusion.process import extract
from src.ddpm.diffusion.schedules import DiffusionSchedule
from src.ddpm.diffusion.parameterization import compute_epsilon_from_x0, compute_x0_from_epsilon, compute_x0_epsilon_from_v, compute_sigma_t

# Compute the sampling step x_t-1
def ddim_step(model, 
              x_t, 
              t, 
              t_prev, 
              c, 
              alpha_bar_t, 
              alpha_bar_t_prev, 
              pred_type = 'epsilon', 
              guidance_scale = 2.5, 
              eta = 0.8):
  # Get device
  device = next(model.parameters()).device

  # Calculate random noise if sampling is stochastic or semi-deterministic
  if eta > 0.0:
    z = torch.randn_like(x_t, device = device)
  else:
    z = 0.0

  # Prediciton
  with torch.inference_mode():
    pred_cond = model(x = x_t, time = t, c = c)
    pred_uncond = model(x = x_t, time = t, c = torch.full((x_t.shape[0], ), -1, device = device, dtype = torch.long))

  # Classifier free guidance prediction
  general_pred = pred_uncond + guidance_scale*(pred_cond - pred_uncond)

  # Compute x0 and epsilon based on pred_type
  if pred_type == 'epsilon':
    epsilon = general_pred
    x_0 = compute_x0_from_epsilon(x_t, epsilon, alpha_bar_t)

  elif pred_type == 'x_0':
    x_0 = general_pred
    epsilon = compute_epsilon_from_x0(x_t, x_0, alpha_bar_t)

  elif pred_type == 'v':
    v = general_pred
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


def sample_ddim(model, timesteps, sampling_timesteps, img_shape, c, pred_type = 'epsilon', save_steps = False, guidance_scale = 2.5, eta = 0.8):
  device = next(model.parameters()).device
  model.eval()

  # Create noise schedule and extract alpha_bar
  schedule = DiffusionSchedule(timesteps = timesteps, schedule = 'cosine', device = device)
  _, _, alpha_bar = schedule()

  # Initialize starting noised img
  x = torch.randn(img_shape).to(device)
  img_steps = []

  # Extract batch size
  if len(img_shape) == 4:
    batch_size = img_shape[0]
  else:
    batch_size = 1

  # Check if c is tensor, if not make it so
  if not isinstance(c, torch.Tensor):
    c = torch.tensor([c], dtype = torch.long, device = device)
  else:
    c = c.to(device)

  # Check if labels batch size matches img_shape batch size
  assert c.shape[0] == batch_size, f"Expected {batch_size} labels, but got {c.shape[0]}"

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
                  c = c,
                  alpha_bar_t = alpha_bar_t,
                  alpha_bar_t_prev = alpha_bar_t_prev,
                  pred_type = pred_type,
                  guidance_scale = guidance_scale,
                  eta = eta)

    # Save steps
    if save_steps == True and ((step%50 == 0) or (step == 0)):
      img_steps.append(x.detach().cpu())

  # Clamp tensor
  x = x.clamp(-1, 1)

  return x, img_steps
