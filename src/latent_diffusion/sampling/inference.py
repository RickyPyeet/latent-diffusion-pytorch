import torch
from src.latent_diffusion.utils.image import plot_image_grid
from src.latent_diffusion.sampling.ddpm import sample_ddpm
from src.latent_diffusion.sampling.ddim import sample_ddim

@torch.inference_mode()
def generate_and_plot(model,
                      clip,
                      vae,
                      prompt,
                      img_shape,
                      sampler: str = 'ddim',
                      timesteps: int = 1000,
                      sampling_timesteps: int = 100,
                      pred_type: str = 'v',
                      guidance_scale: float = 4.0,
                      eta: float = 0.8,
                      save_steps: bool = False,
                      ncols: int = 4,
                      title: str | None = None,
                      show_img: bool = True,
                      save_img: bool = False,
                      save_path: str | None = None,
                      seed: int | None = None):
  """
  Generate samples using DDPM or DDIM and plot them as a grid
  """
  if seed is not None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
   
  if sampler == 'ddpm':
    samples, steps = sample_ddpm(model = model,
                                 timesteps = timesteps,
                                 img_shape = img_shape,
                                 prompt = prompt,
                                 clip = clip,
                                 vae = vae,
                                 pred_type = pred_type,
                                 save_steps = save_steps,
                                 guidance_scale = guidance_scale)
  elif sampler == 'ddim':
    samples, steps = sample_ddim(model = model,
                                 timesteps = timesteps,
                                 sampling_timesteps = sampling_timesteps,
                                 img_shape = img_shape,
                                 prompt = prompt,
                                 clip = clip,
                                 vae = vae,
                                 pred_type = pred_type,
                                 save_steps = save_steps,
                                 guidance_scale = guidance_scale,
                                 eta = eta)
  else:
    raise ValueError(f"Sampler must either be ddpm or ddim, got {sampler}")

  plot_image_grid(samples = samples,
                  ncols = ncols,
                  title = title,
                  show_img = show_img,
                  save_img = save_img,
                  save_path = save_path)
  
  return samples, steps
