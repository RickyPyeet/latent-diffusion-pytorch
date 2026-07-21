import random
import torch
from torch import nn
from tqdm.auto import tqdm

from src.latent_diffusion.diffusion.process import batched_diffusion_kernel, extract
from src.latent_diffusion.diffusion.schedules import DiffusionSchedule
from src.latent_diffusion.sampling.inference import generate_and_plot
from src.latent_diffusion.training.ema import EMA
from src.latent_diffusion.training.objective import get_train_target
from src.latent_diffusion.utils.checkpoint import load_checkpoint, save_checkpoint
from src.latent_diffusion.training.snr import min_snr_loss


def create_optimizer(model, optim_type, lr):
  optim_list = ['adam', 'adamw']

  if optim_type not in optim_list:
    raise ValueError(f"{optim_type} is not a valid optimizer, pick {optim_list}")
  
  if optim_type == 'adam':
    optimizer = torch.optim.Adam(params = model.parameters(),
                                 lr = lr)
  elif optim_type == 'adamw':
    optimizer = torch.optim.AdamW(params = model.parameters(),
                                  lr = lr,
                                  betas = (0.9, 0.999),
                                  eps = 1e-8,
                                  weight_decay = 0.01)
  return optimizer

def trainer(model: nn.Module,
            cached_vae_dataloader,
            epochs,
            device,
            vae,
            clip,
            pred_type: str = 'v',
            lr: float = 1e-4,
            class_free_dropout: float = 0.2,
            guidance_scale: float = 3.5,
            eta: float = 0.8,
            ema_decay: float = 0.999,
            timesteps: int = 1000,
            schedule_type: str = 'cosine',
            optim: str = "adamw",
            save_dir: str = None,
            save_every: int = 200,
            resume_from: str = None,
            sample_every: int | None = 50,
            example_prompts: list[str] | str | None = None,
            sampler: str = 'ddpm',
            sample_timesteps: int = 100,
            use_snr: bool = True,
            snr_gamma: int = 5,
            seed: int = 42):

  valid_pred_type =['epsilon', 'x_0', 'v']

  if pred_type not in valid_pred_type:
    raise ValueError(f"Pred_type must be one of {valid_pred_type}, got {pred_type} instead")

  model = model.to(device)
  clip = clip.to(device)
  vae = vae.to(device)

  # Make example prompts if not passed
  if example_prompts is None:
    example_prompts = ['a motorcycle',
                    'a dog',
                    'a pizza',
                    'a car',
                    'an airplane']
  elif isinstance(example_prompts, str):
    example_prompts = [example_prompts]

  # Create optimizer
  optimizer = create_optimizer(model = model,
                               optim_type = optim,
                               lr = lr)

  # Create loss_fn
  if use_snr:
    loss_fn = torch.nn.MSELoss(reduction = 'none')
  else:
    loss_fn = torch.nn.MSELoss()

  # EMA
  ema = EMA(model = model, decay = ema_decay)

  # Noise Schedule
  schedule = DiffusionSchedule(timesteps = timesteps, schedule = schedule_type, device = device)
  _, _, alpha_bars = schedule()

  # Define training variables
  starting_epoch = 0
  loss_hist = []
  checkpoint = {}

  # Load checkpoint if exists
  if resume_from is not None:
    checkpoint = load_checkpoint(checkpoint_path = resume_from,
                                 model = model,
                                 optimizer = optimizer,
                                 ema = ema,
                                 device = device)

    starting_epoch = checkpoint['epoch']+1
    pred_type = checkpoint['pred_type']
    loss_hist = checkpoint['loss_hist']

    class_free_dropout = checkpoint['class_free_dropout']
    # guidance_scale = checkpoint['guidance_scale']

  for epoch in tqdm(range(starting_epoch, epochs)):
    print(f"Epoch {epoch+1}/{epochs}\n------------------")
    model.train()
    train_loss = 0.0

    for i, batch in enumerate(cached_vae_dataloader):
      latents = batch['latents']
      prompts = batch['prompts']

      latents = latents.to(device)

      batch_size = latents.shape[0]

      # Extract random noise steps between 0-1000
      t = torch.randint(0, timesteps, (batch_size, ), device = device, dtype = torch.long)

      ### FORWARD DIFFUSION
      noisy_latents, noise = batched_diffusion_kernel(x_not = latents,
                                                     t = t,
                                                     alpha_bars = alpha_bars)

      ### Drop Mask for CFG
      drop_mask = (torch.rand(batch_size, device = device) < class_free_dropout).tolist()
      c_input = [random.choice(p) if isinstance(p, (list, tuple)) else p for p in prompts]
      for i in range(batch_size):
        if drop_mask[i]:
          c_input[i] = "" # null token is identified with ''

      ### CLIP EMBEDDING
      text_embeddings = clip.encode(c_input)

      # Make a prediction
      pred_target = model(noisy_latents, time = t, context = text_embeddings)

      # Extract target based on pred_type
      target = get_train_target(x_0 = latents,
                                noise = noise,
                                alpha_bar = alpha_bars,
                                timestep = t,
                                pred_type = pred_type)

      # Compute loss and optimize
      loss = loss_fn(pred_target, target)
      if use_snr:
        alpha_bars_batched = extract(alpha_bars, t, latents.shape)
        loss = min_snr_loss(loss, alpha_bars_batched, pred_type = pred_type, gamma = snr_gamma)

      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 10.0)
      optimizer.step()
      ema.update()

      train_loss += loss.item()

    train_loss /= len(cached_vae_dataloader)
    loss_hist.append(train_loss)

    print(f"Training Loss: {train_loss:.5f}\n")

    # Save checkpoints
    if save_dir is not None and (((epoch+1) % save_every == 0) or ((epoch+1) == epochs)):
      checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'epoch': epoch,
        'pred_type': pred_type,
        'loss_hist': loss_hist,
        'class_free_dropout': class_free_dropout
      }
      save_checkpoint(name = f"checkpoint_epoch_{epoch+1}_{lr}_lr_{pred_type}_pred_type.pt",
                      checkpoint = checkpoint,
                      checkpoint_path = save_dir)

    # Generate samples
    if sample_every is not None:
      if (epoch + 1) % sample_every == 0:
        ema.apply_shadow()
        model.eval()

        with torch.inference_mode():
          generate_and_plot(model = model,
                            clip = clip,
                            vae = vae,
                            prompt = example_prompts,
                            img_shape = (len(example_prompts), 4, 32, 32),
                            sampler = sampler,
                            timesteps = timesteps,
                            sampling_timesteps = sample_timesteps,
                            pred_type = pred_type,
                            guidance_scale = guidance_scale,
                            eta = eta,
                            title = f"Epoch: {epoch+1}",
                            seed = seed)

        ema.restore()

  return loss_hist, checkpoint