import torch
from torch import nn
from tqdm.auto import tqdm

from src.ddpm.diffusion.process import batched_diffusion_kernel
from src.ddpm.diffusion.schedules import DiffusionSchedule
from src.ddpm.sampling.inference import generate_and_plot
from src.ddpm.training.ema import EMA
from src.ddpm.training.objective import get_train_target
from src.ddpm.utils.checkpoint import load_checkpoint, save_checkpoint

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
            padding: str = 'max_length',
            max_length: int = 77,
            truncation: bool = True,
            return_tensors: str = 'pt',
            use_snr: bool = True,
            snr_gamma: int = 5,
            seed: int = 42):

  valid_pred_type =['epsilon', 'x_0', 'v']

  if pred_type not in valid_pred_type:
    raise ValueError(f"Pred_type must be one of {valid_pred_type}, got {pred_type} instead")

  model = model.to(device)

  # Create optimizer
  optimizer = create_optimizer(model = model,
                               optim_type = optim,
                               lr = lr)

  # Create loss_fn
  loss_fn = torch.nn.MSELoss()

  # EMA
  ema = EMA(model = model, decay = ema_decay)

  # Noise Schedule
  schedule = DiffusionSchedule(timesteps = timesteps, schedule = schedule_type, device = device)
  betas, alphas, alpha_bar = schedule()

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
                                                     alpha_bars = alpha_bar)

      ### Drop Mask for CFG
      drop_mask = torch.rand(batch_size, device = device) < class_free_dropout
      c_input = [random.choice(p) if isinstance(p, (list, tuple)) for p in prompts]
      for i in range(batch_size):
        if drop_mask[i]:
          c_input[i] = "" # null token is identified with ''

      ### CLIP EMBEDDING PIPELINE
      # 1. Tokenize c_input after applying CFG
      tokens = tokenizer(c_input,
                         padding = padding,
                         max_length = max_length,
                         truncation = truncation,
                         return_tensors = return_tensors)

      # Move them to device
      tokens = {k: v.to(device) for k,v in tokens.items()}

      # 2. Embed text from tokens
      with torch.no_grad():
        text_embeddings = text_encoder(**tokens).last_hidden_state

      # Make a prediction
      pred_target = model(latents, time = t, context = text_embeddings)

      # Extract target based on pred_type
      target = get_train_target(x_0 = images,
                                noise = noise,
                                alpha_bar = alpha_bar,
                                timestep = t,
                                pred_type = pred_type)

      # Compute loss and optimize
      loss = loss_fn(pred_target, target)
      if use_snr:
        alpha_bars_batched = extract(alpha_bars, t, latents.shape)
        loss = min_snr_loss(loss, alpha_bars_batched, gamma = snr_gamma)

      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 10.0)
      optimizer.step()
      ema.update()

      train_loss += loss.item()

    train_loss /= len(cached_vae_dataloader)
    loss_hist.append(train_loss)

    print(f"Training Loss: {train_loss:.5f}\n")

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'epoch': epoch,
        'pred_type': pred_type,
        'loss_hist': loss_hist,
        'class_free_dropout': class_free_dropout
    }
    # Save checkpoints
    if save_dir is not None and (((epoch+1) % save_every == 0) or ((epoch+1) == epochs)):
      save_checkpoint(name = f"checkpoint_epoch_{epoch+1}_{lr}_lr_{pred_type}_pred_type.pt",
                      checkpoint = checkpoint,
                      checkpoint_path = save_dir)

    # Generate samples
    if sample_every is not None:
      if (epoch + 1) % sample_every == 0:
        ema.apply_shadow()
        model.eval()

        with torch.inference_mode():
          if example_prompts is None:
            example_prompts = ['a motorcycle',
                           'a dog',
                           'a pizza',
                           'a car',
                           'an airplane']
            get_sample_img_with_prompt(model = model,
                              img_shape = (1,4,32,32),
                              timesteps = timesteps,
                              pred_type = pred_type,
                              prompt = random.choice(example_prompts),
                              sampler = sampler,
                              guidance_scale = guidance_scale,
                              plot_img = True,
                              save_steps = False,
                              epoch = epoch,
                              seed = 42,
                              tokenizer = tokenizer,
                              text_encoder = text_encoder,
                              vae = vae)
#          generate_and_plot(model = model,
#                            c = sample_labels,
#                            img_shape = (len(sample_labels), 3, 32, 32),
#                            sampler = sampler,
#                            timesteps = timesteps,
#                            sampling_timesteps = sample_timesteps,
#                            pred_type = pred_type,
#                            guidance_scale = guidance_scale,
#                            eta = eta,
#                            title = f"Epoch: {epoch+1}",
#                            seed = seed)

        ema.restore()

  return loss_hist, checkpoint