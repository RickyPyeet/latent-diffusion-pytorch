import torch
import argparse
from src.latent_diffusion.utils.config import load_config
from src.latent_diffusion.utils.seed import set_seed
from src.latent_diffusion.models.unet import UNet
from src.latent_diffusion.training.ema import EMA
from src.latent_diffusion.sampling.inference import generate_and_plot
from src.latent_diffusion.autoencoder.vae import FrozenVAE
from src.latent_diffusion.conditioning.clip import FrozenCLIP

def parse_args():
  parser = argparse.ArgumentParser()

  # Arguments
  parser.add_argument('--checkpoint', type = str, required = True, help = 'Model checkpoint path')
  parser.add_argument('--config', type = str, default = 'configs/coco.yaml', help = 'Model config path')
  parser.add_argument('--sampler', type = str, default = 'ddim', choices = ['ddpm', 'ddim'])
  parser.add_argument('--prompt', type = str, nargs = '+', default = ["A motorcycle"], help = 'Prompts to generate')
  parser.add_argument('--save_path', type = str, default = None)
  parser.add_argument('--save_img', action = 'store_true')
  parser.add_argument('--seed', type = int, default = 42)

  return parser.parse_args()

def main():
  args = parse_args()

  config = load_config(args.config)
  set_seed(args.seed)

  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  ckpt = torch.load(f = args.checkpoint, map_location = device)

  model = UNet(input_dim = config['model']['input_dim'],
              channels = config['model']['channels'],
              context_dim = config['model']['context_dim'],
              dimension_multiplier = tuple(config['model']['dimension_multiplier']),
              attn_config = tuple(config['model']['attention_config']),
              groupnorm_groups = config['model']['groupnorm_groups']).to(device)
  
  vae = FrozenVAE(vae_name = config['autoencoder']['name']).to(device)
  vae.eval()

  clip = FrozenCLIP(clip_name = config['text_encoder']['name']).to(device)
  clip.eval()

  model.load_state_dict(state_dict = ckpt['model_state_dict'])
  model.eval()

  if 'ema_state_dict' in ckpt:
    ema = EMA(model = model, decay = 0.999)
    ema.load_state_dict(state_dict = ckpt['ema_state_dict'])
    ema.apply_shadow()

  if isinstance(args.prompt, str):
    prompt = [args.prompt]

  with torch.inference_mode():
    generate_and_plot(model = model,
                      clip = clip,
                      vae = vae,
                      prompt = prompt,
                      img_shape = ((len(prompt),config['model']['channels'],32,32)),
                      sampler =  args.sampler,
                      timesteps = config['diffusion']['timesteps'],
                      sampling_timesteps = config['sampling']['sampling_timesteps'],
                      pred_type = config['diffusion']['prediction_type'],
                      guidance_scale = config['cfg']['guidance_scale'],
                      eta = config['sampling']['eta'],
                      show_img = True,
                      save_img = args.save_img,
                      save_path = args.save_path,
                      seed = args.seed)

if __name__ == '__main__':
  main()