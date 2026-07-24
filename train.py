import torch

from src.latent_diffusion.utils.seed import set_seed
from src.latent_diffusion.utils.config import load_config
from src.latent_diffusion.data.cached_dataset import get_cached_coco_loader
from src.latent_diffusion.models.unet import UNet
from src.latent_diffusion.training.trainer import trainer
from src.latent_diffusion.autoencoder.vae import FrozenVAE
from src.latent_diffusion.conditioning.clip import FrozenCLIP

def main():

    config = load_config('configs/coco.yaml')

    set_seed(config['seed'])
  
    device = config['device']
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    train_dataloader = get_cached_coco_loader(cache_dir = config['dataset']['cached_root'],
                                            batch_size = config['dataset']['batch_size'],
                                            shuffle = config['dataset']['shuffle'],
                                            pin_memory = config['dataset']['pin_memory'],
                                            num_workers = config['dataset']['num_workers'])
  
    model = UNet(input_dim = config['model']['input_dim'],
                channels = config['model']['channels'],
                context_dim = config['model']['context_dim'],
                dimension_multiplier = tuple(config['model']['dimension_multiplier']),
                attn_config = tuple(config['model']['attention_config']),
                groupnorm_groups = config['model']['groupnorm_groups'])

    vae = FrozenVAE(vae_name = config['autoencoder']['name'])

    clip = FrozenCLIP(clip_name = config['text_encoder']['name'])

    loss_hist, checkpoint = trainer(model=model,
                                    latent_dataloader=train_dataloader,
                                    epochs=config['training']['epochs'],
                                    device=device,
                                    vae = vae,
                                    clip = clip,
                                    pred_type=config['diffusion']['prediction_type'],
                                    lr=config['training']['learning_rate'],
                                    class_free_dropout=config['cfg']['dropout'],
                                    guidance_scale=config['cfg']['guidance_scale'],
                                    eta=config['sampling']['eta'],
                                    ema_decay=config['training']['ema_decay'],
                                    timesteps=config['diffusion']['timesteps'],
                                    schedule_type=config['diffusion']['schedule'],
                                    optim=config['training']['optimizer'],
                                    save_dir=config['training']['save_dir'],
                                    save_every=config['training']['save_every'],
                                    resume_from=config['training']['resume_from'],
                                    sample_every=config['sampling']['sample_every'],
                                    example_prompts=config['sampling']['labels'],
                                    sampler=config['sampling']['sampler'],
                                    sample_timesteps=config['sampling']['sampling_timesteps'],
                                    use_snr = config['training']['use_snr'],
                                    snr_gamma = config['training']['snr_gamma'],
                                    seed = config['seed'])
    return loss_hist, checkpoint

if __name__ == '__main__':
  loss_hist, checkpoint = main()
