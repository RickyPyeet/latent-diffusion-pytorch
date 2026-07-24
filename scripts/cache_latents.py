import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.latent_diffusion.data.cache import cache_vae_latents
from src.latent_diffusion.data.coco import build_coco_trainval_dataset, get_collate_function, get_train_transforms
from src.latent_diffusion.autoencoder.vae import FrozenVAE

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type = Path, required = True, help = 'Path to the COCO dataset directory')
    parser.add_argument('--output_dir', type = Path, required = True, help ='Directory where cached latents are saved')
    parser.add_argument('--vae_model', type = str, default = 'stabilityai/sd-vae-ft-mse', help = 'Pretrained VAE model name or local path')
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--num_workers', type = int, default = 4)
    parser.add_argument('--device', type = str, default = 'cuda')
    parser.add_argument('--overwrite', action = 'store_true', help = 'Overwrite the output_dir folder')

    return parser.parse_args()

def main():
    args = parse_args()

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is not available")

    transform = get_train_transforms()

    dataset = build_coco_trainval_dataset(data = args.data_dir,
                                          transform = transform)
    dataloader = DataLoader(dataset,
                            batch_size = args.batch_size,
                            shuffle = False,
                            num_workers = args.num_workers,
                            pin_memory = args.device.startswith('cuda'),
                            collate_fn = get_collate_function())

    vae = FrozenVAE(args.vae_model)

    cache_vae_latents(vae = vae,
                      dataloader = dataloader,
                      output_dir = args.output_dir,
                      device = args.device,
                      overwrite = args.overwrite)

if __name__ == '__main__':
    main()

