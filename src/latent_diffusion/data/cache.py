from pathlib import Path

import torch
from tqdm.auto import tqdm

@torch.inference_mode()
def cache_vae_latents(vae,
                      dataloader,
                      output_dir: str | Path,
                      device: torch.device | str,
                      overwrite: bool = False):
    """
    Encodes images into latents and caches them into a output_dir folder for future use.
    args:
        - vae = variational autoencoder
        - dataloader = dataloader to be iterated over and cached
        - device = device to send latents and vae to
        - overwrite = overwrites already cached latents if True
    out:
        - None
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents = True, exist_ok = True)

    device = torch.device(device)

    vae = vae.to(device)
    vae.eval()

    for batch in tqdm(dataloader, desc="Caching VAE latents"):
        images = batch['images'].to(device)

        latents = vae.encode(images)
        latents = latents.cpu()

        captions = batch['captions']
        image_ids = batch['image_ids']
        splits = batch['splits']

        for latent, sample_captions, image_id, split in zip(latents, captions, image_ids, splits):
            image_id = int(image_id)
            cache_path = output_dir / f"{split}_{image_id:012d}.pt"
            if cache_path.exists() and not overwrite:
                continue
            
            sample = {'latent': latent,
                      'captions': sample_captions,
                      'image_id': image_id,
                      'split': split}

            torch.save(sample, cache_path)