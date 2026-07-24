import random 
from pathlib import Path

import torch
from torch.utils.data import Dataset

class CachedLatentDataset(Dataset):
    def __init__(self, cache_dir: str | Path, random_caption: bool = True, horizontal_flip_prob: float = 0.0):
        """
        Creates a dataset to handle cached latent vectors.
        args:
            - cache_dir: directory of cached latents
            - random_caption: extract a random caption if turned on, otherwise extract first caption
            - horizontal_flip_prob: probability of applying horizontal flip to latent. This is done to cache only clean latents
        out:
            - dictionary: {latent, caption, image_id, split}
        """
        self.cache_dir = Path(cache_dir)
        self.random_caption = random_caption
        self.horizontal_flip_prob = horizontal_flip_prob

        if not self.cache_dir.is_dir():
            raise FileNotFoundError(f"{self.cache_dir} was not found")

        self.cache_files = sorted(self.cache_dir.glob("*.pt"))

        if not self.cache_files:
            raise RuntimeError(f"No cached latent files found in {self.cache_dir}")

    def __len__(self):
        return len(self.cache_files)

    def __getitem__(self, idx):
        cache_path = self.cache_files[idx]

        sample = torch.load(cache_path, map_location = 'cpu')

        latent = sample['latent']
        captions = sample['captions']

        if random.random() < self.horizontal_flip_prob:
            latent = torch.flip(latent, dim = (-1,))

        if self.random_caption:
            caption = random.choice(captions)
        else:
            caption = captions[0]

        return {'latent': latent,
                'caption': caption,
                'image_id': sample['image_id'],
                'split': sample['split']}

def collate_cached_batch(batch):
    return {'latents': torch.stack([sample['latent'] for sample in batch]),
            'captions': [sample['caption'] for sample in batch],
            'image_ids': [sample['image_id'] for sample in batch],
            'splits': [sample['split'] for sample in batch]}