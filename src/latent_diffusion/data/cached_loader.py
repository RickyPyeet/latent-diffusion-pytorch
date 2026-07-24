from torch.utils.data import DataLoader

from src.latent_diffusion.data.cached_dataset import CachedLatentDataset, collate_cached_batch

def get_cached_coco_dataset(cache_dir):
    return CachedLatentDataset(cache_dir = cache_dir, random_caption = True, horizontal_flip_prob = 0.5)

def get_cached_coco_loader(cache_dir,
                            batch_size,
                            shuffle,
                            pin_memory,
                            num_workers):
    dataset = get_cached_coco_dataset(cache_dir)

    return DataLoader(dataset = dataset,
                      batch_size = batch_size,
                      shuffle = shuffle,
                      pin_memory = pin_memory,
                      num_workers = num_workers,
                      collate_fn = collate_cached_batch)