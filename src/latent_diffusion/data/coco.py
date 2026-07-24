import json
import torch

from torchvision import transforms
from pathlib import Path
from collections import defaultdict
from PIL import Image

from torch.utils.data import Dataset, ConcatDataset



# Custom dataset
class COCODataset(Dataset):
  def __init__(self, root: str | Path , annot_file: str | Path, split: str, transform = None):
    super().__init__()
    self.root = Path(root)
    self.annot_file = Path(annot_file)
    self.split = split
    self.transform = transform

    with self.annot_file.open('r', encoding = 'utf-8') as file:
        data = json.load(file)

    captions_by_image = defaultdict(list)

    for annotation in data['annotations']:
        captions_by_image[annotation['image_id']].append(annotation['caption'])
    
    filenames_by_id = {image['id']: image['file_name'] for image in data['images']}

    self.samples = []

    for image_id, captions in captions_by_image.items():
        file_name = filenames_by_id.get(image_id)
        if file_name is None:
            continue

        self.samples.append(
            {'image_id': image_id,
            'split': self.split,
            'image_path': self.root / file_name,
            'captions': captions})

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    sample = self.samples[idx]

    with Image.open(sample['image_path']) as image:
        image = image.convert('RGB')

    if self.transform is not None:
        image = self.transform(image)

    return {'image': image,
            'captions': sample['captions'],
            'image_id': sample['image_id'],
            'split': sample['split']}

# Train transform
def get_train_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ])
    return train_transform

# Coco transform
def get_collate_function(batch):
    images = [sample['image'] for sample in batch]

    return {'images': torch.stack(images),
            'captions': [sample['captions'] for sample in batch],
            'image_ids': [sample['image_id'] for sample in batch],
            'splits': [sample['split'] for sample in batch]}



# Create dataset
def build_coco_trainval_dataset(data_dir: str | Path,
                                transform = None):
    data_dir = Path(data_dir)

    train_dataset = COCODataset(root = data_dir / "train2017",
                                annot_file = data_dir / "annotations" / "captions_train2017.json",
                                split = 'train2017',
                                transform = transform)
    val_dataset = COCODataset(root = data_dir / "val2017",
                            annot_file = data_dir / "annotations" / 'captions_val2017.json',
                            split = 'val2017',
                            transform = transform)

    return ConcatDataset([train_dataset, val_dataset])
