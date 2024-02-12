import os

import torch
import torchvision

try:
    import lightning as pl
except ImportError:
    import pytorch_lightning as pl

class _BaseData(pl.LightningDataModule):
    def __init__(self, root,
                 dataset_cls,
                 batch_size=512,
                 train_fraction=1.0,
                 train_transform=None,
                 val_transform=None,
                 num_workers=None,
                 pin_memory=False):
        super().__init__()
        self.root = root        
        self.batch_size = batch_size
        self.num_workers = os.cpu_count() if num_workers is None else num_workers
        self.pin_memory = pin_memory

        self._dataset_cls = dataset_cls  # This will receive 'root', 'train', 'download', and, 'transform' arguments
        self._train_fraction = train_fraction
        self._train_transform = train_transform
        self._val_transform = val_transform

    def prepare_data(self):
        self._dataset_cls(root=self.root, train=True, download=True, transform=None)
        self._dataset_cls(root=self.root, train=False, download=True, transform=None)
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_dataset = self._dataset_cls(root=self.root, train=True, download=False, transform=self._train_transform)
            if self._train_fraction < 1.0:
                train_len = int(len(train_dataset) * self._train_fraction)
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                    train_dataset, [train_len, len(train_dataset) - train_len]
                )
            else:
                self.train_dataset = train_dataset
                self.val_dataset = self._dataset_cls(root=self.root, train=False, download=False, transform=self._val_transform)
        elif stage == 'test':
            self.test_dataset = self._dataset_cls(root=self.root, train=False, download=False, transform=self._val_transform)
        elif stage == 'predict':
            self.predict_dataset = self._dataset_cls(root=self.root, train=False, download=False, transform=self._val_transform)
        else:
            raise ValueError(f'Unknown stage {stage}')
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           pin_memory=self.pin_memory,
                                           persistent_workers=(self.num_workers > 0),
                                           shuffle=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           pin_memory=self.pin_memory,
                                           persistent_workers=(self.num_workers > 0),
                                           shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           pin_memory=self.pin_memory,
                                           persistent_workers=(self.num_workers > 0),
                                           shuffle=False)
    
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           pin_memory=self.pin_memory,
                                           persistent_workers=(self.num_workers > 0),
                                           shuffle=False)
                

class CIFAR10Data(_BaseData):
    def __init__(self, root, batch_size=512, train_fraction=1.0,
                 num_workers=None, pin_memory=False):
        val_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32, padding=4),
            val_transform,
        ])
        super().__init__(
            root,
            dataset_cls=torchvision.datasets.CIFAR10,
            batch_size=batch_size,
            train_fraction=train_fraction,
            train_transform=train_transform,
            val_transform=val_transform,
            num_workers=num_workers,
            pin_memory=pin_memory)