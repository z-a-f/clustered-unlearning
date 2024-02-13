import os

import torch

import lightning as pl

class _Subset(torch.utils.data.Subset):
    r'''A subset of a dataset.
    
    From https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset
    '''
    def __init__(self, another_subset: torch.utils.data.Subset, transform=None):
        assert isinstance(another_subset, torch.utils.data.Subset)
        super().__init__(another_subset.dataset, another_subset.indices)
        self.transform = transform

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __getitems__(self, indices):
        if callable(getattr(self.dataset, '__getitems__', None)):
            return self.transform(self.dataset.__getitems__([self.indices[idx] for idx in indices]))
        else:
            return [self.__getitem__(idx) for idx in indices]
    
    def __len__(self):
        return len(self.indices)


class _BaseLightningDataModule(pl.LightningDataModule):
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
        self.train_fraction = train_fraction
        self.train_transform = train_transform
        self.val_transform = val_transform

    def prepare_data(self):
        self._dataset_cls(root=self.root, train=True, download=True, transform=None)
        self._dataset_cls(root=self.root, train=False, download=True, transform=None)
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            if self.train_fraction < 1.0:
                train_dataset = self._dataset_cls(root=self.root, train=True, download=False, transform=None)
                train_len = int(len(train_dataset) * self.train_fraction)
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                    train_dataset, [train_len, len(train_dataset) - train_len]
                )
                self.train_dataset = _Subset(self.train_dataset)
                self.val_dataset = _Subset(self.val_dataset)
            else:
                self.train_dataset = train_dataset
                self.val_dataset = self._dataset_cls(root=self.root, train=False, download=False, transform=None)
            self.train_dataset.transform = self.train_transform
            self.val_dataset.transform = self.val_transform
        elif stage == 'test':
            self.test_dataset = self._dataset_cls(root=self.root, train=False, download=False, transform=self.val_transform)
        elif stage == 'predict':
            self.predict_dataset = self._dataset_cls(root=self.root, train=False, download=False, transform=self.val_transform)
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
