
import torchvision
from ._base import _BaseLightningDataModule

__all__ = [
    'CIFAR10LightningDataModule'
]

class CIFAR10LightningDataModule(_BaseLightningDataModule):
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