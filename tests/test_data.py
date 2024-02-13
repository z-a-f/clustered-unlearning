import unittest
import tempfile

import torch

import os

class _DummyDataset:
    file_list = []  # For test and cleanup, keep this persistent

    def __init__(self, root, train, download, transform):
        self.root = root
        self.train = train
        self.download = download
        self.transform = transform

        # Create dummy files
        for i in range(3):
            self.file_list.append(tempfile.NamedTemporaryFile(dir=self.root, delete=False))

    def __len__(self):
        return 11
    
    def __getitem__(self, idx):
        return torch.randn(1, 28, 28), torch.randint(0, 10, (1,))


class TestBaseLightningDataModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempfolder = tempfile.TemporaryDirectory()
    
    @classmethod
    def tearDownClass(cls):
        for file in _DummyDataset.file_list:
            file.close()
            os.remove(file.name)
        cls.tempfolder.cleanup()

    def setUp(self):
        import awesomeness.data
        self.base_data = awesomeness.data.BaseLightningDataModule(
            root=self.tempfolder.name,
            dataset_cls=_DummyDataset,
            batch_size=32,
            num_workers=4,
            pin_memory=True,
            train_fraction=0.8,
            train_transform=None,
            val_transform=None,
        )

    def test_constructor(self):
        # Make sure the BaseLightningDataModule is properly setup
        self.assertEqual(self.base_data.root, self.tempfolder.name)
        self.assertEqual(self.base_data.batch_size, 32)
        self.assertEqual(self.base_data.num_workers, 4)
        self.assertTrue(self.base_data.pin_memory)
        self.assertEqual(self.base_data.train_fraction, 0.8)
        self.assertIsNone(self.base_data.train_transform)
        self.assertIsNone(self.base_data.val_transform)
    
    def test_prepare_data(self):
        # Make sure the prepare_data method works
        self.base_data.prepare_data()
        listdir = set(map(lambda f: os.path.join(self.tempfolder.name, f), os.listdir(self.tempfolder.name)))
        listexp = set(map(lambda f: f.name, _DummyDataset.file_list))
        self.assertEqual(listdir, listexp)
    
    def test_setup(self):
        # Make sure the setup method works
        self.base_data.setup('fit')
        self.assertEqual(len(self.base_data.train_dataset), 8)
        self.assertEqual(len(self.base_data.val_dataset), 3)
        self.base_data.setup('test')
        self.assertEqual(len(self.base_data.test_dataset), 3)
        self.base_data.setup('predict')
        self.assertEqual(len(self.base_data.predict_dataset), 3)
        with self.assertRaises(ValueError):
            self.base_data.setup('unknown')
