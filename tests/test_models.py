
import unittest

import torch
import torchmetrics

class TestBaseLightningModule(unittest.TestCase):
    def setUp(self):
        import awesomeness.models
        self.base_model = awesomeness.models.BaseLightningModule(
            criterion='nll',
            criterion_kwargs={'reduction': 'sum'},
            optimizer='sgd',
            optimizer_kwargs={'lr': 0.42},
        )

    def test_constructor(self):
        # Check if criterion is properly setup
        self.assertIsInstance(self.base_model.criterion, torch.nn.NLLLoss)
        self.assertEqual(self.base_model.criterion.reduction, 'sum')
        # Check if criterion can be changed
        self.base_model.criterion = torch.nn.CrossEntropyLoss()
        self.assertIsInstance(self.base_model.criterion, torch.nn.CrossEntropyLoss)

        # No model is attached, so configure_optimizers should fail
        with self.assertRaises(ValueError):
            self.base_model.configure_optimizers()
        self.assertEqual(self.base_model._optimizer_cls, torch.optim.SGD)
        self.assertEqual(self.base_model.optimizer_kwargs['lr'], 0.42)

        # Check if the history and data_lengths are properly setup
        default_datasets = ['train', 'val']
        for dataset in default_datasets:
            self.assertEqual(self.base_model.history[dataset]['loss'], [])
            self.assertEqual(self.base_model.data_lengths[dataset], 0)
        self.assertEqual(self.base_model.history['epoch'], [])

        # Make sure the model is not defined
        self.assertIsNone(self.base_model.model)

    def test_model(self):
        # Make sure the model is not defined
        self.assertIsNone(self.base_model.model)

        # Make sure the model can be set
        self.base_model.model = torch.nn.Linear(10, 1)
        self.assertIsInstance(self.base_model.model, torch.nn.Linear)
        self.assertEqual(self.base_model.model.in_features, 10)
        self.assertEqual(self.base_model.model.out_features, 1)
        self.assertIsNotNone(self.base_model.model.bias)

        # Make sure the model can be changed
        self.base_model.model = torch.nn.Linear(7, 3, bias=False)
        self.assertIsInstance(self.base_model.model, torch.nn.Linear)
        self.assertEqual(self.base_model.model.in_features, 7)
        self.assertEqual(self.base_model.model.out_features, 3)
        self.assertIsNone(self.base_model.model.bias)

        # Make sure the model can be removed
        self.base_model.model = None
        self.assertIsNone(self.base_model.model)

    def test_forward(self):
        # Make sure the forward method fails if the model is not defined
        with self.assertRaises(AssertionError):
            self.base_model(torch.rand(1, 10))

        # Make sure the forward method works if the model is defined
        self.base_model.model = torch.nn.Linear(5, 7)
        self.assertEqual(self.base_model(torch.rand(3, 5)).shape, torch.Size([3, 7]))
        
    def test_shared_step(self):
        # Make sure the shared step fails if the model is not defined
        with self.assertRaises(AssertionError):
            self.base_model._shared_step(
            batch=(torch.rand(3, 5),
                   torch.randint(0, 7, (3,))),
            batch_idx=0,
            step='train')

        # Make sure the shared step works if the model is defined
        self.base_model.model = torch.nn.Linear(5, 7)
        # Because the history is empty, we need to call 'on_train_start' or 'on_validation_start'
        # before calling the shared step. We also need to to call 'on_train_epoch_start' or 'on_validation_epoch_start'
        with self.assertRaises(IndexError):
            self.base_model._shared_step(
            batch=(torch.rand(3, 5),
                   torch.randint(0, 7, (3,))),
            batch_idx=0,
            step='train')
        
        self.base_model.on_train_start()
        for _ in range(11):
            self.base_model.on_train_epoch_start()
            loss, y_hat, y = self.base_model._shared_step(
                batch=(torch.rand(3, 5),
                    torch.randint(0, 7, (3,))),
                batch_idx=0,
                step='train')
            self.base_model.on_train_epoch_end()
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertIsInstance(y_hat, torch.Tensor)
        self.assertEqual(y_hat.shape, torch.Size([3, 7]))

        # Check if the history is properly updated
        self.assertEqual(len(self.base_model.history['train']['loss']), 11)
        self.assertEqual(self.base_model.data_lengths['train'], 3)


class TestResNet18LightningModule(unittest.TestCase):
    def setUp(self):
        import awesomeness.models
        self.resnet18 = awesomeness.models.ResNet18(
            num_outputs=10,
            fc_hidden=None,
            weights='DEFAULT',
            criterion='cross_entropy',
            optimizer='adam',
            lr=0.001
        )
        self.resnet18.accuracy = torchmetrics.Accuracy(
            task='multiclass',
            num_classes=10
        )

    def test_constructor(self):
        # Make sure the model is properly setup
        self.assertIsInstance(self.resnet18.model, torch.nn.Module)
        self.assertIsInstance(self.resnet18.criterion, torch.nn.CrossEntropyLoss)
        self.assertEqual(self.resnet18.optimizer_kwargs['lr'], 0.001)
        self.assertIsInstance(self.resnet18.optimizer_kwargs, dict)
        self.assertEqual(self.resnet18._optimizer_cls, torch.optim.Adam)
        
        # Make sure the history and data_lengths are properly setup
        default_datasets = ['train', 'val']
        for dataset in default_datasets:
            self.assertEqual(self.resnet18.history[dataset]['loss'], [])
            self.assertEqual(self.resnet18.data_lengths[dataset], 0)
        self.assertEqual(self.resnet18.history['epoch'], [])

        # Make sure the accuracy metric is properly setup
        self.assertIsInstance(self.resnet18.accuracy, torchmetrics.classification.MulticlassAccuracy)
        self.assertEqual(self.resnet18.accuracy.num_classes, 10)

    def test_configure_optimizers(self):
        # Make sure the configure_optimizers method works
        optimizers = self.resnet18.configure_optimizers()
        self.assertIsInstance(optimizers, torch.optim.Adam)

    def test_forward(self):
        # Make sure the forward method works
        self.assertEqual(self.resnet18(torch.rand(3, 3, 32, 32)).shape, torch.Size([3, 10]))

    def test_training_step(self):
        # Make sure the training_step method works
        self.resnet18.on_train_start()
        self.resnet18.on_train_epoch_start()
        loss = self.resnet18.training_step(
            batch=(torch.rand(3, 3, 32, 32),
                   torch.randint(0, 10, (3,))),
            batch_idx=0,
            dataloader_idx=None)
        self.resnet18.on_train_epoch_end()
        self.assertTrue(isinstance(loss, torch.Tensor))
        self.assertEqual(loss.shape, torch.Size([]))

    def test_validation_step(self):
        # Make sure the validation_step method works
        self.resnet18.on_validation_start()
        self.resnet18.on_validation_epoch_start()
        loss = self.resnet18.validation_step(
            batch=(torch.rand(3, 3, 32, 32),
                   torch.randint(0, 10, (3,))),
            batch_idx=0,
            dataloader_idx=None)
        self.resnet18.on_validation_epoch_end()
        self.assertTrue(isinstance(loss, torch.Tensor))
        self.assertEqual(loss.shape, torch.Size([]))

