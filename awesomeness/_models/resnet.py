from torch import nn
import torchvision
import torchmetrics

from ._base import _BaseLightningModule

__all__ = [
    'ResNet18'
]

class ResNet18(_BaseLightningModule):
    def __init__(self, num_outputs=10, fc_hidden=None, weights='DEFAULT',
                 criterion='cross_entropy',
                 optimizer='adam',
                 lr=0.001):
        super().__init__(criterion=criterion, optimizer=optimizer, optimizer_kwargs={'lr': lr})
        # Make model
        self.model = torchvision.models.resnet18(weights=weights)  # This creates 512 output channels before fc
        if not fc_hidden:
            self.model.fc = nn.Linear(512, num_outputs)
        else:
            self.model.fc = nn.Sequential()
            Cin = 512
            for Cout in fc_hidden:
                self.model.fc.append(nn.Linear(Cin, Cout))
                self.model.fc.append(nn.ReLU())
                Cin = Cout
            self.model.fc.append(nn.Linear(Cin, num_outputs))
        self.num_classes = num_outputs if num_outputs > 1 else 2
        self.accuracy = torchmetrics.Accuracy(
                task=('multiclass' if self.num_classes > 1 else 'binary'),
                num_classes=self.num_classes)
        self._extra_metrics['accuracy'] = self.accuracy

        # Save per epoch history
        self.save_hyperparameters()
