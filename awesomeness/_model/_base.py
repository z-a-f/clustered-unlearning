import torch
from torch import nn

import lightning as pl

class _BaseModel(pl.LightningModule):
    def __init__(self, *,
                 criterion='cross_entropy', criterion_kwargs=None,
                 optimizer='adam', optimizer_kwargs=None):
        super().__init__()
        self._extra_metrics = {}
        self.history = {'train': {'loss': []}, 'val': {'loss': []}, 'epoch': []}
        self.data_lengths = {'train': 0, 'val': 0}
        self.model = None
        # Make criterion
        criterion_kwargs = criterion_kwargs or {}
        match criterion:
            case 'cross_entropy':
                self._criterion = nn.CrossEntropyLoss(**criterion_kwargs)
            case 'nll':
                self._criterion = nn.NLLLoss(**criterion_kwargs)
            case 'mse':
                self._criterion = nn.MSELoss(**criterion_kwargs)
            case 'bce':
                self._criterion = nn.BCELoss(**criterion_kwargs)
            case _:
                raise ValueError(f'Unknown criterion {criterion}')
        # Make optimizer
        self.optimizer_kwargs = optimizer_kwargs or {}
        match optimizer:
            case 'adam':
                self._optimizer_cls = torch.optim.Adam
            case 'sgd':
                self._optimizer_cls = torch.optim.SGD
            case _:
                raise ValueError(f'Unknown optimizer {optimizer}')

    def configure_optimizers(self):
        return self._optimizer_cls(self.parameters(), **self.optimizer_kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)  # _model should be defined in the subclass

    def training_step(self, batch, batch_idx, dataloader_idx=None):
        loss, _, _ = self._shared_step(batch, batch_idx, step='train')
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        loss, _, _ = self._shared_step(batch, batch_idx, step='val')
        return loss

    def on_train_start(self):
        for name in self._extra_metrics.keys():
            self.history['train'][name] = []

    def on_validation_start(self):
        for name in self._extra_metrics.keys():
            self.history['val'][name] = []

    def on_train_epoch_start(self):
        self.data_lengths['train'] = 0
        self.history['train']['loss'].append(0)
        for name in self._extra_metrics.keys():
            self.history['train'][name].append(0)

    def on_train_epoch_end(self):
        self.history['epoch'].append(self.current_epoch)
        self.history['train']['loss'][-1] /= self.data_lengths['train']
        for name in self._extra_metrics.keys():
            self.history['train'][name][-1] /= self.data_lengths['train']

    def on_validation_epoch_start(self):
        self.data_lengths['val'] = 0
        self.history['val']['loss'].append(0)
        for name in self._extra_metrics.keys():
            self.history['val'][name].append(0)

    def on_validation_epoch_end(self):
        self.history['val']['loss'][-1] /= self.data_lengths['val']
        for name in self._extra_metrics.keys():
            self.history['val'][name][-1] /= self.data_lengths['val']

    def _shared_step(self, batch, batch_idx, step=None):
        x, y = batch
        self.data_lengths[step] += len(x)
        y_hat = self(x)
        loss = self._criterion(y_hat, y)
        if step is not None:
            log_dict = {f'{step}_loss': loss}
            self.history[step]['loss'][-1] += loss.item() * x.size(0)
            for name, metric in self._extra_metrics.items():
                metric_val = metric(y_hat, y)
                log_dict[f'{step}_{name}'] = metric_val
                self.history[step][name][-1] += metric_val.item() * x.size(0)
            self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)
        return loss, y_hat, y
