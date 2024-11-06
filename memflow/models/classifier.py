from itertools import chain

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import lightning as L

class BaseClassifier(L.LightningModule):
    def __init__(
        self,
        number_objects,
        dim_features,
        neurons = [],
        hidden_activation = None,
        batch_norm = False,
        dropout = 0.,
    ):
        super().__init__()

        # Public attributes #
        self.number_objects = number_objects
        self.dim_features = dim_features
        self.neurons = neurons
        self.hidden_activation = hidden_activation
        self.batch_norm = batch_norm
        self.dropout = dropout

        # Private attributes #
        self._optimizer = None
        self._scheduler_config = None

        # Make layers #
        self.layers = []
        for i in range(len(self.neurons)):
             in_neurons = self.neurons[i - 1] if i > 0 else self.number_objects * self.dim_features
             out_neurons = self.neurons[i]
             self.layers.append(nn.Linear(in_neurons, out_neurons))
             if self.hidden_activation is not None:
                 self.layers.append(self.hidden_activation())
             if self.batch_norm:
                 self.layers.append(nn.BatchNorm1d(out_neurons))
             if self.dropout > 0:
                 self.layers.append(nn.Dropout(self.dropout))
        self.hidden_layers = nn.Sequential(*self.layers)
        self.output_layer = self.make_output_layer()

    def make_output_layer(self):
        raise NotImplementedError

    def set_optimizer(self,optimizer):
        self._optimizer = optimizer

    def set_scheduler_config(self,scheduler_config):
        self._scheduler_config = scheduler_config

    def configure_optimizers(self):
        if self._optimizer is None:
            raise RuntimeError('Optimizer not set')
        if self._scheduler_config is None:
            return self._optimizer
        else:
            return {
                'optimizer' : self._optimizer,
                'lr_scheduler': self._scheduler_config,
            }

    def shared_eval(self, batch, batch_idx, prefix):
        if len(batch) == 2:
            x,t = batch
            w = None
        elif len(batch) == 3:
            x,t,w = batch
        else:
            raise RuntimeError

        y = self.forward(x)
        loss = self.loss_function(y,t)
        if w is not None:
            loss *= w
        loss = loss.mean()
        self.log(f"{prefix}/loss_tot", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'val')

    def forward(self,x):
        if x.dim() == 3:
            # Linearize for DNN
            x = x.reshape(x.shape[0],-1)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x


class BinaryClassifier(BaseClassifier):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        # Loss function #
        self.loss_function = nn.BCELoss(reduction="none")


    def make_output_layer(self):
        return nn.Sequential(
            nn.Linear(self.neurons[-1],1),
            nn.Sigmoid(),
        )

class MultiClassifier(BaseClassifier):
    def __init__(self,n_max,**kwargs):
        self.n_max = n_max
        super().__init__(**kwargs)

        # Loss function #
        self.loss_function = nn.CrossEntropyLoss(reduction="none")


    def make_output_layer(self):
        return nn.Sequential(
            nn.Linear(self.neurons[-1],self.n_max),
            nn.Softmax(dim=1),
        )

