from itertools import chain

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import lightning as L



class BaseMLP(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out=None,
        neurons = [],
        hidden_activation = None,
        output_activation = None,
        batch_norm = False,
        dropout = 0.,
    ):
        super().__init__()

        # Make layers #
        layers = []
        out_neurons = dim_in
        for i in range(len(neurons)):
            in_neurons = neurons[i - 1] if i > 0 else dim_in
            out_neurons = neurons[i]
            layers.append(nn.Linear(in_neurons, out_neurons))
            if hidden_activation is not None:
                layers.append(hidden_activation())
                if batch_norm:
                    layers.append(nn.BatchNorm1d(out_neurons))
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
        if dim_out is not None:
            layers.append(nn.Linear(out_neurons, dim_out))
            if output_activation is not None:
                layers.append(output_activation())
        self.layers = nn.Sequential(*layers)

    def forward(self,x,mask=None):
        if isinstance(x,(list,tuple)):
            x = torch.cat(x,dim=1)
        if mask is not None:
            # zero-padding if mask #
            if isinstance(mask,(list,tuple)):
                mask = torch.cat(mask,dim=1)
            x[~mask] = 0.

        # Need to flatten input and mask #
        if x.dim() == 3:
            x = x.reshape(
                x.shape[0],
                x.shape[1] * x.shape[2],
            )
        elif x.dim() == 2:
            pass
        else:
            raise RuntimeError(f'x dim = {x.dim()}')

        return self.layers(x)

class BaseTransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dims,
        activation,
        num_layers,
        nhead,
        dim_feedforward,
        n_particles_per_type,
        particle_type_names,
        input_features_per_type,
        layer_norm = False,
        dropout = 0.,
    ):
        super().__init__()

        # Public attributes #
        self.dropout = dropout
        self.embed_dims = embed_dims
        if isinstance(self.embed_dims,int):
            self.embed_dims = [self.embed_dims]
        self.num_layers = num_layers
        self.embed_dim = self.embed_dims[-1]
        self.activation = activation
        self.nhead = nhead
        self.layer_norm = layer_norm
        self.dim_feedforward = dim_feedforward
        self.n_particles_per_type = n_particles_per_type
        self.particle_type_names = particle_type_names
        self.input_features_per_type = input_features_per_type

        # Make layers #
        self.embeddings = self.make_embeddings(self.input_features_per_type)
        self.encoder = nn.TransformerEncoder(
            encoder_layer = nn.TransformerEncoderLayer(
                d_model = self.embed_dim,
                nhead = self.nhead,
                dim_feedforward = self.dim_feedforward,
                dropout = self.dropout,
                activation = self.activation(),
                batch_first = True,
            ),
            num_layers = self.num_layers,
            norm = nn.LayerNorm(self.embed_dim) if self.layer_norm else None,
        )

    def make_embeddings(self,input_features):
        feature_embeddings = {}
        embeddings = nn.ModuleList()
        # Make sure inputs with same features are processed through same embedding #
        for features in input_features:
            if not isinstance(features,tuple):
                features = tuple(features)
            if features not in feature_embeddings.keys():
                layers = []
                for i in range(len(self.embed_dims)):
                    layers.append(
                        nn.Linear(
                            in_features = len(features) if i==0 else self.embed_dims[i-1],
                            out_features = self.embed_dims[i],
                        )
                    )
                    if self.activation is not None and i < len(self.embed_dims) - 1:
                        layers.append(self.activation())
                    #if self.dropout != 0.:
                    #    layers.append(nn.Dropout(self.dropout))
                embedding = nn.Sequential(*layers)
                feature_embeddings[features] = embedding
            else:
                embedding = feature_embeddings[features]
            embeddings.append(embedding)
        return embeddings


    def forward(self,xs,masks=None):
        assert len(xs) == len(self.embeddings)
        # Apply embeddings #
        x = torch.cat(
            [
                self.embeddings[i](xs[i])
                for i in range(len(xs))
            ],
            dim = 1,
        )

        # Process mask #
        if masks is not None:
            if isinstance(masks,(list,tuple)):
                mask = torch.cat(masks,dim=1)
            else:
                assert masks.dim() == 2
                mask = masks
        else:
            mask = np.full((x.shape[0],x.shape[1]),fill_value=False)

        # Pass through encoder #
        x = self.encoder(x,src_key_padding_mask=~mask)

        # Use mean pooling #
        den = mask.sum(dim=1)
        x = (x*mask.unsqueeze(-1)).sum(dim=1) / den.unsqueeze(-1)

        return x


class Classifier(L.LightningModule):
    def __init__(
        self,
        backbone,
        head,
        loss_function,
    ):
        super().__init__()

        # Public attributes #
        self.backbone = backbone
        self.head = head
        self.loss_function = loss_function

        # Private attributes #
        self._optimizer = None
        self._scheduler_config = None

        # Store hyperparameters when saving model
        self.save_hyperparameters(ignore=["optimizer", "scheduler_config"])

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
        y = self.forward(batch)
        loss = self.loss_function(y,batch['target'])
        if 'weights' in batch:
            w = torch.cat(batch['weights'],dim=1).mean(dim=-1)
        else:
            w = torch.ones(y.shape[0])
        #loss = (loss * w).mean()
        loss = (loss).mean()
        self.log(f"{prefix}/loss_tot", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'val')

    def forward(self,batch):
        x = self.backbone(batch['data'],batch['mask'])
        x = self.head(x)
        return x
