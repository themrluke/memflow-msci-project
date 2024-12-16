import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MLP(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        hidden_activation = None,
        output_activation = None,
        batchnorm = False,
        layernorm = False,
        conv_layer = False,
        neurons = [],
        dropout = 0.,
    ):
        # Call abs class #
        super().__init__()

        # Save attributes #
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.neurons = neurons
        self.dropout = dropout

        # Initialize layers #
        self.layers = []
        for i in range(len(self.neurons)+1):
            in_neurons = self.neurons[i - 1] if i > 0 else self.dim_in
            out_neurons = self.neurons[i] if i < len(self.neurons) else self.dim_out
            if conv_layer:
                self.layers.append(nn.Conv1d(in_neurons, out_neurons, 1))
            else:
                self.layers.append(nn.Linear(in_neurons, out_neurons))
            if i < len(self.neurons):
                if self.hidden_activation is not None:
                    self.layers.append(self.hidden_activation())
                if self.layernorm:
                    self.layers.append(nn.LayerNorm(out_neurons))
                if self.batchnorm:
                    self.layers.append(nn.BatchNorm1d(out_neurons))
            else:
                if self.output_activation is not None:
                    self.layers.append(self.output_activation())
            if self.dropout > 0:
                self.layers.append(nn.Dropout(self.dropout))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x, mask=None):
        if mask is not None:
            assert x.shape == mask.shape, f'x has shape {x.shape}, while mask has shape {mask.shape}'
            x = x.masked_fill(~mask,0.)
        return self.layers(x)


