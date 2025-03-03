import math
import copy
import torch
import torch.nn as nn
from itertools import chain

import lightning as L

from abc import ABCMeta,abstractmethod

import zuko
from zuko.distributions import BoxUniform
from zuko.distributions import DiagNormal
import numpy as np


class AbsModel(L.LightningModule,metaclass=ABCMeta):
    def __init__(self,optimizer=None,scheduler_config=None,**kwargs):
        super().__init__(**kwargs)

        # Private attributes #
        self._optimizer = optimizer
        self._scheduler_config = scheduler_config

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

    def training_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'val')

    @abstractmethod
    def shared_eval(self, batch, batch_idx, prefix):
        pass

class MLP(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out=None,
        neurons = [],
        hidden_activation = None,
        output_activation = None,
        dropout = 0.,
        batch_norm = False,
        normalize = False,
        flatten_sequence = False
    ):
        super().__init__()

        self.flatten_sequence = flatten_sequence

        # Make layers #
        layers = []
        if normalize:
            layers.append(nn.BatchNorm1d(dim_in))
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
            if self.flatten_sequence:
                x = x.reshape(
                    x.shape[0],
                    x.shape[1] * x.shape[2],
                )
        elif x.dim() == 2:
            pass
        else:
            raise RuntimeError(f'x dim = {x.dim()}')

        return self.layers(x)


class Embedding(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dims,
        hidden_activation = None,
        output_activation = None,
        add_dim = 0,
        dropout = 0,
    ):
        super().__init__()

        # Save attributes #
        self.input_dim = input_dim
        if not isinstance(embed_dims,(tuple,list)):
            assert isinstance(embed_dims,int)
            embed_dims = [embed_dims]
        self.embed_dims = embed_dims
        self.embed_dim = embed_dims[-1]
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.add_dim = add_dim
        self.dropout = dropout

        # Make layers #
        layers = []
        for i in range(len(self.embed_dims)):
            layers.append(
                nn.Linear(
                    in_features = self.input_dim + self.add_dim if i==0 else self.embed_dims[i-1],
                    out_features = self.embed_dims[i],
                )
            )
            if self.hidden_activation is not None and i < len(self.embed_dims) - 1:
                layers.append(self.hidden_activation())
            if self.dropout != 0.:
                layers.append(nn.Dropout(self.dropout))
        if self.output_activation is not None:
            layers.append(self.output_activation())

        self.embedding = nn.Sequential(*layers)

    def forward(self,x):
        return self.embedding(x)


class MultiEmbeddings(nn.Module):
    def __init__(
        self,
        features_per_type,
        embed_dims,
        hidden_activation = None,
        output_activation = None,
        dropout = 0,
        batch_norm = False,
        normalize = False,
    ):
        super().__init__()

        # Save attributes #
        self.features_per_type = features_per_type
        if not isinstance(embed_dims,(tuple,list)):
            assert isinstance(embed_dims,int)
            embed_dims = [embed_dims]
        self.embed_dim = embed_dims[-1]

        # Make embeddings (using the same one for same features as input) #
        feature_embeddings = {}
        self.embeddings = nn.ModuleList()
        for features in self.features_per_type:
            # Make sure hashable for dict #
            if not isinstance(features,tuple):
                features = tuple(features)
            # Get the embedding if already made, else make it #
            if features in feature_embeddings.keys():
                embedding = feature_embeddings[features]
            else:
                embedding = MLP(
                    dim_in = len(features),
                    dim_out = embed_dims[-1],
                    neurons = embed_dims[:-1],
                    hidden_activation = hidden_activation,
                    output_activation = output_activation,
                    dropout = dropout,
                    batch_norm = batch_norm,
                    normalize = normalize,
                )
            # Save in modules #
            self.embeddings.append(embedding)

    def __len__(self):
        return len(self.embeddings)

    def forward(self,xs):
        assert len(xs) == len(self.embeddings), f'{len(xs)} inputs but {len(self.embeddings)} embeddings'
        return torch.cat(
            [
                self.embeddings[i](xs[i])
                for i in range(len(xs))
            ],
            dim = 1
        )

class InverseEmbeddings(nn.Module):
    def __init__(
        self,
        features_per_type,
        n_per_type,
        embed_dims,
        hidden_activation = None,
        output_activation = None,
        dropout = 0,
        batch_norm = False,
        normalize = False,
    ):
        super().__init__()

        # Public attributes #
        self.n_per_type = n_per_type
        self.features_per_type = features_per_type

        # Private attributes #
        #self.

        # Make embeddings (using the same one for same features as input) #
        feature_embeddings = {}
        self.embeddings = nn.ModuleList()
        for features in self.features_per_type:
            # Make sure hashable for dict #
            if not isinstance(features,tuple):
                features = tuple(features)
            # Get the embedding if already made, else make it #
            if features in feature_embeddings.keys():
                embedding = feature_embeddings[features]
            else:
                embedding = MLP(
                    dim_out = len(features),
                    dim_in = embed_dims[0],
                    neurons = embed_dims[1:],
                    hidden_activation = hidden_activation,
                    output_activation = output_activation,
                    dropout = dropout,
                    batch_norm = batch_norm,
                    normalize = normalize,
                )
            # Save in modules #
            self.embeddings.append(embedding)

    def __len__(self):
        return len(self.embeddings)

    def forward(self,x):
        assert x.shape[1] == sum(self.n_per_type), f'x has shape {x.shape}, but n_per_type is {self.n_per_type} which sums up to {sum(self.n_per_type)}'
        xs = torch.split(x,self.n_per_type,dim=1)
        assert len(xs) == len(self.embeddings)
        return [
            self.embeddings[i](xs[i])
            for i in range(len(xs))
        ]


class AbsolutePositionalEncoding(nn.Module):
    """ From pytoch tutorials """
    def __init__(self,d_model,max_seq_len,frequency=10000.,dropout=0.):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(frequency) / d_model))
        pe = torch.zeros(1, max_seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class EntityEmbedding(nn.Module):
    """ From pytoch tutorials """
    def __init__(self,d_model,max_seq_len,dropout=0.):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(
            num_embeddings = max_seq_len,
            embedding_dim = d_model,
        )

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        positions = torch.arange(x.shape[1]).unsqueeze(0).repeat(x.shape[0],1).to(x.device)
        x = x + self.embedding(positions)
        return self.dropout(x)


class TransformerClassLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, activation, dropout=0, batch_first=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = nhead,
            dropout     = dropout,
            batch_first = batch_first,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1    = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2    = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x, x_cls, key_padding_mask=None, attn_mask=None):
        # Concatenate the class token along the sequence length
        u = torch.cat((x_cls, x), dim=1) # (batch,seq_length,embed_dim)
        # if padding mask is provided, need to extend to the additional sequence
        if key_padding_mask is not None:
            with torch.no_grad():
                key_padding_mask = torch.cat(
                    (
                        torch.zeros_like(key_padding_mask[:, :1]),
                        key_padding_mask,
                    ),
                    dim=1,
                )
        # Attention #
        x_att,_ = self.attn(
            x_cls, u, u,
            key_padding_mask = key_padding_mask,
            attn_mask = attn_mask,
        )
        x_att = self.dropout1(x_att)
        # First add + norm #
        x = self.norm1(x_cls + x_att)
        # Feed forward #
        x_ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x_ff = self.dropout2(x_ff)
        # Second add + norm #
        x = self.norm2(x + x_ff)
        return x


class TransformerClass(nn.Module):
    def __init__(self, class_layer, num_layers, norm=None):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                copy.deepcopy(class_layer)
                for _ in range(num_layers)
            ]
        )
        if norm is None:
            self.norm_layers = nn.ModuleList(
                [
                    None
                    for _ in range(num_layers)
                ]
            )
        else:
            self.norm_layers = nn.ModuleList(
                [
                    copy.deepcopy(norm)
                    for _ in range(num_layers)
                ]
            )

    def forward(self, x, x_cls, key_padding_mask=None, attn_mask=None):
        for layer,norm in zip(self.layers,self.norm_layers):
            x_cls = layer(x,x_cls,key_padding_mask,attn_mask)
            if norm is not None:
                x_cls = norm(x_cls)
        return x_cls




class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layers,
        nhead,
        dim_feedforward,
        activation,
        embeddings = None,
        embed_dim = None,
        pooling = False,
        class_layers = 0,
        layer_norm = False,
        dropout = 0.,
        causal_mask = None,
        position_encoding = None,
    ):
        super().__init__()

        # Public attributes #
        self.embeddings = embeddings
        if self.embeddings is None:
            assert embed_dim is not None
            self.embed_dim = embed_dim
        else:
            if embed_dim is not None and embed_dim != self.embeddings.embed_dim:
                raise RuntimeError(f'Provided embed dim is {embed_dim}, but got {self.embeddings.embed_dim} from embeddings')
            self.embed_dim = self.embeddings.embed_dim
        self.encoder_layers = encoder_layers
        self.class_layers = class_layers
        self.pooling = pooling
        self.activation = activation
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.layer_norm = layer_norm
        self.causal_mask = causal_mask
        self.dropout = dropout
        self.position_encoding = position_encoding

        if self.pooling and len(self.class_layers) > 0:
            raise RuntimeError(f'Cannot use both mean pooling and class layers')

        # Make layers #
        self.encoder = nn.TransformerEncoder(
            encoder_layer = nn.TransformerEncoderLayer(
                d_model = self.embed_dim,
                nhead = self.nhead,
                dim_feedforward = self.dim_feedforward,
                dropout = self.dropout,
                activation = self.activation(),
                batch_first = True,
            ),
            num_layers = self.encoder_layers,
            norm = nn.LayerNorm(self.embed_dim) if self.layer_norm else None,
        )
        self.class_encoder = TransformerClass(
            class_layer = TransformerClassLayer(
                d_model = self.embed_dim,
                nhead = self.nhead,
                dim_feedforward = self.dim_feedforward,
                dropout = self.dropout,
                activation = self.activation(),
                batch_first = True,
            ),
            num_layers = self.class_layers,
            norm = nn.LayerNorm(self.embed_dim) if self.layer_norm else None,
        )
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, self.embed_dim),
            requires_grad = True,
        )
        nn.init.trunc_normal_(self.cls_token,std=.02)

        if self.causal_mask is not None:
            print ('Using causal mask')
            print (self.causal_mask)


    def forward(self,xs,masks=None):
        if self.embeddings is None:
            # No embedding, xs must be a tensor
            assert torch.is_tensor(xs), f'xs must be tensor, is {type(xs)}'
            if masks is not None:
                assert torch.is_tensor(masks), f'masks must be tensor, is {type(masks)}'
            x = xs
        else:
            # Apply embeddings #
            x = self.embeddings(xs)

        # Position encoding #
        if self.position_encoding is not None:
            x = self.position_encoding(x)

        # Process mask #
        if masks is not None:
            if isinstance(masks,(list,tuple)):
                mask = torch.cat(masks,dim=1)
            else:
                assert masks.dim() == 2
                mask = masks
        else:
            mask = np.full((x.shape[0],x.shape[1]),fill_value=False)
        if self.causal_mask is not None:
            self.causal_mask = self.causal_mask.to(x.device)

        # Pass through encoder #
        x = self.encoder(
            x,
            mask = self.causal_mask,
            src_key_padding_mask = ~mask,
        )

        # Pass through decoder #
        if self.pooling:
            # Use mean pooling #
            den = mask.sum(dim=1)
            x = (x*mask.unsqueeze(-1)).sum(dim=1) / den.unsqueeze(-1)
        if self.class_layers > 0:
            cls_token = self.cls_token.expand(x.size(0), 1, -1)
            # (batch, 1 (collapsed seq length), embed dim)
            cls_token = self.class_encoder(x,cls_token,key_padding_mask=~mask)
            x = cls_token.squeeze(1)

        return x

    def encode(self,xs,masks):
        return self(xs,masks)


class Transformer(nn.Module):
    def __init__(
        self,
        d_model,
        encoder_layers,
        decoder_layers,
        nhead,
        dim_feedforward,
        activation,
        encoder_mask_attn,
        decoder_mask_attn,
        dropout = 0.,
        use_null_token = False,
    ):
        super().__init__()

        # Save arguments #
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.activation = activation
        self.encoder_mask_attn = encoder_mask_attn
        self.decoder_mask_attn = decoder_mask_attn
        self.dropout = dropout
        self.use_null_token = use_null_token

        # Modify attention mask for decoder if null token #
        if self.use_null_token:
            self.decoder_mask_attn = torch.cat(
                (
                    torch.tensor([True]),
                    self.decoder_mask_attn,
                ),
                dim = 0,
            )

        # Declare transformer #
        self.transformer = nn.Transformer(
            d_model = self.d_model,
            nhead = self.nhead,
            num_encoder_layers = self.encoder_layers,
            num_decoder_layers = self.decoder_layers,
            dim_feedforward = self.dim_feedforward,
            dropout = self.dropout,
            activation = self.activation(),
            batch_first = True,
        )

    def forward(self,x_enc,m_enc,x_dec,m_dec):
        # Expand attention mask #
        # Need to turn 0->1 when particle exists #
        if self.encoder_mask_attn is None:
            encoder_mask_attn = m_enc
        else:
            encoder_mask_attn  = torch.logical_or(
                self.encoder_mask_attn.to(m_enc.device),
                m_enc,
            )
        if self.decoder_mask_attn is None:
            decoder_mask_attn = m_dec
        else:
            decoder_mask_attn = torch.logical_or(
                self.decoder_mask_attn.to(m_dec.device),
                m_dec,
            )
        # -> Make sure that particles we want in the attention are considered even if missing
        # (in which case the default values are set in the dataset class, no need to re default them)
        # Turn them into boolean arrays #
        if encoder_mask_attn.dtype != torch.bool:
            encoder_mask_attn = encoder_mask_attn > 0
        if decoder_mask_attn.dtype != torch.bool:
            decoder_mask_attn = decoder_mask_attn > 0
        # replace True->0, False->-inf
        # To have same dtype as tgt_mask
        encoder_mask_attn = torch.zeros_like(encoder_mask_attn).to(torch.float32).masked_fill(~encoder_mask_attn,float("-inf"))
        decoder_mask_attn = torch.zeros_like(decoder_mask_attn).to(torch.float32).masked_fill(~decoder_mask_attn,float("-inf"))

        # make causal mask #
        tgt_mask = self.transformer.generate_square_subsequent_mask(x_dec.shape[1]).to(x_enc.device)

        # Transformer processing #
        condition = self.transformer(
            src = x_enc,                                       # encoder input
            tgt = x_dec,                                       # decorder input
            tgt_mask = tgt_mask,                               # triangular (causality) mask
            src_key_padding_mask = encoder_mask_attn,          # encoder mask
            memory_key_padding_mask = encoder_mask_attn,       # encoder / memory mask
            tgt_key_padding_mask = decoder_mask_attn,          # decoder mask
        )

        return condition

class KinematicFlow(nn.Module):
    def __init__(
        self,
        d_model,
        flow_mode,
        flow_features,
        flow_classes = {},
        flow_common_args = {},
        flow_specific_args = {},
        dropout = 0.,
    ):
        super().__init__()

        # Save arguments #
        self.d_model = d_model
        self.flow_mode = flow_mode
        self.flow_features = flow_features
        self.flow_classes = flow_classes
        self.flow_common_args = flow_common_args
        self.flow_specific_args = flow_specific_args
        self.dropout = dropout

        # Override #
        if 'context' in self.flow_common_args:
            print (f'Flow args: will override `context` depending on the flow features for each object')
            del self.flow_common_args['context']
        if 'features' in self.flow_common_args:
            print (f'Flow args: will override `features` to 1, as our model has 1D flows')
        self.flow_common_args['features'] = 1

        # Make flows #
        if self.flow_mode == 'global':
            # One common flow for all, need to find global features #
            # trick below to remove duplicates while keeping order
            self.global_flow_features = list(dict.fromkeys(chain.from_iterable(flow_features)))
            self.flows = self.make_flow_block(self.global_flow_features)
        elif self.flow_mode == 'type':
            # One flow per type #
            self.flows = nn.ModuleList(
                [
                    self.make_flow_block(features)
                    for features in flow_features
                ]
            )
        else:
            raise RuntimeError(f'Flow mode should be `global` or `type`, got {self.flow_mode}')

    def make_flow_block(self,features):
        flows = nn.ModuleDict()
        for feature in features:
            # Find class #
            if feature not in self.flow_classes.keys():
                raise NotImplementedError(f'No class found for feature {feature}')
            flow_cls = self.flow_classes[feature]
            # Specific args #
            if feature in self.flow_specific_args.keys():
                add_args = self.flow_specific_args[feature]
            else:
                add_args = {}
            # Modify context based on feature #
            if feature == 'pt':
                context_dim = sum([other_feat == feat for other_feat in ['eta','phi'] for feat in features])
                # include reco phi + reco eta (if present)
            elif feature == 'eta':
                context_dim = sum([other_feat == feat for other_feat in ['pt','phi'] for feat in features])
                # include latent pt + reco phi (if present)
            elif feature == 'phi':
                context_dim = sum([other_feat == feat for other_feat in ['pt','eta'] for feat in features])
                # include latent pt + latent eta (if present)
            elif feature == 'mass' or feature == 'm':
                context_dim = sum([other_feat == feat for other_feat in ['pt','eta','phi'] for feat in features])
                # include latent pt+eta+phi (if present)
            else:
                raise NotImplementedError(f'This model does not include feature {feature}')
            # Initialise #
            flows[feature] = flow_cls(
                context = self.d_model+ context_dim,
                **self.flow_common_args,
                **add_args,
            )
        return flows

    def pad_input(self,x,features):
        values = []
        mask   = []
        for j,feature in enumerate(self.global_flow_features):
            if feature in features:
                values.append(x[...,features.index(feature)].unsqueeze(-1))
                mask.append(torch.ones((x.shape[0],x.shape[1],1)).to(x.device))
            else:
                values.append(torch.zeros(x.shape[0],x.shape[1],1).to(x.device))
                mask.append(torch.zeros((x.shape[0],x.shape[1],1)).to(x.device))
        return torch.cat(values,dim=-1),torch.cat(mask,dim=-1)


    def process_flow_block(self,x,c,m,flow_dict,flow_features,mode):
        if m is None:
            m = torch.ones_like(x)
        log_probs = torch.zeros((x.shape[0],x.shape[1])).to(x.device)
        if mode not in ['log_prob','sample']:
            raise RuntimeError(f'Available modes are `log_prob` and `sample`, not `{mode}`')
        # loop over features #
        for j, flow_feature in enumerate(flow_features):
            assert flow_feature in flow_dict.keys(), f'Could not find `{flow_feature}` in {flow_dict.keys()}'
            flow = flow_dict[flow_feature]
            # Seclect features to add to condition (for autoregressive) #
            if flow_feature == 'pt':
                features_for_condition = ['eta','phi']
            elif flow_feature == 'eta':
                features_for_condition = ['pt','phi']
            elif flow_feature == 'phi':
                features_for_condition = ['pt','eta']
            elif flow_feature == 'mass' or flow_feature == 'm':
                features_for_condition = ['pt','eta','phi']
            else:
                raise RuntimeError
            # Concatenate condition from tranformer + feature #
            cf = torch.cat(
                [
                    x[...,flow_features.index(feature)].unsqueeze(-1)
                    for feature in features_for_condition if feature in flow_features
                ],
                dim = -1,
            )
            ct = torch.cat((c,cf),dim=-1)
            # Get log_prob #
            if mode == 'log_prob':
                log_probs += flow(ct).log_prob(x[...,j].unsqueeze(-1)) * m[...,j]
            # Sample and replace #
            if mode == 'sample':
                x[...,flow_features.index(flow_feature)] = flow(ct).sample().squeeze(-1)
        if mode == 'log_prob':
            return log_probs
        if mode == 'sample':
            return x

    def forward(self,xs,cs):
        assert len(xs) == len(cs), f'Got {len(xs)} inputs and {len(cs)} conditions'
        # Global mode : treat all particles together #
        if self.flow_mode == 'global':
            # x needs to be padded before concat #
            ms = []
            for i in range(len(xs)):
                x,m = self.pad_input(xs[i],self.flow_features[i])
                xs[i] = x
                ms.append(m)
            log_probs = self.process_flow_block(
                x = torch.cat(xs,dim=1),
                c = torch.cat(cs,dim=1),
                m = torch.cat(ms,dim=1),
                flow_dict = self.flows,
                flow_features = self.global_flow_features,
                mode = 'log_prob',
            )
        # Type mode : treat particles per type #
        if self.flow_mode == 'type':
            assert len(xs) == len(self.flows), f'In `type` mode, got {len(xs)} inputs and {len(self.flows)} flows'
            # Loop over types, process and concat #
            log_probs = torch.cat(
                [
                    self.process_flow_block(
                        x = xs[i],
                        c = cs[i],
                        m = None,
                        flow_dict = self.flows[i],
                        flow_features = self.flow_features[i],
                        mode = 'log_prob',
                    )
                    for i in range(len(self.flows))
                ],
                dim = -1
            )
        # Return #
        return log_probs

    def sample(self,xs,cs,idx_type,idx_part):
        if self.flow_mode == 'global':
            # Select correct type and particle, use the global flow #
            x,m = self.pad_input(xs[idx_type][:,idx_part,:].unsqueeze(1),self.flow_features[idx_type])
            c = cs[idx_type][:,idx_part,:].unsqueeze(1)
            s = self.process_flow_block(
                x = x,
                c = c,
                m = m,
                flow_dict = self.flows,
                flow_features = self.global_flow_features,
                mode = 'sample',
            )
            return s * m
        if self.flow_mode == 'type':
            # Select correct type and article, and associated flow and features #
            s = self.process_flow_block(
                x = xs[idx_type][:,idx_part,:].unsqueeze(1),
                c = cs[idx_type][:,idx_part,:].unsqueeze(1),
                m = None,
                flow_dict = self.flows[idx_type],
                flow_features = self.flow_features[idx_type],
                mode = 'sample',
            )
            return s

class Flow(nn.Module):
    def __init__(
        self,
        d_model,
        d_condition,
        flow_cls,
        flow_args,
    ):
        super().__init__()

        # Save arguments #
        self.d_model = d_model
        self.d_condition = d_condition
        self.flow_cls = flow_cls
        self.flow_args = flow_args

        # Make flow #
        self.flow = flow_cls(
            features = d_model,
            context = d_condition,
            **flow_args
        )

    def forward(self,x,c):
        assert x.dim() == 2, f'x must be a 2D tensor, got {x.dim()} with shape {x.shape}'
        assert c.dim() == 2, f'x must be a 2D tensor, got {c.dim()} woht shape {c.shape}'
        assert x.shape[-1] == self.d_model,     f'x feature dim {x.shape[-1]} != d_model {self.d_model}'
        assert c.shape[-1] == self.d_condition, f'c feature dim {c.shape[-1]} != d_model {self.d_condition}'
        return self.flow(c).log_prob(x)

    def sample(self,c,N):
        assert c.dim() == 2, f'x must be a 2D tensor, got {c.dim()}'
        assert c.shape[-1] == self.d_condition, f'c feature dim {c.shape[-1]} != d_model {self.d_condition}'
        return self.flow(c).sample((N,))

