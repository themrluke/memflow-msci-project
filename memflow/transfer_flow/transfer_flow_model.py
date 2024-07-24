from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import lightning as L

import zuko
from zuko.distributions import BoxUniform
from zuko.distributions import DiagNormal

class TransferFlow(L.LightningModule):
    def __init__(
        self,
        embed_dim,
        embed_act,
        n_reco_particles_per_type,
        n_gen_particles_per_type,
        reco_input_features,
        gen_input_features,
        flow_input_features,
        reco_mask_corr,
        gen_mask_corr,
        transformer_args = {},
        flow_args = {},
        onehot_encoding = False,
        optimizer = None,
        scheduler_config = None,
    ):
        super().__init__()

        # Public attributes #
        self.embed_dim = embed_dim
        self.embed_act = embed_act
        self.n_reco_particles_per_type = n_reco_particles_per_type
        self.n_gen_particles_per_type = n_gen_particles_per_type
        self.reco_input_features = reco_input_features
        self.gen_input_features = gen_input_features
        self.flow_input_features = flow_input_features
        self.reco_mask_corr = torch.cat((torch.tensor([True]),reco_mask_corr),dim=0) # Adding True at index=0 for null token
        self.gen_mask_corr  = gen_mask_corr
        self.onehot_encoding = onehot_encoding

        # Safety checks #
        assert len(n_reco_particles_per_type) == len(reco_input_features), f'{len(n_reco_particles_per_type)} sets of reco particles but got {len(reco_input_features)} sets of input features'
        assert len(n_gen_particles_per_type) == len(gen_input_features), f'{len(n_gen_particles_per_type)} sets of gen particles but got {len(gen_input_features)} sets of input features'
        assert len(flow_input_features) == len(reco_input_features), f'Number of reco features ({len(reco_input_features)}) != number of flow features ({len(flow_input_features)})'

        # Private attributes #
        self._optimizer = optimizer
        self._scheduler_config = scheduler_config

        # Make onehot encoded vectors #
        if self.onehot_encoding:
            onehot_dim = sum(self.n_reco_particles_per_type)+1
            onehot_matrix = F.one_hot(torch.arange(onehot_dim))
            reco_indices = torch.tensor(self.n_reco_particles_per_type)
            reco_indices[0] += 1 # add the null
            reco_indices = torch.cat([torch.tensor([0]),reco_indices.cumsum(dim=0)],dim=0)
            self.onehot_tensors = [onehot_matrix[:,ia:ib].T for ia,ib in zip(reco_indices[:-1],reco_indices[1:]) ]
        else:
            onehot_dim = 0

        # Make embedding layers #
        self.gen_embeddings  = self.make_embeddings(self.gen_input_features)
        self.reco_embeddings = self.make_embeddings(self.reco_input_features,onehot_dim)

        # Define transformer #
        if 'd_model' in transformer_args:
            print (f'Transformer args: will override `d_model` to {self.embed_dim}')
        transformer_args['d_model'] = self.embed_dim
        self.transformer = nn.Transformer(**transformer_args,batch_first=True)
        self.tgt_mask = self.transformer.generate_square_subsequent_mask(sum(self.n_reco_particles_per_type)+1)

        # Define flows #
        if 'context' in flow_args:
            print (f'Flow args: will override `context` to {self.embed_dim}')
        flow_args['context'] = self.embed_dim
        if 'features' in flow_args:
            print (f'Flow args: will override `features` depending on the number of features of each object')
        self.flows = nn.ModuleList()
        self.flow_indices = []
        for i,(n,reco_features,flow_features) in enumerate(zip(self.n_reco_particles_per_type,self.reco_input_features,self.flow_input_features)):
            assert len(set(flow_features).intersection(set(reco_features))) == len(flow_features), f'Not all flow features {flow_features} found in reco_features for particle set #{i}'
            indices = [reco_features.index(feature) for feature in flow_features]
            self.flow_indices.append(indices)
            for j in range(n):
                flow_args['features'] = len(indices)
                self.flows.append(zuko.flows.NSF(**flow_args))

    def make_embeddings(self,input_features,onehot_dim=0):
        feature_embeddings = {}
        embeddings = nn.ModuleList()
        # Make sure inputs with same features are processed through same embedding #
        for features in input_features:
            if not isinstance(features,tuple):
                features = tuple(features)
            if features not in feature_embeddings.keys():
                layers = [
                    nn.Linear(
                        in_features = len(features)+onehot_dim,
                        out_features = self.embed_dim,
                    )
                ]
                if self.embed_act is not None:
                    layers.append(self.embed_act())
                embedding = nn.Sequential(*layers)
                feature_embeddings[features] = embedding
            else:
                embedding = feature_embeddings[features]
            embeddings.append(embedding)
        return embeddings

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
        # Get log-prob loss #
        loss = self(batch)
        # Record #
        self.log(f"{prefix}/loss", loss, prog_bar=True)
        # Return #
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'val')

#    def predict_step(self, batch, batch_idx, **kwargs):
#        inputs = batch[0]
#        ancillaries = batch[3:]
#        outputs = self.forward(inputs,*ancillaries,**kwargs)
#        return outputs

    def conditioning(self,gen_data,gen_mask_exist,reco_data,reco_mask_exist):
        # Add null token to first reco object #
        null_token = torch.ones((reco_data[0].shape[0],1,reco_data[0].shape[2])) * -1
        reco_data_null = [
            torch.cat(
                (
                    null_token.to(reco_data[0].device),
                    reco_data[0],
                ),
                dim = 1, # along particle axis
            )
        ] + [data[:] for data in reco_data[1:]]
        reco_mask_exist_null = [
            torch.cat(
                (
                    torch.full((reco_mask_exist[0].shape[0],1),fill_value=True).to(reco_mask_exist[0].device),
                    reco_mask_exist[0],
                ),
                dim = 1,
            )

        ] + [mask[:] for mask in reco_mask_exist[1:]]

        # Apply onehot encoding #
        if self.onehot_encoding:
            reco_data_null = [
                torch.cat(
                    [
                        data,
                        onehot.repeat(data.shape[0],1,1).to(data.device),
                    ],
                    dim = 2,
                )
                for data,onehot in zip(reco_data_null,self.onehot_tensors)
            ]

        # Apply embeddings and concat along particle axis #
        gen_data = torch.cat(
            [
                self.gen_embeddings[i](gen_data[i])
                for i in range(len(self.gen_embeddings))
            ],
            dim = 1
        )
        gen_mask_exist = torch.cat(gen_mask_exist,dim=1)
        reco_data_null = torch.cat(
            [
                self.reco_embeddings[i](reco_data_null[i])
                for i in range(len(self.reco_embeddings))
            ],
            dim = 1
        )
        reco_mask_exist_null = torch.cat(reco_mask_exist_null,dim=1)

        # Expand correlation mask #
        # Need to turn 0->1 when particle exists #
        gen_mask_corr  = torch.logical_or(
            self.gen_mask_corr.to(gen_mask_exist.device),
            gen_mask_exist,
        )
        reco_mask_corr = torch.logical_or(
            self.reco_mask_corr.to(reco_mask_exist_null.device),
            reco_mask_exist_null,
        )
        # -> Make sure that particles we want in the attention are considered even if missing
        # (in which case the default values are set in the dataset class, no need to re default them)

        # Transformer processing #
        condition = self.transformer(
            src = gen_data,
            tgt = reco_data_null,
            tgt_mask = self.tgt_mask.to(gen_data.device),
            src_key_padding_mask = (~gen_mask_corr).to(self.tgt_mask.dtype),
            tgt_key_padding_mask = (~reco_mask_corr).to(self.tgt_mask.dtype),
        )

        return condition

    def forward(self,batch):
        # Extract different components #
        gen_data  = batch['gen']['data']
        gen_mask_exist = batch['gen']['mask']
        reco_data = batch['reco']['data']
        reco_mask_exist = batch['reco']['mask']

        # Safety checks #
        assert len(gen_data) == len(self.gen_embeddings), f'{len(gen_data)} gen objects but {len(self.gen_embeddings)} gen embeddings'
        assert len(gen_mask_exist) == len(self.gen_embeddings), f'{len(gen_mask_exist)} gen objects but {len(self.gen_embeddings)} gen embeddings'
        assert len(reco_data) == len(self.reco_embeddings), f'{len(reco_data)} reco objects but {len(self.reco_embeddings)} reco embeddings'
        assert len(reco_mask_exist) == len(self.reco_embeddings), f'{len(reco_mask_exist)} reco objects but {len(self.reco_embeddings)} reco embeddings'

        # Obtain condition #
        condition = self.conditioning(gen_data,gen_mask_exist,reco_data,reco_mask_exist)

        # Obtain all the flow log probabilities #
        log_probs = []
        idx = 0
        for i,(n,indices) in enumerate(zip(self.n_reco_particles_per_type,self.flow_indices)):
            for j in range(n):
                log_probs.append(
                    self.flows[idx](
                        condition[:,idx:idx+1,:]        # Condition on the flow on idx-th condition
                    ).log_prob(
                        reco_data[i][:,j:j+1,indices]   # Apply on ith object (only take flow features)
                    ) * reco_mask_exist[i][:,j:j+1]     # Apply mask
                )
                idx += 1

        # Sum object wise and average event wise #
        return -sum(log_probs).mean()

    def sample(self,gen_data,gen_mask_exist,reco_data,reco_mask_exist,N):
        # Obtain condition #
        condition = self.conditioning(gen_data,gen_mask_exist,reco_data,reco_mask_exist)

        # Sample for each particle #
        samples = []
        idx = 0
        for i,(n,indices) in enumerate(zip(self.n_reco_particles_per_type,self.flow_indices)):
            for j in range(n):
                #if reco_mask_exist[i][:,j]:
                samples.append(
                    self.flows[idx](
                        condition[:,idx:idx+1,:]
                    ).sample((N,))
                )
                idx += 1
        # Return list of [S,N,P,F]
        # S = number of samples
        # N = number of events in the batch
        # P = number of particles in the batch
        # F = number of features
        return samples
