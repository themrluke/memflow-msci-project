from itertools import chain

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import lightning as L

import zuko
from zuko.distributions import BoxUniform
from zuko.distributions import DiagNormal

from memflow.models.utils import lowercase_recursive

class TransferFlow(L.LightningModule):
    def __init__(
        self,
        embed_dims,
        embed_act,
        n_hard_particles_per_type,
        hard_particle_type_names,
        hard_input_features_per_type,
        n_reco_particles_per_type,
        reco_particle_type_names,
        reco_input_features_per_type,
        flow_input_features,
        reco_mask_attn,
        hard_mask_attn,
        flow_mode,
        dropout = 0.,
        transformer_args = {},
        flow_common_args = {},
        flow_classes = {},
        flow_specific_args = {},
        onehot_encoding = False,
        process_names = None,
        optimizer = None,
        scheduler_config = None,
    ):
        super().__init__()

        # Public attributes #
        self.dropout = dropout
        self.embed_dims = embed_dims
        if isinstance(self.embed_dims,int):
            self.embed_dims = [self.embed_dims]
        self.embed_dim = self.embed_dims[-1]
        self.embed_act = embed_act

        self.n_hard_particles_per_type = n_hard_particles_per_type
        self.hard_particle_type_names = lowercase_recursive(hard_particle_type_names)
        self.hard_input_features_per_type = lowercase_recursive(hard_input_features_per_type)

        self.n_reco_particles_per_type = n_reco_particles_per_type
        self.reco_particle_type_names = lowercase_recursive(reco_particle_type_names)
        self.reco_input_features_per_type = lowercase_recursive(reco_input_features_per_type)

        self.flow_input_features = lowercase_recursive(flow_input_features)
        self.flow_mode = lowercase_recursive(flow_mode)
        self.flow_common_args = flow_common_args
        self.flow_classes = flow_classes
        self.flow_specific_args = flow_specific_args
        assert self.flow_mode in ['type','particle','global']

        if reco_mask_attn is None:
            self.reco_mask_attn = None
            print ('No reco attention mask provided, will use the exist mask for the attention')
        else:
            self.reco_mask_attn = torch.cat((torch.tensor([True]),reco_mask_attn),dim=0) # Adding True at index=0 for null token
        if hard_mask_attn is None:
            print ('No hard attention mask provided, will use the exist mask for the attention')
        self.hard_mask_attn  = hard_mask_attn

        self.onehot_encoding = onehot_encoding
        self.process_names = process_names

        # Safety checks #
        assert len(n_reco_particles_per_type) == len(reco_input_features_per_type), f'{len(n_reco_particles_per_type)} sets of reco particles but got {len(reco_input_features_per_type)} sets of input features'
        assert len(n_hard_particles_per_type) == len(hard_input_features_per_type), f'{len(n_hard_particles_per_type)} sets of hard particles but got {len(hard_input_features_per_type)} sets of input features'
        assert len(flow_input_features) == len(reco_input_features_per_type), f'Number of reco features ({len(reco_input_features_per_type)}) != number of flow features ({len(flow_input_features)})'

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
        self.hard_embeddings  = self.make_embeddings(self.hard_input_features_per_type)
        self.reco_embeddings = self.make_embeddings(self.reco_input_features_per_type,onehot_dim)

        # Define transformer #
        if 'd_model' in transformer_args.keys():
            print (f'Transformer args: will override `d_model` to {self.embed_dim}')
        transformer_args['d_model'] = self.embed_dim
        if 'dropout' not in transformer_args.keys():
            transformer_args['dropout'] = self.dropout
        self.transformer = nn.Transformer(**transformer_args,batch_first=True)
        self.tgt_mask = self.transformer.generate_square_subsequent_mask(sum(self.n_reco_particles_per_type)+1)

        # Define flows #
        if 'context' in self.flow_common_args:
            print (f'Flow args: will override `context` depending on the flow features for each object')
            del self.flow_common_args['context']
        if 'features' in self.flow_common_args:
            print (f'Flow args: will override `features` to 1, as our model has 1D flows')
        self.flow_common_args['features'] = 1

        self.flows = nn.ModuleList()
        self.flow_indices = []
        self.global_flow_features = []
        for i,(n,reco_features,flow_features) in enumerate(zip(self.n_reco_particles_per_type,self.reco_input_features_per_type,self.flow_input_features)):
            assert len(set(flow_features).intersection(set(reco_features))) == len(flow_features), f'Not all flow features {flow_features} found in reco_features for particle set #{i} ({reco_features})'
            indices = [reco_features.index(feature) for feature in flow_features]
            self.flow_indices.append(indices)
            self.global_flow_features.extend([feat for feat in flow_features if feat not in self.global_flow_features])
            if self.flow_mode == 'global':
                continue
            if self.flow_mode == 'type':
                # we use a single flow for all the particles of that type
                n_flows = 1
            if self.flow_mode == 'particle':
                # If mode is particles, we use a flow for each particles
                n_flows = n
            self.flows.append(
                nn.ModuleList(
                    [
                        self.make_flows(flow_features)
                        for _ in range(n_flows)
                    ]
                )
            )
        if self.flow_mode == 'global':
            self.flows.append(self.make_flows(self.global_flow_features))

    def make_embeddings(self,input_features,onehot_dim=0):
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
                            in_features = len(features)+onehot_dim if i==0 else self.embed_dims[i-1],
                            out_features = self.embed_dims[i],
                        )
                    )
                    if self.embed_act is not None and i < len(self.embed_dims) - 1:
                        layers.append(self.embed_act())
                    if self.dropout != 0.:
                        layers.append(nn.Dropout(self.dropout))
                embedding = nn.Sequential(*layers)
                feature_embeddings[features] = embedding
            else:
                embedding = feature_embeddings[features]
            embeddings.append(embedding)
        return embeddings

    def make_flows(self,features):
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
                context = self.embed_dim + context_dim,
                **self.flow_common_args,
                **add_args,
            )
        return flows


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
        log_probs, mask, weights = self(batch)
        assert log_probs.shape == mask.shape, f'Log prob has shape {log_probs.shape}, and mask {mask.shape}'
        assert log_probs.shape == weights.shape, f'Log prob has shape {log_probs.shape}, and weights {weights.shape}'
        if torch.isnan(log_probs).sum()>0:
            where_nan = torch.where(torch.isnan(log_probs))
            mask_nan = mask[where_nan]>0
            where_nan = [coord[mask_nan] for coord in where_nan]
            if where_nan[0].nelement() > 0:
                #raise RuntimeError(f'nans at coordinates {where_nan}')
                print (f'nans at coordinates {where_nan}')
        if torch.isinf(log_probs).sum()>0:
            where_inf = torch.where(torch.isinf(log_probs))
            mask_inf = mask[where_inf]>0
            where_inf = [coord[mask_inf] for coord in where_inf]
            if where_inf[0].nelement() > 0:
                #raise RuntimeError(f'infs at coordinates {where_inf}')
                print (f'infs at coordinates {where_inf}')
        log_probs = torch.nan_to_num(log_probs,nan=0.0,posinf=0.,neginf=0.)
        # Log per object loss #
        idx = 0
        for i,n in enumerate(self.n_reco_particles_per_type):
            for j in range(n):
                if mask[:,idx].sum() != 0:
                    # average over existing particles
                    self.log(
                        f"{prefix}/loss_{self.reco_particle_type_names[i]}_{j}",
                        (log_probs[:,idx] * mask[:,idx]).sum() / mask[:,idx].sum(),
                        # Only log the log_prob for existing objects
                        # averaged over number of existing objects
                        prog_bar=False,
                    )
                idx += 1
        # Loss per process #
        if 'process' in batch.keys():
            for idx in torch.unique(batch['process']).sort()[0]:
                process_idx = torch.where(batch['process'] == idx)[0]
                process_name = self.process_names[idx] if self.process_names is not None else str(idx.item())
                self.log(
                    f"{prefix}/loss_process_{process_name}",
                    ((log_probs[process_idx,:] * mask[process_idx,:]).sum(dim=-1) / mask[process_idx,:].sum(dim=-1)).mean(),
                    prog_bar=False,
                )

        # Get total loss, weighted and averaged over existing objects and events #
        log_prob_tot = (
            (
                log_probs * mask * weights          # log prob masked and weighted
            ).sum(dim=1) / mask.sum(dim=1)          # averaged over number of existing particles
        ).mean()                                    # averaged on all events
        self.log(f"{prefix}/loss_tot", log_prob_tot, prog_bar=True)
        # Return #
        return log_prob_tot

    def training_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'val')

    def conditioning(self,hard_data,hard_mask_exist,reco_data,reco_mask_exist):
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
        hard_mask_exist = torch.cat(hard_mask_exist,dim=1)
        hard_data = torch.cat(
            [
                self.hard_embeddings[i](hard_data[i])
                for i in range(len(self.hard_embeddings))
            ],
            dim = 1
        ) * hard_mask_exist[...,None]
        reco_mask_exist_null = torch.cat(reco_mask_exist_null,dim=1)
        reco_data_null = torch.cat(
            [
                self.reco_embeddings[i](reco_data_null[i])
                for i in range(len(self.reco_embeddings))
            ],
            dim = 1
        ) * reco_mask_exist_null[...,None]

        # Expand attention mask #
        # Need to turn 0->1 when particle exists #
        if self.hard_mask_attn is None:
            hard_mask_attn = hard_mask_exist
        else:
            hard_mask_attn  = torch.logical_or(
                self.hard_mask_attn.to(hard_mask_exist.device),
                hard_mask_exist,
            )
        if self.reco_mask_attn is None:
            reco_mask_attn = reco_mask_exist_null
        else:
            reco_mask_attn = torch.logical_or(
                self.reco_mask_attn.to(reco_mask_exist_null.device),
                reco_mask_exist_null,
            )
        # -> Make sure that particles we want in the attention are considered even if missing
        # (in which case the default values are set in the dataset class, no need to re default them)
        # Turn them into boolean arrays #
        if hard_mask_attn.dtype != torch.bool:
            hard_mask_attn = hard_mask_attn > 0
        if reco_mask_attn.dtype != torch.bool:
            reco_mask_attn = reco_mask_attn > 0
        # replace True->0, False->-inf
        # To have same dtype as tgt_mask
        hard_mask_attn = torch.zeros_like(hard_mask_attn).to(torch.float32).masked_fill(~hard_mask_attn,float("-inf"))
        reco_mask_attn = torch.zeros_like(reco_mask_attn).to(torch.float32).masked_fill(~reco_mask_attn,float("-inf"))

        # Transformer processing #
        condition = self.transformer(
            src = hard_data,                                # encoder (hard) input
            tgt = reco_data_null,                           # decorder (reco) input
            tgt_mask = self.tgt_mask.to(hard_data.device),  # triangular (causality) mask
            src_key_padding_mask = hard_mask_attn,          # encoder (hard) mask
            memory_key_padding_mask = hard_mask_attn,       # encoder output / memory mask
            tgt_key_padding_mask = reco_mask_attn,          # decoder (reco) mask
        )

        # Split condition per particle type to match reco_data segmentation #
        slices = np.r_[0,np.array(self.n_reco_particles_per_type).cumsum()]
        conditions = [condition[:,ni:nf,:] for ni,nf in zip(slices[:-1],slices[1:])]

        return conditions

    def process_flow(self,reco_data,conditions,mode,N=None):
        assert mode in ['log_prob','sample']

        # Init full list of log_probs and samples #
        log_probs_particles = []
        samples = []

        # Loop over global features #
        if self.flow_mode == 'global':
            condition_decoder = torch.cat(conditions,dim=1)
            samples_features = []
            global_feature_values = []
            global_feature_mask   = []
            for flow_feature in self.global_flow_features:
                # Combine features of all particles #
                feature_values = []
                feature_mask = []
                for i,(n,flow_features,flow_indices) in enumerate(zip(self.n_reco_particles_per_type,self.flow_input_features,self.flow_indices)):
                    if flow_feature in flow_features:
                        values = reco_data[i][:,:,flow_indices[flow_features.index(flow_feature)]]
                        feature_values.append(values)
                        feature_mask.append(torch.ones_like(values))
                    else:
                        feature_values.append(torch.zeros((reco_data[i].shape[0],n)).to(reco_data[i].device))
                        feature_mask.append(torch.zeros((reco_data[i].shape[0],n)).to(reco_data[i].device))
                global_feature_values.append(torch.cat(feature_values,dim=1).unsqueeze(-1))
                global_feature_mask.append(torch.cat(feature_mask,dim=1).unsqueeze(-1))
            global_feature_values = torch.cat(global_feature_values,dim=-1)
            global_feature_mask = torch.cat(global_feature_mask,dim=-1)

            for k,flow_feature in enumerate(self.global_flow_features):
                # Get flow #
                flow_dict = self.flows[0]
                if flow_feature not in flow_dict.keys():
                    raise RuntimeError(f'For block {i}, feature {flow_feature} not in flow keys {flow_dict.keys()}')
                flow = flow_dict[flow_feature]
                # Make condition for feature k (decoder + other features)
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
                condition_features_select = torch.cat(
                    [
                        global_feature_values[...,self.global_flow_features.index(feat)].unsqueeze(-1)
                        for feat in features_for_condition if feat in self.global_flow_features
                    ],
                    dim = -1,
                )
                condition = torch.cat(
                    [
                        condition_decoder,              # from transformer decoder
                        condition_features_select,  # from the other features (latent or reco)
                    ],
                    dim = -1
                )
                if mode == 'log_prob':
                    # Obtain log prob #
                    log_probs_features = flow(condition).log_prob(global_feature_values[...,k].unsqueeze(-1))
                    # Apply mask to have log_prob of missing features (padded with zero above) at zero
                    log_probs_features *= global_feature_mask[...,k]
                    # Record #
                    log_probs_particles.append(log_probs_features)
                if mode == 'sample':
                    samples_features.append(flow(condition).sample((N,)))

                # Update the condition feature by the latent (transformed) feature
                global_feature_values[...,k] = flow(condition).transform(global_feature_values[...,k].unsqueeze(-1)).squeeze(-1)
            if mode == 'log_prob':
                # Reshape #
                # log_probs_particles is list of log probs with shape [#event,#particles]
                # We want a list of log probs per particle, so [#event,1],
                # with log_prob(1 particle) = log_prob(pt) + log_prob(eta) + ...
                # (this is because later we will apply the exist mask, which is defined at particle level)
                log_probs_particles = sum(log_probs_particles) # sum per feature
                log_probs_particles = torch.split(log_probs_particles,split_size_or_sections=1,dim=1) # split per particle
            if mode == 'sample':
                # Reshape #
                # Similarly, got list of [#sample,#events,#particles,1=#features],
                # need to turn into list of samples per particle of same type [#sample,#events,#particles,#features]
                samples_all = torch.cat(samples_features,dim=-1) # concat per features [#sample,#events,#particles,#features]
                idxs = np.r_[0,np.cumsum(self.n_reco_particles_per_type)]
                for idx_i,idx_f in zip(idxs[:-1],idxs[1:]):
                    # Select samples corresponding to specific type (between idx_i and idx_f)
                    samples_type = samples_all[:,:,idx_i:idx_f:]
                    # Remove features that are not present for that sample #
                    idx_to_select = torch.where(global_feature_mask[0,idx_i]>0)[0]
                    samples.append(samples_type[:,:,:,idx_to_select])
        else:
            # Loop over type of particles #
            for i,(n,flow_features,flow_indices) in enumerate(zip(self.n_reco_particles_per_type,self.flow_input_features,self.flow_indices)):
                if mode == 'log_prob':
                    log_probs_features = torch.zeros((reco_data[i].shape[0],n)).to(reco_data[i].device)
                if mode == 'sample':
                    particles = []
                # For type mode, we use the condition of all particles in that type #
                if self.flow_mode == 'type':
                    condition_features = reco_data[i][:,:,flow_indices]
                    condition_decoder = conditions[i]
                    # Note : for the type mode, we apply it for all particles -> no need for particle loop
                    # Loop over features #
                    for k,(flow_feature,flow_index) in enumerate(zip(flow_features,flow_indices)):
                        # Get corresponding flow #
                        flow_dict = self.flows[i][0] # one flow for all particle per feature, so only one item in list
                        if flow_feature not in flow_dict.keys():
                            raise RuntimeError(f'For block {i}, feature {flow_feature} not in flow keys {flow_dict.keys()}')
                        flow = flow_dict[flow_feature]
                        # Make condition for feature k (decoder + other features)
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
                        condition_features_select = torch.cat(
                            [
                                condition_features[...,flow_features.index(feat)].unsqueeze(-1)
                                for feat in features_for_condition if feat in flow_features
                            ],
                            dim = -1,
                        )
                        condition = torch.cat(
                            [
                                condition_decoder,              # from transformer decoder
                                condition_features_select,  # from the other features (latent or reco)
                            ],
                            dim = -1
                        )
                        # Record #
                        if mode == 'log_prob':
                            # Add log prob for that feature #
                            log_probs_features += flow(condition).log_prob(reco_data[i][...,k].unsqueeze(-1))
                        if mode == 'sample':
                            particles.append(flow(condition).sample((N,)))
                        # Update the condition feature by the latent (transformed) feature
                        condition_features[...,k] = flow(condition).transform(condition_features[...,k].unsqueeze(-1)).squeeze(-1)
                if self.flow_mode == 'particle':
                    # Loop over particles #
                    particles_type = []
                    for j in range(n):
                        particle = []
                        # Get flow of particle j #
                        flow_dict = self.flows[i][j]
                        # Get condition #
                        condition_features = reco_data[i][:,j,flow_indices]
                        condition_decoder = conditions[i][:,j,:]
                        # Loop over features #
                        for k,(flow_feature,flow_index) in enumerate(zip(flow_features,flow_indices)):
                            # Get corresponding flow #
                            if flow_feature not in flow_dict.keys():
                                raise RuntimeError(f'For block {i} and particle {j}, feature {flow_feature} not in flow keys {flow_dict.keys()}')
                            flow = flow_dict[flow_feature]
                            # Make condition for feature k (decoder + other features)
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
                            condition_features_select = torch.cat(
                                [
                                    condition_features[...,flow_features.index(feat)].unsqueeze(-1)
                                    for feat in features_for_condition if feat in flow_features
                                ],
                                dim = -1,
                            )
                            condition = torch.cat(
                                [
                                    condition_decoder,              # from transformer decoder
                                    condition_features_select,  # from the other features (latent or reco)
                                ],
                                dim = -1
                            )
                            # Record #
                            if mode == 'log_prob':
                                # Add log prob for that feature #
                                log_probs_features[:,j] += flow(condition).log_prob(reco_data[i][:,j,k].unsqueeze(-1))
                            if mode == 'sample':
                                particle.append(flow(condition).sample((N,)))

                            # Add log prob for that feature #
                            # Update the condition feature by the latent (transformed) feature
                            condition_features[...,k] = flow(condition).transform(condition_features[...,k].unsqueeze(-1)).squeeze(-1)
                        if mode == 'sample':
                            particles_type.append(torch.cat(particle,dim=-1).unsqueeze(-2))
                    # End of particles per type loop, concat samples #
                    if mode == 'sample':
                        particles.append(torch.cat(particles_type,dim=-2))
                # End of feature (and particle) loop
                # Record objects for type i
                if mode == 'log_prob':
                    # Add to list of per-particle log probs #
                    log_probs_particles.append(log_probs_features)
                if mode == 'sample':
                    # particles is list of particles for given type #
                    samples.append(torch.cat(particles,dim=-1))

        if mode == 'log_prob':
            return - torch.cat(log_probs_particles,dim=1)
        if mode == 'sample':
            return samples

    def forward(self,batch):
        # Extract different components #
        hard_data  = batch['hard']['data']
        hard_mask_exist = batch['hard']['mask']
        hard_weights = batch['hard']['weights']
        reco_data = batch['reco']['data']
        reco_mask_exist = batch['reco']['mask']
        reco_weights = batch['reco']['weights']

        # Safety checks #
        assert len(hard_data) == len(self.hard_embeddings), f'{len(hard_data)} hard objects but {len(self.hard_embeddings)} hard embeddings'
        assert len(hard_mask_exist) == len(self.hard_embeddings), f'{len(hard_mask_exist)} hard objects but {len(self.hard_embeddings)} hard embeddings'
        assert len(reco_data) == len(self.reco_embeddings), f'{len(reco_data)} reco objects but {len(self.reco_embeddings)} reco embeddings'
        assert len(reco_mask_exist) == len(self.reco_embeddings), f'{len(reco_mask_exist)} reco objects but {len(self.reco_embeddings)} reco embeddings'

        # Obtain condition #
        conditions = self.conditioning(hard_data,hard_mask_exist,reco_data,reco_mask_exist)

        # Obtain log probs #
        log_probs_particles = self.process_flow(reco_data,conditions,mode='log_prob')

        # Return log_prob, mask and weights #
        return log_probs_particles, torch.cat(reco_mask_exist,dim=1), torch.cat(reco_weights,dim=1)

    def sample(self,hard_data,hard_mask_exist,reco_data,reco_mask_exist,N):
        # Obtain condition #
        conditions = self.conditioning(hard_data,hard_mask_exist,reco_data,reco_mask_exist)

        # Obtain flows #
        samples = self.process_flow(reco_data,conditions,mode='sample',N=N)

        # Return list of [S,N,P,F]
        # S = number of samples
        # N = number of events in the batch
        # P = number of particles in the batch
        # F = number of features
        return samples

