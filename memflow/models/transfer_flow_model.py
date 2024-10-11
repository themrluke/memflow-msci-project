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

from memflow.transfer_flow.periodicNSF_gaussian import NCSF_gaussian
from memflow.transfer_flow.NCSF_custom import UniformNCSF, UniformNSF
from memflow.transfer_flow.utils import lowercase_recursive

class TransferFlow(L.LightningModule):
    def __init__(
        self,
        embed_dim,
        embed_act,
        n_gen_particles_per_type,
        gen_particle_type_names,
        gen_input_features_per_type,
        n_reco_particles_per_type,
        reco_particle_type_names,
        reco_input_features_per_type,
        flow_input_features,
        autoregressive_mode,
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

        self.n_gen_particles_per_type = n_gen_particles_per_type
        self.gen_particle_type_names = lowercase_recursive(gen_particle_type_names)
        self.gen_input_features_per_type = lowercase_recursive(gen_input_features_per_type)

        self.n_reco_particles_per_type = n_reco_particles_per_type
        self.reco_particle_type_names = lowercase_recursive(reco_particle_type_names)
        self.reco_input_features_per_type = lowercase_recursive(reco_input_features_per_type)

        self.flow_input_features = lowercase_recursive(flow_input_features)
        self.autoregressive_mode = lowercase_recursive(autoregressive_mode)
        assert self.autoregressive_mode in ['type','particle','global']

        self.reco_mask_corr = torch.cat((torch.tensor([True]),reco_mask_corr),dim=0) # Adding True at index=0 for null token
        self.gen_mask_corr  = gen_mask_corr
        self.onehot_encoding = onehot_encoding

        # Safety checks #
        assert len(n_reco_particles_per_type) == len(reco_input_features_per_type), f'{len(n_reco_particles_per_type)} sets of reco particles but got {len(reco_input_features_per_type)} sets of input features'
        assert len(n_gen_particles_per_type) == len(gen_input_features_per_type), f'{len(n_gen_particles_per_type)} sets of gen particles but got {len(gen_input_features_per_type)} sets of input features'
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
        self.gen_embeddings  = self.make_embeddings(self.gen_input_features_per_type)
        self.reco_embeddings = self.make_embeddings(self.reco_input_features_per_type,onehot_dim)

        # Define transformer #
        if 'd_model' in transformer_args:
            print (f'Transformer args: will override `d_model` to {self.embed_dim}')
        transformer_args['d_model'] = self.embed_dim
        self.transformer = nn.Transformer(**transformer_args,batch_first=True)
        self.tgt_mask = self.transformer.generate_square_subsequent_mask(sum(self.n_reco_particles_per_type)+1)

        # Define flows #
        if 'context' in flow_args:
            print (f'Flow args: will override `context` depending on the flow features for each object')
        if 'features' in flow_args:
            print (f'Flow args: will override `features` to 1, as our model has 1D flows')
        flow_args['features'] = 1

        self.flows = nn.ModuleList()
        self.flow_indices = []
        self.global_flow_features = []
        for i,(n,reco_features,flow_features) in enumerate(zip(self.n_reco_particles_per_type,self.reco_input_features_per_type,self.flow_input_features)):
            assert len(set(flow_features).intersection(set(reco_features))) == len(flow_features), f'Not all flow features {flow_features} found in reco_features for particle set #{i} ({reco_features})'
            indices = [reco_features.index(feature) for feature in flow_features]
            self.flow_indices.append(indices)
            self.global_flow_features.extend([feat for feat in flow_features if feat not in self.global_flow_features])
            if self.autoregressive_mode == 'global':
                continue
            if self.autoregressive_mode == 'type':
                # we use a single flow for all the particles of that type
                n_flows = 1
            if self.autoregressive_mode == 'particle':
                # If mode is particles, we use a flow for each particles
                n_flows = n
            self.flows.append(
                nn.ModuleList(
                    [
                        self.make_flows(flow_features,flow_args,self.embed_dim)
                        for _ in range(n_flows)
                    ]
                )
            )
        if self.autoregressive_mode == 'global':
            self.flows.append(self.make_flows(self.global_flow_features,flow_args,self.embed_dim))

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

    def make_flows(self,features,flow_args,embed_dim):
        flows = nn.ModuleDict()
        add_args = {}
        for feature in features:
            if feature == 'pt':
                flow_cls = zuko.flows.NSF
                context_dim = sum([other_feat == feat for other_feat in ['eta','phi'] for feat in features])
                # include reco phi + reco eta (if present)
            elif feature == 'eta':
                flow_cls = UniformNSF
                add_args['bound'] = 1.
                #flow_cls = zuko.flows.NSF
                context_dim = sum([other_feat == feat for other_feat in ['pt','phi'] for feat in features])
                # include latent pt + reco phi (if present)
            elif feature == 'phi':
                #flow_cls = NCSF_gaussian
                flow_cls = UniformNCSF
                add_args['bound'] = 1.
                context_dim = sum([other_feat == feat for other_feat in ['pt','eta'] for feat in features])
                # include latent pt + latent eta (if present)
            elif feature == 'mass' or feature == 'm':
                flow_cls = zuko.flows.NSF
                context_dim = sum([other_feat == feat for other_feat in ['pt','eta','phi'] for feat in features])
                # include latent pt+eta+phi (if present)
            else:
                raise NotImplementedError(f'This model does not include feature {feature}')
            flow_args['context'] = embed_dim + context_dim
            flows[feature] = flow_cls(**flow_args,**add_args)
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
        # Record per object loss #
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
        # Get total loss, weighted and averaged over existing objects and events #
        log_prob_tot = (
            (
                torch.nan_to_num(
                    (log_probs * mask * weights),   # log prob masked and weighted
                    nan = 0.
                )
            ).sum(dim=1) / mask.sum(dim=1)          # averaged over number of existing particles
        ).mean()                                    # averaged on all events
        self.log(f"{prefix}/loss", log_prob_tot, prog_bar=True)
        # Return #
        return log_prob_tot

    def training_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'val')

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
        gen_mask_exist = torch.cat(gen_mask_exist,dim=1)
        gen_data = torch.cat(
            [
                self.gen_embeddings[i](gen_data[i])
                for i in range(len(self.gen_embeddings))
            ],
            dim = 1
        ) * gen_mask_exist[...,None]
        reco_mask_exist_null = torch.cat(reco_mask_exist_null,dim=1)
        reco_data_null = torch.cat(
            [
                self.reco_embeddings[i](reco_data_null[i])
                for i in range(len(self.reco_embeddings))
            ],
            dim = 1
        ) * reco_mask_exist_null[...,None]

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
        if self.autoregressive_mode == 'global':
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
                if self.autoregressive_mode == 'type':
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
                if self.autoregressive_mode == 'particle':
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
        gen_data  = batch['gen']['data']
        gen_mask_exist = batch['gen']['mask']
        gen_weights = batch['gen']['weights']
        reco_data = batch['reco']['data']
        reco_mask_exist = batch['reco']['mask']
        reco_weights = batch['reco']['weights']

        # Safety checks #
        assert len(gen_data) == len(self.gen_embeddings), f'{len(gen_data)} gen objects but {len(self.gen_embeddings)} gen embeddings'
        assert len(gen_mask_exist) == len(self.gen_embeddings), f'{len(gen_mask_exist)} gen objects but {len(self.gen_embeddings)} gen embeddings'
        assert len(reco_data) == len(self.reco_embeddings), f'{len(reco_data)} reco objects but {len(self.reco_embeddings)} reco embeddings'
        assert len(reco_mask_exist) == len(self.reco_embeddings), f'{len(reco_mask_exist)} reco objects but {len(self.reco_embeddings)} reco embeddings'

        # Obtain condition #
        conditions = self.conditioning(gen_data,gen_mask_exist,reco_data,reco_mask_exist)

        # Obtain log probs #
        log_probs_particles = self.process_flow(reco_data,conditions,mode='log_prob')

        # Return log_prob, mask and weights #
        return log_probs_particles, torch.cat(reco_mask_exist,dim=1), torch.cat(reco_weights,dim=1)

    def sample(self,gen_data,gen_mask_exist,reco_data,reco_mask_exist,N):
        # Obtain condition #
        conditions = self.conditioning(gen_data,gen_mask_exist,reco_data,reco_mask_exist)

        # Obtain flows #
        samples = self.process_flow(reco_data,conditions,mode='sample',N=N)

        # Return list of [S,N,P,F]
        # S = number of samples
        # N = number of events in the batch
        # P = number of particles in the batch
        # F = number of features
        return samples

