import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import math
import numpy as np
import torch.optim as optim
from typing import Optional, Dict, Any, List
from zuko.distributions import DiagNormal

from memflow.models.utils import lowercase_recursive



class TransferCFM(L.LightningModule):
    """
    Conditional Flow Matching model operating in a 'global' mode.
    Includes conditioning using a Transformer architecture.
    """
    def __init__(
        self,
        embed_dims, # Embedding dimensions for input features
        embed_act,  # Activation function for embeddings
        n_hard_particles_per_type,
        hard_particle_type_names,
        hard_input_features_per_type,
        n_reco_particles_per_type,
        reco_particle_type_names,
        reco_input_features_per_type,
        flow_input_features,    # Features used for flow matching
        reco_mask_attn,
        hard_mask_attn,
        dropout=0.0,
        process_names=None, # HERE, LOOK AT SHARED EVAL FUNC IN TRANSFER FLOW
        transformer_args={},
        sigma=0.1,
        optimizer=None,
        scheduler_config=None,
        onehot_encoding=False,
    ):
        super().__init__()
        if transformer_args is None:
            transformer_args = {}

        self.dropout = dropout
        self.embed_dims = embed_dims if isinstance(embed_dims, list) else [embed_dims]
        self.embed_dim = self.embed_dims[-1]
        self.embed_act = embed_act

        self.n_hard_particles_per_type = n_hard_particles_per_type
        self.hard_input_features_per_type = lowercase_recursive(hard_input_features_per_type)
        self.hard_particle_type_names = lowercase_recursive(hard_particle_type_names)

        self.n_reco_particles_per_type = n_reco_particles_per_type
        self.reco_input_features_per_type = lowercase_recursive(reco_input_features_per_type)
        self.reco_particle_type_names = lowercase_recursive(reco_particle_type_names)

        self.flow_input_features = lowercase_recursive(flow_input_features)
        self.len_flow_feats = max(len(flow_feats) for flow_feats in self.flow_input_features)
        self.onehot_encoding = onehot_encoding
        self.process_names = process_names

        self.sigma = sigma
        self._optimizer = optimizer
        self._scheduler_config = scheduler_config

        # Handle attention masks
        # If no attention mask provided, the existence masks are used: Model attends to all particles in event
        if reco_mask_attn is None:
            self.reco_mask_attn = None
            print("No reco attention mask provided; will use existence mask only.")
        else:
            # Add a single True for the null token
            self.reco_mask_attn = torch.cat((torch.tensor([True]), reco_mask_attn), dim=0)
        if hard_mask_attn is None:
            print("No hard attention mask provided; will use existence mask only.")
        self.hard_mask_attn = hard_mask_attn

        # Safety checks
        assert len(n_reco_particles_per_type) == len(reco_input_features_per_type), f'{len(n_reco_particles_per_type)} sets of reco particles but got {len(reco_input_features_per_type)} sets of input features'
        assert len(n_hard_particles_per_type) == len(hard_input_features_per_type), f'{len(n_hard_particles_per_type)} sets of hard particles but got {len(hard_input_features_per_type)} sets of input features'
        assert len(flow_input_features) == len(reco_input_features_per_type), f'Number of reco features ({len(reco_input_features_per_type)}) != number of flow features ({len(flow_input_features)})'

        # Make onehot encoded vectors #
        if self.onehot_encoding:
            onehot_dim = sum(self.n_reco_particles_per_type) + 1 # Total num of reco particles for event (+1 null particle) 
            onehot_matrix = F.one_hot(torch.arange(onehot_dim)) # A one-hot vector for each particle stored in matrix
            reco_indices = torch.tensor(self.n_reco_particles_per_type) # Num of each particle type in reco
            reco_indices[0] += 1 # Ensure first catagory includes null particle
            # Now compute catagory boundaries
            # E.g. if self.n_reco_particles_per_type = [2, 3] (2 particles in type 1, 3 in type 2) then reco_indices = [0, 3, 6]
            # The first reco type gets one-hot vectors from index 0 to 2
            # The second reco type gets one-hot vectors from index 3 to 5
            reco_indices = torch.cat([torch.tensor([0]),reco_indices.cumsum(dim=0)],dim=0)
            self.onehot_tensors = [onehot_matrix[:,ia:ib].T for ia,ib in zip(reco_indices[:-1],reco_indices[1:])] # Extracts one-hot vectors for each reco type
        else:
            onehot_dim = 0

        # Build embeddings
        self.hard_embeddings = self.make_embeddings(self.hard_input_features_per_type)
        self.reco_embeddings = self.make_embeddings(self.reco_input_features_per_type, onehot_dim)

        # Build Transformer
        if 'd_model' in transformer_args.keys():
            print (f'Transformer args: will override `d_model` to {self.embed_dim}')
        transformer_args["d_model"] = self.embed_dim # Ensure transformer internal feature dimension matches embedding size
        if "dropout" not in transformer_args:
            transformer_args["dropout"] = self.dropout
        self.transformer = nn.Transformer(batch_first=True, **transformer_args) # By default nn.Transformer expects (seq_length, batch_size, feature_dim)

        # Missing particles across events are padded, therefore able to create one global tgt_mask:
        # (assume sum(n_reco_particles_per_type)+1 is the same for all events)
        self.max_reco_len = sum(self.n_reco_particles_per_type) + 1
        self.tgt_mask = self.transformer.generate_square_subsequent_mask(self.max_reco_len)

        # Velocity net for bridging
        d_in = self.embed_dim + self.len_flow_feats + 1
        d_hidden = 128
        self.velocity_net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, self.len_flow_feats),
        )

        # Map flow features to corresponding indices in reco features
        self.flow_indices = []
        self.global_flow_features = []
        for i,(n,reco_features,flow_features) in enumerate(zip(self.n_reco_particles_per_type,self.reco_input_features_per_type,self.flow_input_features)):
            assert len(set(flow_features).intersection(set(reco_features))) == len(flow_features), f'Not all flow features {flow_features} found in reco_features for particle set #{i} ({reco_features})'
            indices = [reco_features.index(feature) for feature in flow_features]
            self.flow_indices.append(indices)
            self.global_flow_features.extend([feat for feat in flow_features if feat not in self.global_flow_features])


    def make_embeddings(self,input_features,onehot_dim=0):
        """
        Create embedding layers for different input features.
        Ensures inputs with identical features share the same embedding layers.
        """
        feature_embeddings = {} # Dict to store embedding layers for different feature sets
        embeddings = nn.ModuleList() # List of PyTorch layers to be registered as part of model
        for features in input_features: # Iterate over each feature set
            if not isinstance(features,tuple):
                features = tuple(features) # Ensure treated as tuples
            if features not in feature_embeddings.keys(): # Check if an embedding exists for this features set
                # If this set of features is new, create a new embedding layer
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
                # Reuse embedding for feature sets with the same structure where embedding already exists
                embedding = feature_embeddings[features]
            embeddings.append(embedding)
        return embeddings

    # Methods to set optimizer and scheduler externally after model initialization
    def set_optimizer(self, optimizer):
        self._optimizer = optimizer


    def set_scheduler_config(self, scheduler_config):
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


    def conditioning(self,hard_data,hard_mask_exist,reco_data,reco_mask_exist):
        """
        Computes the conditioning context for the CFM model by encoding hard and reco data using a Transformer.
        """
        # Add null token to first reco object
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

        # Apply onehot encoding
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

        # Apply embeddings and concat along particle axis
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

        # Expand attention mask
        # Need to turn 0->1 when particle exists
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

        # Transformer processing
        condition = self.transformer(
            src = hard_data,                                # encoder (hard) input
            tgt = reco_data_null,                           # decorder (reco) input
            tgt_mask = self.tgt_mask.to(hard_data.device),  # triangular (causality) mask
            src_key_padding_mask = hard_mask_attn,          # encoder (hard) mask
            memory_key_padding_mask = hard_mask_attn,       # encoder output / memory mask
            tgt_key_padding_mask = reco_mask_attn,          # decoder (reco) mask
        )

        return condition


    def bridging_distribution(self, x0, x1, t):
        """
        Defines bridging distribution used in flow matching to interpolate between x0 and x1.
        Adds Gaussian noise scaled by sigma
        """
        eps = torch.randn_like(x0) # Random gaussian noise
        while t.dim() < x0.dim(): # Ensure t can can be broadcasted to match x dimensions
            t = t.unsqueeze(-1) # Reshapes t → (Batch, Particles, Features)
        xt = (1.0 - t) * x0 + t * x1 + self.sigma * eps

        return xt


    def compute_velocity(self, context, x_t, t_):
        """
        Compute velocity vector using the `velocity_net` network.
        """
        B, sum_reco_tokens, _ = context.shape
        # Prepare net_in to include: transformer contex, current state, timestep
        net_in = torch.cat([
            context.reshape(B * sum_reco_tokens, -1),          # [B*sum_reco, embed_dim]
            x_t.reshape(B * sum_reco_tokens, self.len_flow_feats),
            t_.repeat_interleave(sum_reco_tokens).unsqueeze(-1),# [B*sum_reco, 1]
        ], dim=1)  # [B*sum_reco, embed_dim + len_flow_feats + 1]

        # Obtain predicted velocities
        v_pred = self.velocity_net(net_in).reshape(B, sum_reco_tokens, self.len_flow_feats)  # [B, sum_reco, len_flow_feats]

        return v_pred

    def shared_eval(self, batch, batch_idx, prefix):
        # Compute velocity loss
        loss = self.forward(batch)  # Forward pass computes MSE loss on velocity fields

        # Extract masks
        reco_mask = batch["reco"]["mask"]

        # Log per object loss
        idx = 0
        for i, n in enumerate(self.n_reco_particles_per_type):
            for j in range(n):
                if reco_mask[i][:, j].sum() != 0:  # average over existing particles
                    self.log(
                        f"{prefix}/velocity_loss_{self.reco_particle_type_names[i]}_{j}",
                        loss,
                        prog_bar=False,
                    )
                idx += 1

        # Loss per process
        if "process" in batch.keys():
            for idx in torch.unique(batch["process"]).sort()[0]:
                process_idx = torch.where(batch["process"] == idx)[0]
                process_name = self.process_names[idx] if self.process_names is not None else str(idx.item())
                self.log(
                    f"{prefix}/velocity_loss_process_{process_name}",
                    loss.mean(),  # Averaged over all reco particles
                    prog_bar=False,
                )

        # Log total velocity loss
        self.log(f"{prefix}/velocity_loss_tot", loss.mean(), prog_bar=True)
        self.log("val_loss", loss.mean(), prog_bar=True)

        return loss.mean()


    def training_step(self, batch, batch_idx):
        """Defines the training step for each batch and computes the loss."""
        return self.shared_eval(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        """Validation step for each batch, logs validation loss."""
        return self.shared_eval(batch, batch_idx, 'val')


    def pack_reco_features(self, reco_data, reco_mask):
        """Organize the features of the reco-level particles into a single flattened tensor with masking"""
        B = reco_data[0].shape[0] # Batch size
        n_each_type = [rd.shape[1] for rd in reco_data] # List of num particles for each reco type
        sum_reco = sum(n_each_type) # Total num of reco particles for a given event (across all types)
        len_flow_feats = self.len_flow_feats  # Maximum number of flow features across all reco particles

        # Initialize tensors
        x_real = torch.zeros((B, sum_reco, len_flow_feats), device=reco_data[0].device) # To store feature values
        feat_mask = torch.zeros((B, sum_reco, len_flow_feats), device=reco_data[0].device) # Which values are valid

        offset = 0 # Where in the flat tensor current reco type should be placed
        for type_i, n in enumerate(self.n_reco_particles_per_type):
            # type_i = index of reco particle type (eg. 0 could correspond to MET)
            # n = number of particles of this particle type (e.g. 15 for jets)
            rd_i = reco_data[type_i]      # Contents for that particle type. Has shape [B, Particles, Features]
            mask_i = reco_mask[type_i]    # Existance mask for the current particle type. Shape: [B, Particles]
            feat_list_i = self.flow_input_features[type_i]  # Feature names (e.g., ["pt", "phi"]) for this particle type

            for feat_j, feat_name in enumerate(self.flow_input_features[type_i]): # Loop over each flow feature
                if feat_name in feat_list_i:
                    col_idx = feat_list_i.index(feat_name) # Find idx of feat_name in feat_list_i
                    # Assign the value of a feature to the corresponding position
                    # Since flow_input_features is done per reco particle type, map to global position
                    x_real[:, offset:offset + n, feat_j] = rd_i[:, :, col_idx] # Assigns feature values
                    feat_mask[:, offset:offset + n, feat_j] = mask_i # Mark valid features

            # For features not present in this reco type, they remain zero and masked
            offset += n # Move pointer for next reco particle type

        return x_real, feat_mask, sum_reco


    def unpack_reco_samples(self, x_t, reco_mask_exist):
        """Unpack the flat x_t tensor into per reco type tensors."""
        reco_samples = [] # Will contain list of reco particle types
        offset = 0 # Track where in x_t the next particle type starts
        for type_i, n in enumerate(self.n_reco_particles_per_type): # Loop over each particle type
            flow_features = self.flow_input_features[type_i] # List of features used for this particle type
            F_j = len(flow_features) # Number of features

            # Extract portion of x_t for this particle type
            sample_i = x_t[:, offset:offset + n, :F_j]  # [Batch size, particles, Features]
            reco_samples.append(sample_i)

            # Update offset
            offset += n

        return reco_samples


    def forward(self, batch):
            """Compute the Conditional Flow-Matching loss."""
            hard_data = batch["hard"]["data"]
            hard_mask = batch["hard"]["mask"]
            reco_data = batch["reco"]["data"]
            reco_mask = batch["reco"]["mask"]

            # Safety checks #
            assert len(hard_data) == len(self.hard_embeddings), f'{len(hard_data)} hard objects but {len(self.hard_embeddings)} hard embeddings'
            assert len(hard_mask) == len(self.hard_embeddings), f'{len(hard_mask)} hard objects but {len(self.hard_embeddings)} hard embeddings'
            assert len(reco_data) == len(self.reco_embeddings), f'{len(reco_data)} reco objects but {len(self.reco_embeddings)} reco embeddings'
            assert len(reco_mask) == len(self.reco_embeddings), f'{len(reco_mask)} reco objects but {len(self.reco_embeddings)} reco embeddings'

            # Get the Transformer output
            transformer_out = self.conditioning(hard_data, hard_mask, reco_data, reco_mask)  # [B, sum_reco+1, embed_dim]

            # Remove the null token
            context = transformer_out[:, 1:, :]  # [B, sum_reco, embed_dim]

            # Pack the reco features
            x_real, feat_mask, sum_reco = self.pack_reco_features(reco_data, reco_mask)  # [B, sum_reco, len_flow_feats]
            B, sum_reco_tokens, len_flow_feats = x_real.shape

            # Initialize the bridging distribution
            x0 = torch.randn_like(x_real)  # Ranom Gaussian distribution. Shape: [B, sum_reco, len_flow_feats]
            x1 = x_real
            t = torch.rand(B, device=x_real.device)  # Each event within a batch gets a random timestep

            # We want to transform from Gaussian noise to the reco data, conditioned on hard data:
            x_t = self.bridging_distribution(x0, x1, t)  # [B, sum_reco, len_flow_feats]

            # Compute the velocity vector
            v_pred = self.compute_velocity(context, x_t, t)  # [B, sum_reco, len_flow_feats]

            # Calculate the TRUE velocity vector
            v_true = x1 - x0  # [B, sum_reco, len_flow_feats]

            # Get the loss
            diff = (v_pred - v_true) ** 2 * feat_mask  # [B, sum_reco, len_flow_feats]
            loss_per_event = diff.mean(dim=(1, 2))  # [B]
            loss = loss_per_event.mean(dim=0)  # Scalar

            return loss


    def sample(self, hard_data, hard_mask_exist, reco_data, reco_mask_exist, N_sample=1, steps=10):
        """
        Generate N_sample new saomples by evolving the bridging distribution using the learned velocity field.

        Args:
            hard_data: List of tensors, 1 per hard particle type, shape [batch_size, particles, features]
            hard_mask_exist: List of tensors, 1 per hard particle type, shape [batch_size, particles, features]
            reco_data: List of tensors, 1 per reco particle type, shape [batch_size, particles, features]
            reco_mask_exist: List of tensors, 1 per reco particle type, shape [batch_size, particles, features]
            N_sample: Number of samples to generate
            steps: Number of Euler integration steps for bridging

        Returns:
            samples: List of tensors, one per reco type, shape [N_sample, batch_size, particles, features]
        """
        N_reco = len(reco_data) # number of reco particle types
        B = reco_data[0].shape[0] # Batch size
        samples = [ [] for _ in range(N_reco) ]  # Initialize list for each reco type

        for s in range(N_sample):
            # Get transformer conditions
            conditions = self.conditioning(hard_data, hard_mask_exist, reco_data, reco_mask_exist)  # [B, sum_reco+1, embed_dim]

            # Remove null token
            context = conditions[:, 1:, :]  # [B, sum_reco, embed_dim]

            # Pack the reco features
            x_real, _, _ = self.pack_reco_features(reco_data, reco_mask_exist)  # [B, sum_reco, len_flow_feats]
            B = x_real.shape[0]

            # Initialize the bridging distribution
            x0 = torch.randn_like(x_real)  # [B, sum_reco, len_flow_feats]
            x1 = x_real
            t = torch.rand(B, device=x_real.device)

            # Transform from Gaussian noise to reco data, conditioned on hard data
            x_t = self.bridging_distribution(x0, x1, t)  # [B, sum_reco, len_flow_feats]

            # Euler integration stepping for the bridging process
            dt = 1.0 / steps
            for step_i in range(steps):
                t_value = step_i * dt
                t_ = torch.full((B,), t_value, device=x_t.device)
                v_t = self.compute_velocity(context, x_t, t_)  # [B, sum_reco, len_flow_feats]
                x_t = x_t + dt * v_t  # [B, sum_reco, len_flow_feats]

            # Unpack samples
            reco_samples = self.unpack_reco_samples(x_t, reco_mask_exist)  # List of [B, particles, features]

            for i in range(N_reco): # Append samples
                samples[i].append(reco_samples[i])  # Append [B, particles, features] to list for reco particle type i

        # 9. Stack Samples per Reco Type
        samples = [torch.stack(s, dim=0) for s in samples]  # List of [N_sample, B, particles, features]

        return samples