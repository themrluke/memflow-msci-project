import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import math
import numpy as np
import torch.optim as optim
from typing import Optional, Union, Dict, Any, List
from zuko.distributions import DiagNormal
import warnings

from .optimal_transport import OTPlanSampler
from memflow.models.utils import lowercase_recursive



def pad_t_like_x(t, x):
    """Reshapes time vector t so it can be broadcast with with the tensor x"""
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))



class CircularEmbedding(nn.Module):
    """
    Replaces raw φ values with [sin(φ), cos(φ)].
    It expects the indices of the raw φ columns in the input.
    """
    def __init__(self, in_features, out_features, circular_indices=None, embed_act=None, dropout=0.0):
        super().__init__()
        self.circular_indices = circular_indices  # list of indices corresponding to phi
        # The linear layer expects extra channels for each phi replaced.
        self.linear = nn.Linear(in_features + (len(circular_indices) if circular_indices else 0), out_features)
        self.act = embed_act() if embed_act is not None else None
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None


    def forward(self, x):
        if self.circular_indices is not None:
            # Extract raw phi columns.
            circular_vals = x[..., self.circular_indices]
            sin_phi = torch.sin(circular_vals)
            cos_phi = torch.cos(circular_vals)

            # Remove the raw phi columns.
            idx_all = list(range(x.shape[-1]))
            idx_remaining = [i for i in idx_all if i not in self.circular_indices]
            x_remaining = x[..., idx_remaining]

            # Concatenate the remaining features with the sin/cos values.
            x = torch.cat([x_remaining, sin_phi, cos_phi], dim=-1)

        out = self.linear(x)
        if self.act is not None:
            out = self.act(out)
        if self.dropout is not None:
            out = self.dropout(out)

        return out



class BaseCFM(L.LightningModule):
    """
    Base class for Conditional Flow Matching models operating in a 'global' mode.
    Serves as a parent class for other methods.
    Subclasses can override the bridging logic.
    Handles:
        - Transformer-based conditioning
        - Feature packing/unpacking
        - Velocity network
        - Data flow in forward()
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
        reco_mask_attn, # Which particles should contribute to the Transformer attention
        hard_mask_attn,
        dropout=0.0,
        transformer_args={},
        cfm_args={},
        sigma: Union[float, int] = 0.0,
        optimizer=None,
        scheduler_config=None,
        onehot_encoding=False, # One-hot vectors are appended to reco features to help model understand particle type
    ):
        super().__init__()

        self.save_hyperparameters(
            ignore=["optimizer", "scheduler_config"]  # Don't store these
        )

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
        # For each reco type, count "phi" as two channels.
        def effective_feature_count(feats):
            return sum(2 if feat == "phi" else 1 for feat in feats)
        self.len_flow_feats = max([effective_feature_count(feats) for feats in self.flow_input_features])

        self.onehot_encoding = onehot_encoding
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
        assert len(n_reco_particles_per_type) == len(reco_input_features_per_type), \
            f'{len(n_reco_particles_per_type)} sets of reco particles but got {len(reco_input_features_per_type)} sets of input features'
        assert len(n_hard_particles_per_type) == len(hard_input_features_per_type), \
            f'{len(n_hard_particles_per_type)} sets of hard particles but got {len(hard_input_features_per_type)} sets of input features'
        assert len(flow_input_features) == len(reco_input_features_per_type), \
            f'Number of reco features ({len(reco_input_features_per_type)}) != number of flow features ({len(flow_input_features)})'

        # Make onehot encoded vectors
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
        self.norm = nn.LayerNorm(self.embed_dim)  # Apply LayerNorm after Transformer

        # Missing particles across events are padded, therefore able to create one global tgt_mask:
        # (assume sum(n_reco_particles_per_type)+1 is the same for all events)
        self.max_reco_len = sum(self.n_reco_particles_per_type) + 1
        self.tgt_mask = self.transformer.generate_square_subsequent_mask(self.max_reco_len)

         # --- Define the Partial Embedding Module ---
        # This module maps a partial feature vector (of length F) to a fixed dimension.
        # We set d_partial equal to embed_dim.
        self.partial_embed = nn.Linear(self.len_flow_feats, self.embed_dim)

        # Velocity network for bridging
        d_in = 2 * self.embed_dim + 2
        d_hidden = cfm_args['dim_feedforward']
        activation_fn = cfm_args['activation']
        vel_layers = []
        for _ in range(cfm_args['num_layers']):
            vel_layers.append(nn.Linear(d_in, d_hidden))
            vel_layers.append(nn.BatchNorm1d(d_hidden))   # BatchNorm before activation
            vel_layers.append(activation_fn()) # New instance every time
            d_in = d_hidden
        vel_layers.append(nn.Linear(d_hidden, 1))
        vel_layers.append(nn.BatchNorm1d(1))  # Normalize final output
        self.velocity_net = nn.Sequential(*vel_layers)

        # Mapping from flow-feature space to embedding space (for autoregressive history update).
        self.flow_to_embed = nn.Linear(self.len_flow_feats, self.embed_dim)

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
            if not isinstance(features, tuple):
                features = tuple(features) # Ensure treated as tuples

            if 'phi' in features:
                circular_indices = [i for i, feat in enumerate(features) if feat == 'phi']
                embedding = CircularEmbedding(
                    in_features=len(features),
                    out_features=self.embed_dims[0],
                    circular_indices=circular_indices,
                    embed_act=self.embed_act,
                    dropout=self.dropout
                )
                layers = [embedding]
                for i in range(1, len(self.embed_dims)):
                    layers.append(nn.Linear(self.embed_dims[i - 1], self.embed_dims[i]))
                    if self.embed_act is not None and i < len(self.embed_dims) - 1:
                        layers.append(self.embed_act())
                    if self.dropout != 0.0:
                        layers.append(nn.Dropout(self.dropout))
                embedding = nn.Sequential(*layers)
                key = features
            else:
                adjusted_features = list(features)
                key = tuple(adjusted_features)
                if key not in feature_embeddings: # Check if an embedding exists for this features set
                    # If this set of features is new, create a new embedding layer
                    layers = []
                    in_dim = len(adjusted_features) + onehot_dim
                    for i, out_dim in enumerate(self.embed_dims):
                        layers.append(nn.Linear(in_dim if i == 0 else self.embed_dims[i - 1], out_dim))
                        if self.embed_act is not None and i < len(self.embed_dims) - 1:
                            layers.append(self.embed_act())
                        if self.dropout != 0.0:
                            layers.append(nn.Dropout(self.dropout))
                    embedding = nn.Sequential(*layers)
                    feature_embeddings[key] = embedding
                else:
                    # Reuse embedding for feature sets with the same structure where embedding already exists
                    embedding = feature_embeddings[key]
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
        # Turn them into boolean arrays
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
        condition = self.norm(condition)   # Apply LayerNorm

        return condition


    def get_bridging_pair(self, x0, x1):
        """
        Base version: Returns the same pair (no Optimal Transport pairing).
        Child processes can override this function if they need to re-pair x0 & x1 via OT
        and record this change to pass to the loss function calculation.
        """
        return x0, x1


    def bridging_distribution(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Placeholder function in base class which subclasses will override."""
        raise NotImplementedError("Child classes need to override this bridging_distribution function.")


    def compute_velocity(self, context, partial_features, x_t, t):
        """
        Computes the predicted velocity for a single scalar feature.
        Inputs:
          - context: [B, 1, embed_dim] from the Transformer for the current particle.
          - partial_features: [B, F] where F is self.len_flow_feats.
            This is the partially generated feature vector for the current particle (zeros for not-yet-generated features).
          - x_t: [B, 1, 1] the bridging sample for the current feature.
          - t: [B, 1] the current time (after unsqueeze).
        We first embed the partial feature vector to get a fixed-length representation,
        then concatenate with context and t to feed to the velocity network.
        """
        B = context.shape[0]
        # Get embedding for the partial features.
        partial_emb = self.partial_embed(partial_features)  # [B, embed_dim]
        partial_emb = partial_emb.unsqueeze(1)              # [B, 1, embed_dim]
        net_in = torch.cat([
            context,              # [B, 1, embed_dim]
            partial_emb,          # [B, 1, embed_dim]
            t.unsqueeze(-1)       # [B, 1, 1]
        ], dim=2)  # Now shape [B, 1, 2*embed_dim + 1]
        net_in = net_in.reshape(B, -1)  # [B, 2*embed_dim + 1]
        v_pred = self.velocity_net(net_in).reshape(B, 1, 1)
        return v_pred


    def shared_eval(self, batch, batch_idx, prefix):
        # Compute velocity loss
        diff = self.forward(batch)  # Forward pass computes MSE loss on velocity fields

        # 1) Overall loss: average over particles and features, then over batch
        loss_per_event = diff.mean(dim=(1, 2))  # [B]
        loss = loss_per_event.mean(dim=0)  # Scalar
        self.log(f"{prefix}/velocity_loss_tot", loss, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)

        # 2) Per-feature error: average over particles
        global_offset = 0  # This tracks our current starting index along the object axis.
        for type_idx, num_obj in enumerate(self.n_reco_particles_per_type):
            # Determine the effective number of columns for this reco type.
            effective_count = sum(2 if feat == "phi" else 1 for feat in self.flow_input_features[type_idx])
            # Extract the slice of diff that corresponds to this reco type:
            #   diff_type has shape [B, num_obj, effective_count]
            diff_type = diff[:, global_offset:global_offset + num_obj, :effective_count]

            # Now, for each field in the flow features for this reco type, compute its error.
            local_col = 0  # Tracks column position in the effective representation for this type.
            for feat in self.flow_input_features[type_idx]:
                if feat == "phi":
                    # For phi, two columns represent it.
                    # We compute the mean error over both columns.
                    field_slice = diff_type[:, :, local_col:local_col + 2]  # shape: [B, num_obj, 2]
                    field_error = field_slice.mean()  # overall mean error for phi for this type.
                    self.log(f"velocity_loss_field_{self.reco_particle_type_names[type_idx]}_{feat}", field_error, prog_bar=False)
                    local_col += 2
                else:
                    # For non-phi features, take one column.
                    field_slice = diff_type[:, :, local_col:local_col + 1]  # shape: [B, num_obj, 1]
                    field_error = field_slice.mean()
                    self.log(f"velocity_loss_field_{self.reco_particle_type_names[type_idx]}_{feat}", field_error, prog_bar=False)
                    local_col += 1

            global_offset += num_obj  # Move to the next reco type in the concatenated dimension.

        # 3) Per-object error: average over features
        object_error = diff.mean(dim=2)  # [B, sum_reco]
        mean_object_error = object_error.mean(dim=0)  # [sum_reco]
        # To log per object by reco type, iterate over types using self.n_reco_particles_per_type
        offset = 0
        for type_idx, num_obj in enumerate(self.n_reco_particles_per_type):
            for j in range(num_obj):
                # Compute error for object j of type type_idx (averaged over batch)
                obj_err = mean_object_error[offset + j]
                self.log(f"velocity_loss_object_{self.reco_particle_type_names[type_idx]}_{j}", obj_err, prog_bar=False)
            offset += num_obj

        return loss


    def training_step(self, batch, batch_idx):
        """Defines the training step for each batch and computes the loss."""
        return self.shared_eval(batch, batch_idx, 'train')


    def validation_step(self, batch, batch_idx):
        """Validation step for each batch, logs validation loss."""
        return self.shared_eval(batch, batch_idx, 'val')


    def pack_reco_features(self, reco_data, reco_mask):
        """
        Organize the features of the reco-level particles into a single flattened tensor with masking
        For the feature "phi", replace the raw value with [sin(φ), cos(φ)].
        """
        B = reco_data[0].shape[0] # Batch size
        n_each_type = [rd.shape[1] for rd in reco_data] # List of num particles for each reco type
        sum_reco = sum(n_each_type) # Total num of reco particles for a given event (across all types)

        # Determine the correct number of output feature columns
        feature_col_count = max(
            sum(2 if feat == "phi" else 1 for feat in self.flow_input_features[type_i])
            for type_i in range(len(self.n_reco_particles_per_type))
        )

        # Initialize tensors
        x_real = torch.zeros((B, sum_reco, feature_col_count), device=reco_data[0].device) # To store feature values
        feat_mask = torch.zeros((B, sum_reco, feature_col_count), device=reco_data[0].device) # Which values are valid

        offset = 0 # Where in the flat tensor current reco type should be placed
        for type_i, n in enumerate(self.n_reco_particles_per_type):
            # type_i = index of reco particle type (eg. 0 could correspond to MET)
            # n = number of particles of this particle type (e.g. 15 for jets)
            rd_i = reco_data[type_i]      # Contents for that particle type. Has shape [B, Particles, Features]
            mask_i = reco_mask[type_i]    # Existance mask for the current particle type. Shape: [B, Particles]
            flow_feats = self.flow_input_features[type_i]  
            reco_feats = self.reco_input_features_per_type[type_i]  # Feature order for this type

            col_offset = 0
            for feat in flow_feats:
                if feat not in reco_feats:
                    raise ValueError(f"Feature '{feat}' not found in reco_features for type {type_i}: {reco_feats}")

                col_idx = reco_feats.index(feat)  # Get the real index of this feature
                if feat == "phi":
                    phi_raw = rd_i[:, :, col_idx]
                    x_real[:, offset:offset+n, col_offset] = torch.sin(phi_raw)
                    x_real[:, offset:offset+n, col_offset+1] = torch.cos(phi_raw)
                    feat_mask[:, offset:offset+n, col_offset] = mask_i
                    feat_mask[:, offset:offset+n, col_offset+1] = mask_i
                    col_offset += 2  # Use 2 columns for phi

                else:
                    # Assign the value of a feature to the corresponding position
                    # Since flow_input_features is done per reco particle type, map to global position
                    x_real[:, offset:offset+n, col_offset] = rd_i[:, :, col_idx] # Assigns feature values
                    feat_mask[:, offset:offset+n, col_offset] = mask_i # Mark valid features
                    col_offset += 1

            # For features not present in this reco type, they remain zero and masked
            offset += n # Move pointer for next reco particle type

        return x_real, feat_mask, sum_reco


    def unpack_reco_samples(self, x_t, reco_mask_exist):
        """
        Unpack the flat x_t tensor into per reco type tensors.
        For the feature "phi", combine [sin(φ), cos(φ)] back into an angle via atan2.
        Normalise the (sin, cos) pair to ensure they lie on the unit circle.
        """
        reco_samples = [] # Will contain list of reco particle types
        offset = 0 # Track where in x_t the next particle type starts
        for type_i, n in enumerate(self.n_reco_particles_per_type): # Loop over each particle type
            flow_feats = self.flow_input_features[type_i] # List of features used for this particle type
            effective_count = sum(2 if feat == "phi" else 1 for feat in flow_feats)
            sample_i = x_t[:, offset:offset+n, :effective_count]
            out_features = []
            col_offset = 0
            for feat in flow_feats:
                if feat == "phi":
                    sin_phi = sample_i[:, :, col_offset]
                    cos_phi = sample_i[:, :, col_offset+1]
                    # Normalize the (sin, cos) pair.
                    norm = torch.sqrt(sin_phi**2 + cos_phi**2 + 1e-6)
                    sin_phi = sin_phi / norm
                    cos_phi = cos_phi / norm
                    phi = torch.atan2(sin_phi, cos_phi).unsqueeze(-1)
                    out_features.append(phi)
                    col_offset += 2
                else:
                    feat_val = sample_i[:, :, col_offset].unsqueeze(-1)
                    out_features.append(feat_val)
                    col_offset += 1

            out_sample = torch.cat(out_features, dim=-1)
            reco_samples.append(out_sample)
            offset += n # Update offset

        return reco_samples


    def velocity_target(self, x0, x1, x_t, t):
        """
        By default, the velocity target used for the loss calculation is (x1 - x0).
        Child classes can override this to match their bridging distribution's derivative.
        """
        return x1 - x0


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
            diff_list = []

            # Loop over each reco particle token.
            for i in range(sum_reco):
                context_particle = context[:, i:i+1, :]  # [B, 1, embed_dim]
                # x0_particle is sampled as noise.
                x0_particle = torch.randn(B, 1, self.len_flow_feats, device=x_real.device)
                x1_particle = x_real[:, i:i+1, :]         # Ground truth features for particle i.
                feature_diffs = []
                # Initialize partial_features as zeros (shape: [B, F])
                partial_features = torch.zeros(B, F, device=x_real.device)
                # For teacher forcing, we generate features sequentially.
                for f in range(F):
                    t_val = torch.rand(B, device=x_real.device)  # [B]
                    # We unsqueeze t_val later in compute_velocity.
                    # For each particle, we use the same re-pairing for all features.
                    paired_x0, paired_x1 = self.get_bridging_pair(x0_particle, x1_particle)
                    # Compute bridging sample for the current feature f.
                    # Here we assume that bridging_distribution works per feature.
                    # We extract only the f-th feature from paired_x0 and paired_x1:
                    x0_f = paired_x0[:, :, f:f+1]  # [B, 1, 1]
                    x1_f = paired_x1[:, :, f:f+1]  # [B, 1, 1]
                    x_t_f = self.bridging_distribution(x0_f, x1_f, t_val)  # [B, 1, 1]
                    # Compute predicted velocity, conditioned on current particle context and partial_features.
                    v_pred = self.compute_velocity(context_particle, partial_features, x_t_f, t_val)
                    # Compute target velocity (using teacher forcing: ground-truth x1_f and x0_f).
                    v_true = self.velocity_target(x0_f, x1_f, x_t_f, t_val)
                    feature_diffs.append((v_pred - v_true) ** 2)
                    # Update partial_features: use ground truth feature for position f.
                    gt_feature = x1_particle[:, :, f]  # [B, 1]
                    # Set column f of partial_features to ground truth.
                    partial_features[:, f] = gt_feature.squeeze(-1)
                diff_particle = torch.cat(feature_diffs, dim=-1)  # [B, 1, F]
                diff_list.append(diff_particle)
            diff = torch.cat(diff_list, dim=1)  # [B, total, F]
            return diff


    def rk4_step(self, context, x, t_val, dt):
        k1 = self.compute_velocity(context, x, t_val)
        k2 = self.compute_velocity(context, x + 0.5*dt*k1, t_val + 0.5*dt)
        k3 = self.compute_velocity(context, x + 0.5*dt*k2, t_val + 0.5*dt)
        k4 = self.compute_velocity(context, x +     dt*k3, t_val +     dt)
        return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


    def sample(self, hard_data, hard_mask_exist, reco_data, reco_mask_exist, N_sample=1, steps=10, store_trajectories=False):
        """
        Autoregressively generates new samples.
        - The Transformer encoder processes hard_data.
        - For each reco particle (token), the Transformer decoder is run sequentially.
        - Then, for each particle, features are generated one-by-one:
            Each feature is generated conditioned on the particle context and the already generated features (partial_features).
        Returns:
          - reco_samples: list of tensors (one per reco type) of final generated samples.
          - all_traj: (optional) trajectories of intermediate steps.
        """
        B = hard_data[0].shape[0]
        device = hard_data[0].device
        hard_mask_cat = torch.cat(hard_mask_exist, dim=1)
        hard_emb = torch.cat([self.hard_embeddings[i](hard_data[i]) for i in range(len(self.hard_embeddings))], dim=1)
        hard_emb = hard_emb * hard_mask_cat[..., None]
        if self.hard_mask_attn is None:
            hard_mask_attn = hard_mask_cat
        else:
            hard_mask_attn = torch.logical_or(self.hard_mask_attn.to(device), hard_mask_cat)
        if hard_mask_attn.dtype != torch.bool:
            hard_mask_attn = hard_mask_attn > 0
        hard_mask_attn = torch.zeros_like(hard_mask_attn, dtype=torch.float32).masked_fill(~hard_mask_attn, float("-inf"))
        encoder_output = self.transformer.encoder(hard_emb, src_key_padding_mask=hard_mask_attn)
        total = sum(self.n_reco_particles_per_type)
        d_model = self.embed_dim
        null_token = torch.ones((B, 1, d_model), device=device) * -1
        generated_tokens = [null_token]
        final_particles = []  # List to store full particle features
        all_traj = [] if store_trajectories else None

        for i in range(1, total+1):
            prev_seq = torch.cat(generated_tokens, dim=1)  # [B, i, d_model]
            tgt_mask = self.transformer.generate_square_subsequent_mask(prev_seq.size(1)).to(device)
            dec_out = self.transformer.decoder(prev_seq, encoder_output, tgt_mask=tgt_mask)
            token_context = dec_out[:, -1:, :]  # [B, 1, d_model]
            # For each particle, we generate its features sequentially.
            # Initialize partial_features with zeros: shape [B, F]
            partial_features = torch.zeros(B, self.len_flow_feats, device=device)
            particle_feature_list = []
            particle_traj = [] if store_trajectories else None
            for f in range(self.len_flow_feats):
                x0_feat = torch.randn(B, 1, 1, device=device)
                x_feat = x0_feat.clone()
                traj_feat = [] if store_trajectories else None
                dt = 1.0 / steps
                for step in range(steps):
                    t_val = torch.full((B,), step * dt, device=device)
                    x_feat = self.rk4_step(token_context, x_feat, t_val, dt)
                    if store_trajectories:
                        traj_feat.append(x_feat.detach().cpu())
                if store_trajectories:
                    particle_traj.append(torch.stack(traj_feat, dim=0))
                particle_feature_list.append(x_feat)  # x_feat is [B, 1, 1]
                # In sampling, we update partial_features with the generated value.
                partial_features[:, f] = x_feat.squeeze(-1).squeeze(-1)
            if store_trajectories:
                all_traj.append(particle_traj)
            # Concatenate all generated features for this particle: [B, 1, F]
            particle_flow = torch.cat(particle_feature_list, dim=-1)
            final_particles.append(particle_flow)
            # Update autoregressive history: map particle_flow to an embedding.
            new_embed = self.flow_to_embed(particle_flow)
            generated_tokens.append(new_embed)
        x_t_full = torch.cat(final_particles, dim=1)  # [B, total, F]
        reco_samples = self.unpack_reco_samples(x_t_full, reco_mask_exist)
        return reco_samples, all_traj



class StandardCFM(BaseCFM):
    """
    Child class which uses a standard bridging distribution:
        x(t) = (1 - t)*x0 + t*x1 + sigma*eps
    """
    def bridging_distribution(self, x0, x1, t):
        """
        Defines bridging distribution used in flow matching to interpolate between x0 and x1.
        Adds Gaussian noise scaled by sigma
        """
        eps = torch.randn_like(x0) # Random gaussian noise
        t = pad_t_like_x(t, x0) # Expand t if needed to match x1 shape
        xt = (1.0 - t) * x0 + t * x1 + self.sigma * eps

        return xt



class OptimalTransportCFM(BaseCFM):
    """Child class for bridging that uses Optimal Transport to re-pair x0 <-> x1."""
    def __init__(
            self,
            *args,
            sigma: Union[float, int] = 0.0,
            ot_method="exact",
            ot_reg: float=0.05,
            normalize_cost=False,
            **kwargs,
    ):

        super().__init__(*args, sigma=sigma, **kwargs)
        # Create an OTPlanSampler. Use exact method by default
        self.ot_sampler = OTPlanSampler(method=ot_method, reg=ot_reg, normalize_cost=normalize_cost)


    def get_bridging_pair(self, x0, x1):
        # Do OT re-pairing:
        paired_x0, paired_x1 = self.ot_sampler.sample_plan(x0, x1, replace=True)

        return paired_x0, paired_x1


    def bridging_distribution(self, x0, x1, t):
        """Re-pair x0, x1 with an Optimal Transport plan and perform bridging."""

        # Now do the original CFM bridging
        eps = torch.randn_like(x0)
        t = pad_t_like_x(t, x0)
        x_t = (1 - t) * x0 + t * x1 + self.sigma * eps
        return x_t



class TargetBridgingCFM(BaseCFM):
    """
    A subclass which inherits the BaseCFM and that implements Lipman et al. (ICLR 2023)
    style target Optimal Transport CFM. Sets the bridging distribution to:
        x(t) ~ N( t * x1, [1 - (1 - sigma)* t]^2 ).
    It disregards x0 in the bridging step, focusing solely on x1 and time t.
    """

    def velocity_target(self, x0: torch.Tensor, x1: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        For target bridging:
            x(t) = t * x1 + sigma_t * eps,
        where:
            sigma_t = 1 - (1 - sigma) * t.
        so derivative wrt t is:
            d/dt x(t) = x1 + d(sigma_t)/dt * eps.
        Since d(sigma_t)/dt = -(1 - sigma) (a constant), set:
            v_true = x1 - (1 - sigma) * eps.
        """
        dsigma_dt = -(1.0 - self.sigma)
        v_true = x1 + dsigma_dt * self.last_eps
        return v_true


    def bridging_distribution(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Ignore x0 here

        t = pad_t_like_x(t, x0)

        # Sample noise and store it in self.last_eps for use in the velocity target.
        self.last_eps = torch.randn_like(x1)
        sigma_t = 1.0 - (1.0 - self.sigma) * t # Compute time-dependent std
        x_t = t * x1 + sigma_t * self.last_eps # Bridging distribution

        return x_t



class SchrodingerBridgeCFM(BaseCFM):
    """
    Child class for Schrödinger bridge conditional flow matching method.
    This subclass inherits the BaseCFM parent class.
    """

    def __init__(self, *args, sigma: Union[float, int] = 0.1, ot_method='exact', **kwargs):
        super().__init__(*args, sigma=sigma, **kwargs)

        if sigma <= 0:
            raise ValueError(f"Sigma must be strictly positive, got {sigma}.")
        elif sigma < 1e-3:
            warnings.warn("Small sigma values may lead to numerical instability.")

        self.ot_method = ot_method
        self.ot_sampler = OTPlanSampler(method=ot_method, reg=2 * self.sigma**2)

    def get_bridging_pair(self, x0, x1):
        # Do OT re-pairing:
        paired_x0, paired_x1 = self.ot_sampler.sample_plan(x0, x1)
        return paired_x0, paired_x1

    def velocity_target(self, x0: torch.Tensor, x1: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = pad_t_like_x(t, x0)
        # Compute the derivative of the deterministic part.
        v_det = x1 - x0
        # Compute the derivative of the noise term.
        # Adding a small epsilon for stability.
        denom = 2 * torch.sqrt(t * (1.0 - t) + 1e-6)
        v_noise = self.sigma * self.last_eps * (1.0 - 2.0 * t) / denom
        v_true = v_det + v_noise
        return v_true

    def bridging_distribution(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        t = pad_t_like_x(t, x0)
        self.last_eps = torch.randn_like(x0)
        mu_t = (1.0 - t) * x0 + t * x1
        sigma_t = torch.sqrt(t * (1.0 - t)) * self.sigma
        x_t = mu_t + sigma_t * self.last_eps
        return x_t


class VariancePreservingCFM(BaseCFM):
    """
    Albergo et al. 2023 trigonometric interpolants class
    [3] Stochastic Interpolants: A Unifying Framework for Flows and Diffusions, Albergo et al.
    """

    def velocity_target(self, x0: torch.Tensor, x1: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Computes the true velocity for the variance preserving bridging.
        For:
            x(t) = cos((pi/2)*t) * x0 + sin((pi/2)*t) * x1 + sigma * eps,
        the derivative (ignoring the noise, which is independent of t) is:
            d/dt x(t) = - (pi/2) * sin((pi/2)*t) * x0 + (pi/2) * cos((pi/2)*t) * x1.
        """
        t = pad_t_like_x(t, x0)  # Ensure t has the proper shape.
        d_cos_dt = - (math.pi / 2) * torch.sin((math.pi / 2) * t)
        d_sin_dt = (math.pi / 2) * torch.cos((math.pi / 2) * t)
        v_true = d_cos_dt * x0 + d_sin_dt * x1

        return v_true


    def bridging_distribution(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = pad_t_like_x(t, x0)
        eps = torch.randn_like(x0)

        cos_part = torch.cos((math.pi / 2) * t)
        sin_part = torch.sin((math.pi / 2) * t)

        x_t = cos_part * x0 + sin_part * x1 + self.sigma * eps
        return x_t