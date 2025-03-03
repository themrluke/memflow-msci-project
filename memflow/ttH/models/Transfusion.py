# Script Name: Transfusion.py
# Author: Luke Johnson
# Description:
#     Implementation of various Conditional Flow Matching (CFM) models  using a Transformer-based
#     architecture for conditioning. Should use Transfusion style conditioning and autoregressive sampling from
#     "Precision-Machine Learning for the Matrix Element Method, SciPost Phys. 17 129 (2024)".
#     There are a variety of bridging distributions and Optimal Transport plans available in the subclasses.
#     This model is not learning very well at the momemnt.

#     Designed to be trained with the `train_CFM.ipynb notebook`

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

from .optimal_transport import OTPlanSampler
from memflow.models.utils import lowercase_recursive
from .utils import CircularEmbedding, pad_t_like_x



class BaseCFM(L.LightningModule):
    """
    Base class for Conditional Flow Matching (CFM) models operating in 'global' mode.
    Serves as a parent class for other CFM methods with different bridging distributions & Optimal Transport plans.
    Handles:
        - Transformer-based conditioning:
            - Transformer encoder: self-attention hard level events
            - Transformer decoder: self-attention current state and time value. Cross-attention hard-level events
        - Feature embeddings & transformations.
        - Learnable time-dependent velocity network for flow matching.

    Parameters:
        - embed_dims (Union[int, List[int]]): Embedding dimensions for input features.
        - embed_act (Callable[[], nn.Module]): Activation function for embeddings.
        - n_hard_particles_per_type (List[int]): Number of hard-level particles per type.
        - hard_particle_type_names (List[str]): Name of each hard particle type.
        - hard_input_features_per_type (List[List[str]]): Names for each hard particle type.
        - n_reco_particles_per_type (List[int]): Number of reco particles per type.
        - reco_particle_type_names (List[str]): Names for each reco particle type.
        - reco_input_features_per_type (List[List[str]]): Feature names for each reco particle type.
        - flow_input_features (List[List[str]]): Subset of feature names used in flow.
        - reco_mask_attn (Optional[torch.Tensor]): Attention mask for reco-level particles.
        - hard_mask_attn (Optional[torch.Tensor]): Attention mask for hard-level particles.
        - dropout (float): Dropout.
        - transformer_args (Dict[str, Any]): Arguments specifying Transformer architecture.
        - cfm_args (Dict[str, Any]): Arguments specifying CFM velocity network architecture.
        - sigma (Union[float, int]): Noise scaling factor.
        - optimizer (Optional[Any]): optimizer set externally.
        - scheduler_config (Optional[Any]): Scheduler configuration.
        - onehot_encoding (bool): If True, one-hot encoding is appended to reco features.
    """
    def __init__(
        self,
        embed_dims: Union[int, List[int]],
        embed_act: Callable[[], nn.Module],
        n_hard_particles_per_type: List[int],
        hard_particle_type_names: List[str],
        hard_input_features_per_type: List[List[str]],
        n_reco_particles_per_type: List[int],
        reco_particle_type_names: List[str],
        reco_input_features_per_type: List[List[str]],
        flow_input_features: List[List[str]],
        reco_mask_attn: Optional[torch.Tensor],
        hard_mask_attn: Optional[torch.Tensor],
        dropout: float = 0.0,
        transformer_args: Dict[str, Any] = {},
        cfm_args: Dict[str, Any] = {},
        sigma: Union[float, int] = 0.0,
        optimizer: Optional[Any] = None,
        scheduler_config: Optional[Any] = None,
        onehot_encoding: bool = False,
    ) -> None:

        super().__init__()

        # Store hyperparameters when saving model
        self.save_hyperparameters(ignore=["optimizer", "scheduler_config"])

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

        self.max_reco_len = sum(self.n_reco_particles_per_type) + 1
        self.tgt_mask = self.transformer.generate_square_subsequent_mask(self.max_reco_len)

         # --- Define the Partial Embedding Module ---
        # For autoregressive sampling we now want to update one “group” at a time.
        # For scalar features the group dimension is 1,
        # for circular features the group dimension is 2.
        # We define two separate “partial embed” layers for these two cases.
        self.partial_embed_scalar = nn.Linear(1, self.embed_dim)
        self.partial_embed_circular = nn.Linear(2, self.embed_dim)


        # Velocity network for bridging
        d_in = 2 * self.embed_dim + 1
        d_hidden = cfm_args['dim_hidden']
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


    def make_embeddings(
            self,
            input_features: List[Union[List[str], Tuple[str, ...]]],
            onehot_dim: int = 0
    ) -> nn.ModuleList:
        """
        Create embedding layers for different input features. Inputs with identical features share the same embedding layers.

        Args:
            - input_features (List[Union[List[str], Tuple[str, ...]]]): Feature names.
            - onehot_dim (int): Additional dimension if one-hot encoding used.

        Returns:
            - embeddings (nn.ModuleList): Embedding modules.
        """
        feature_embeddings = {} # Shared embedding layers
        embeddings = nn.ModuleList()
        for features in input_features: # Iterate over each feature set
            if not isinstance(features, tuple):
                features = tuple(features)

            if 'phi' in features: # Identify indices for circular features
                circular_indices = [i for i, feat in enumerate(features) if feat == 'phi']
                embedding = CircularEmbedding(
                    in_features=len(features),
                    out_features=self.embed_dims[0],
                    circular_indices=circular_indices,
                    embed_act=self.embed_act,
                    dropout=self.dropout
                )
                # Create additional linear layers inline with requested embedding dims
                layers = [embedding]
                for i in range(1, len(self.embed_dims)):
                    layers.append(nn.Linear(self.embed_dims[i - 1], self.embed_dims[i]))
                    if self.embed_act is not None and i < len(self.embed_dims) - 1:
                        layers.append(self.embed_act())
                    if self.dropout != 0.0:
                        layers.append(nn.Dropout(self.dropout))
                embedding = nn.Sequential(*layers)
                key = features

            else: # Non-circular features
                adjusted_features = list(features)
                key = tuple(adjusted_features)
                if key not in feature_embeddings: # Check if an embedding exists for this features set
                    # If this set of features is new, create an embedding layer
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
                    # Reuse embedding for feature sets with the same structure (embedding already exists)
                    embedding = feature_embeddings[key]
            embeddings.append(embedding)

        return embeddings


    def set_optimizer(self, optimizer):
        """Set the optimizer after the model is initialised."""
        self._optimizer = optimizer


    def set_scheduler_config(self, scheduler_config):
        """Set the learning rate scheduler config."""
        self._scheduler_config = scheduler_config


    def configure_optimizers(self):
        """Returns the optimizer and the scheduler config."""
        if self._optimizer is None:
            raise RuntimeError('Optimizer not set')
        if self._scheduler_config is None:
            return self._optimizer
        else:
            return {
                'optimizer' : self._optimizer,
                'lr_scheduler': self._scheduler_config,
            }


    def conditioning(
            self,
            hard_data: List[torch.Tensor],
            hard_mask_exist: List[torch.Tensor],
            reco_data: List[torch.Tensor],
            reco_mask_exist: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Computes the conditioning context for the Parallel Transfusion CFM model.

        Args:
            - hard_data (List[torch.Tensor]): Tensor for each type of hard-level particle, each of shape [B, particles, features].
            - hard_mask_exist (List[torch.Tensor]): One mask per hard object type, shape [B, particles].
            - reco_data (List[torch.Tensor]): Tensor for each type of reco-level particle, each of shape [B, particles, features].
            - reco_mask_exist (List[torch.Tensor]): One mask per reco object type, shape [B, particles].

        Returns:
            - condition (torch.Tensor): Conditioning context from Transformer.
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


    def get_bridging_pair(
            self,
            x0: torch.Tensor,
            x1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Default method: Returns the same pair (no Optimal Transport re-pairing).
        Child processes can override this function if they need to re-pair x0 & x1 via OT plans.

        Args:
            - x0 (torch.Tensor): Initial state (from Gaussian distribution).
            - x1 (torch.Tensor): Target state.

        Returns:
            - x0 (torch.Tensor): Initial state.
            - x1 (torch.Tensor): Target state.
        """
        return x0, x1


    def bridging_distribution(
            self,
            x0: torch.Tensor,
            x1: torch.Tensor,
            t: torch.Tensor
    ) -> torch.Tensor:
        """
        Placeholder function for the bridging step. Provides the transformation from the starting state x0
        to the target x1 during interpolation as a function of time t. Subclasses will override.

        Args:
            - x0 (torch.Tensor): Starting state (Gaussian noise).
            - x1 (torch.Tensor): Target state (reco data).
            - t (torch.Tensor): Time tensor.

        Raises:
            - NotImplementedError: Error encouraging subclasses to override with custom function.
        """
        raise NotImplementedError("Child classes need to override this bridging_distribution function.")


    def compute_velocity(
            self,
            context: torch.Tensor,
            group_feature: torch.Tensor,
            t: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the predicted velocity field. Concatenates the Transformer context
        then passes this as an input to the velocity network.

        Args:
            - context (torch.Tensor): Transformer context.
            - group_feature (torch.Tensor): Feature group for reco particle, either scalar [B, 1] or circular [B, 2] (e.g. phi).
            - t (torch.Tensor): Current time value, shape [B] or [B, 1].

        Returns:
            - v_pred (torch.Tensor): The predicted velocity vector, shape [B, sum_reco, len_flow_feats].
        """
        B = context.shape[0]
        if group_feature.shape[1] == 1:
            partial_emb = self.partial_embed_scalar(group_feature)
        elif group_feature.shape[1] == 2:
            partial_emb = self.partial_embed_circular(group_feature)
        else:
            raise ValueError(f"Unexpected group dimension: {group_feature.shape[1]}")
        partial_emb = partial_emb.unsqueeze(1)
        net_in = torch.cat([context, partial_emb, t.view(B, 1, 1)], dim=2)
        net_in = net_in.reshape(B, -1)
        v_pred = self.velocity_net(net_in).reshape(B, 1)

        return v_pred


    def shared_eval(
            self,
            batch: Dict[str, Any],
            batch_idx: int, 
            prefix: str
    ) -> torch.Tensor:
        """
        Evaluation function for training and validation. Takes the error outputted by the forward pass and computes loss functions.
        Calculates the MSE loss between the predicted and the true velocity fields. Also logs losses per object/

        Args:
            - batch (Dict[str, Any]): Batch data containing "hard" and "reco"keys.
            - batch_idx (int): Index of the current batch.
            - prefix (str): "train" or "val", used for logging.

        Returns:
            - loss (torch.Tensor): The total loss for the entire batch.
        """

        diff = self.forward(batch)  # Forward pass computes MSE loss on velocity fields, shape [B, num_obj, effective_count]

        # Overall loss: average over particles and features, then batch
        loss_per_event = diff.mean(dim=(1, 2))  # [B]
        loss = loss_per_event.mean(dim=0)  # Scalar
        self.log(f"{prefix}/velocity_loss_tot", loss, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)

        # Log per-feature loss for each reco particle type
        global_offset = 0  # Track current starting index along the object axis
        for type_idx, num_obj in enumerate(self.n_reco_particles_per_type):
            effective_count = sum(2 if feat == "phi" else 1 for feat in self.flow_input_features[type_idx]) # Determine the  number of columns for this reco type
            diff_type = diff[:, global_offset:global_offset + num_obj, :effective_count] # Extract the slice of diff that corresponds to this reco type

            # Compute error for each field in flow features for this reco type
            local_col = 0  # Tracks column position
            for feat in self.flow_input_features[type_idx]:
                if feat == "phi":
                    # For phi, two columns represent it, compute the mean error over both columns
                    field_slice = diff_type[:, :, local_col:local_col + 2]  # shape: [B, num_obj, 2]
                    field_error = field_slice.mean()  # overall mean error for phi for this type.
                    self.log(f"velocity_loss_field_{self.reco_particle_type_names[type_idx]}_{feat}", field_error, prog_bar=False)
                    local_col += 2
                else:
                    # For non-phi features, take 1 column
                    field_slice = diff_type[:, :, local_col:local_col + 1]  # shape: [B, num_obj, 1]
                    field_error = field_slice.mean()
                    self.log(f"velocity_loss_field_{self.reco_particle_type_names[type_idx]}_{feat}", field_error, prog_bar=False)
                    local_col += 1

            global_offset += num_obj  # Move to the next reco type in the concatenated dimension.

        # Log per-object error, average over features
        object_error = diff.mean(dim=2)  # [B, sum_reco]
        mean_object_error = object_error.mean(dim=0)  # [sum_reco]
        offset = 0
        for type_idx, num_obj in enumerate(self.n_reco_particles_per_type):
            for j in range(num_obj):
                # Compute error for object j of type type_idx (averaged over batch)
                obj_err = mean_object_error[offset + j]
                self.log(f"velocity_loss_object_{self.reco_particle_type_names[type_idx]}_{j}", obj_err, prog_bar=False)
            offset += num_obj

        return loss


    def training_step(self, batch, batch_idx):
        """Performs a single training step for a batch."""
        return self.shared_eval(batch, batch_idx, 'train')


    def validation_step(self, batch, batch_idx):
        """Performs a single validation step for a batch."""
        return self.shared_eval(batch, batch_idx, 'val')


    def pack_reco_features(
            self,
            reco_data: List[torch.Tensor],
            reco_mask: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Organises reco-level particle features into a flattened tensor (with masking). For φ, raw values are replaced
        by sin(φ), cos(φ) components.

        Args:
            - reco_data (List[torch.Tensor]): Tensors for each type of reco particle, each of shape [B, particles, features].
            - reco_mask (List[torch.Tensor]): Existence masks for each type of reco particle, each of shape [B, particles].

        Returns:
            - x_real (torch.Tensor): Flattened tensor containing all reco features, shape [B, total_particles, len_flow_feats].
            - feat_mask (torch.Tensor): Tensor indicating valid features (mask).
            - sum_reco (int): Total number of reco particles across all types.
        """
        B = reco_data[0].shape[0] # Batch size
        n_each_type = [rd.shape[1] for rd in reco_data] # List of num particles for each reco type
        sum_reco = sum(n_each_type) # Total num of reco particles for a given event (across all types)

        # Determine the correct number of output feature columns
        feature_col_count = max(
            sum(2 if feat == "phi" else 1 for feat in self.flow_input_features[type_i])
            for type_i in range(len(self.n_reco_particles_per_type))
        )

        # Initialise tensors
        x_real = torch.zeros((B, sum_reco, feature_col_count), device=reco_data[0].device) # To store feature values
        feat_mask = torch.zeros((B, sum_reco, feature_col_count), device=reco_data[0].device) # Which values are valid

        offset = 0 # Where in the flat tensor current reco type should be placed
        for type_i, n in enumerate(self.n_reco_particles_per_type): # Loop over each reco type
            # n = number of particles of this particle type (e.g. 6 for jets)
            rd_i = reco_data[type_i]      # Contents for that particle type, shape [B, Particles, Features]
            mask_i = reco_mask[type_i]    # Existance mask for the current particle type, shape: [B, Particles]
            flow_feats = self.flow_input_features[type_i]
            reco_feats = self.reco_input_features_per_type[type_i]  # Feature order for this type

            col_offset = 0
            # Loop over each feature in the flow
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
                    x_real[:, offset:offset+n, col_offset] = rd_i[:, :, col_idx] # Assigns feature values
                    feat_mask[:, offset:offset+n, col_offset] = mask_i # Mark valid features
                    col_offset += 1

            # For features not present in this reco type, they remain zero and masked
            offset += n # Move pointer for next reco particle type

        return x_real, feat_mask, sum_reco


    def unpack_reco_samples(
            self,
            x_t: torch.Tensor,
            reco_mask_exist: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Unpacks the flattened x_t tensor back into a list of tensors, one per reco particle type.
        For φ, the sin(φ), cos(φ) components need to be combined back into a single angle in the
        range [-π, π] using `atan2` and normalisation to ensure (sin, cos) pair lies on the unit circle.

        Args:
            - x_t (torch.Tensor): Flattened x_t, shape [B, total_particles, effective_feature_count].
            - reco_mask_exist (List[torch.Tensor]): List, masks for each reco particle type.

        Returns:
            - reco_samples (List[torch.Tensor]): Tensor for each reco particle type, original feature shapes restored.
        """
        reco_samples = [] # Will contain list of reco particle types
        offset = 0 # Track where in x_t the next particle type starts

        # Loop over each particle type
        for type_i, n in enumerate(self.n_reco_particles_per_type):
            flow_feats = self.flow_input_features[type_i] # List of features used for this particle type
            effective_count = sum(2 if feat == "phi" else 1 for feat in flow_feats)
            sample_i = x_t[:, offset:offset+n, :effective_count]
            out_features = []
            col_offset = 0
            for feat in flow_feats:
                if feat == "phi":
                    # Extract sin and cos components
                    sin_phi = sample_i[:, :, col_offset]
                    cos_phi = sample_i[:, :, col_offset+1]
                    # Normalise
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

            out_sample = torch.cat(out_features, dim=-1) # Concatenate features back
            reco_samples.append(out_sample)
            offset += n # Update offset

        return reco_samples


    def velocity_target(
            self,
            x0: torch.Tensor, 
            x1: torch.Tensor, 
            x_t: torch.Tensor, 
            t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the true velocity for training. Calculated by taking the derivative of x_t in the bridging distribution.
        By default, the velocity target is simply (x1 - x0). Child classes can override this to match their bridging distribution's derivative.

        Args:
            - x0 (torch.Tensor): Initial state.
            - x1 (torch.Tensor): Target state.
            - x_t (torch.Tensor): Current state.
            - t (torch.Tensor): Current time.

        Returns:
            - torch.Tensor: Target velocity vector.
        """
        return x1 - x0


    def forward(
            self,
            batch: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Autoregressive forward pass. Each reco particle token  features are updated group-by-group.

        Args:
                - batch (Dict[str, Any]): Batch containing input hard & reco data & masks.
            Returns:
                - diff (torch.Tensor): Squared error for every element, to be later manipulated for logging.
        """
        hard_data = batch["hard"]["data"]
        hard_mask = batch["hard"]["mask"]
        reco_data = batch["reco"]["data"]
        reco_mask = batch["reco"]["mask"]

        # Safety checks as before...
        assert len(hard_data) == len(self.hard_embeddings), \
            f'{len(hard_data)} hard objects but {len(self.hard_embeddings)} hard embeddings'
        assert len(hard_mask) == len(self.hard_embeddings), \
            f'{len(hard_mask)} hard objects but {len(self.hard_embeddings)} hard embeddings'
        assert len(reco_data) == len(self.reco_embeddings), \
            f'{len(reco_data)} reco objects but {len(self.reco_embeddings)} reco embeddings'
        assert len(reco_mask) == len(self.reco_embeddings), \
            f'{len(reco_mask)} reco objects but {len(self.reco_embeddings)} reco embeddings'

        # Obtain conditioning via Transformer.
        transformer_out = self.conditioning(hard_data, hard_mask, reco_data, reco_mask)
        # Remove the null token.
        context_all = transformer_out[:, 1:, :]  # [B, total_particles, embed_dim]
        x_real, feat_mask, total_particles = self.pack_reco_features(reco_data, reco_mask)  # [B, total_particles, len_flow_feats]
        B, _, _ = x_real.shape

        # Build the autoregressive ordering over particles.
        particle_order = []
        for type_idx, num in enumerate(self.n_reco_particles_per_type):
            particle_order.extend([type_idx] * num)

        diff_list = []
        token_idx = 0
        for type_idx in particle_order:
            # For each particle token, get the decoder context.
            context_particle = context_all[:, token_idx:token_idx+1, :]  # [B, 1, embed_dim]
            # For teacher forcing, sample x0 as noise and set x1 as ground-truth.
            x0_particle = torch.randn(B, 1, self.len_flow_feats, device=x_real.device)
            x1_particle = x_real[:, token_idx:token_idx+1, :]  # [B, 1, len_flow_feats]

            # Determine the group structure for this particle.
            groups = [2 if feat=="phi" else 1 for feat in self.flow_input_features[type_idx]]
            particle_diff_groups = []
            group_start = 0

            # For each group in the particle, generate it sequentially.
            for group_size in groups:
                # Re-pair if needed.
                paired_x0, paired_x1 = self.get_bridging_pair(x0_particle, x1_particle)
                x0_group = paired_x0[:, :, group_start:group_start+group_size].squeeze(1)  # [B, group_size]
                x1_group = paired_x1[:, :, group_start:group_start+group_size].squeeze(1)  # [B, group_size]
                # Sample a random time for this group.
                t_val = torch.rand(B, device=x_real.device)
                # Compute the bridging sample *using x0_group and x1_group*.
                x_t_group = self.bridging_distribution(x0_group, x1_group, t_val)  # [B, group_size]
                v_pred = self.compute_velocity(context_particle, x_t_group, t_val)  # [B, 1]
                v_true = self.velocity_target(x0_group, x1_group, x_t_group, t_val)  # [B, 1]
                # Compute squared error and expand to group size.
                group_error = (v_pred - v_true) ** 2  # [B, 1]
                group_error_expanded = group_error.repeat(1, group_size)  # [B, group_size]
                particle_diff_groups.append(group_error_expanded)
                group_start += group_size

            # Concatenate group errors to form the error for the particle.
            particle_diff = torch.cat(particle_diff_groups, dim=1)  # [B, effective_dim]
            effective_dim = sum([2 if feat=="phi" else 1 for feat in self.flow_input_features[type_idx]])
            if effective_dim < self.len_flow_feats:
                pad_width = self.len_flow_feats - effective_dim
                pad_tensor = torch.zeros(B, pad_width, device=x_real.device, dtype=particle_diff.dtype)
                particle_diff = torch.cat([particle_diff, pad_tensor], dim=1)
            diff_list.append(particle_diff.unsqueeze(1))  # [B, 1, len_flow_feats]
            token_idx += 1

        diff = torch.cat(diff_list, dim=1)  # [B, total_particles, len_flow_feats]
        return diff


    def rk4_step(
            self,
            context: torch.Tensor,
            x: torch.Tensor,
            t_val: torch.Tensor,
            dt: float
    ) -> torch.Tensor:
        """
        Performs a single integration step using the RK4 method. Used to integrate the ODE defined by the velocity field.

        Args:
            - context (torch.Tensor): Transformer context used in velocity computation.
            - x (torch.Tensor): Current state.
            - t_val (torch.Tensor): Current time value.
            - dt (float): Time step for the integration.

        Returns:
            torch.Tensor: Updated state after the RK4 integration step.
        """
        k1 = self.compute_velocity(context, x, t_val)
        k2 = self.compute_velocity(context, x + 0.5*dt*k1, t_val + 0.5*dt)
        k3 = self.compute_velocity(context, x + 0.5*dt*k2, t_val + 0.5*dt)
        k4 = self.compute_velocity(context, x +     dt*k3, t_val +     dt)
        return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


    def sample(
            self,
            hard_data: List[torch.Tensor],
            hard_mask_exist: List[torch.Tensor],
            reco_data: List[torch.Tensor],
            reco_mask_exist: List[torch.Tensor],
            N_sample: int = 1,
            steps: int = 10,
            store_trajectories: bool = False
    ) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], torch.Tensor]]:
        """
        Autoregressively generate new samples for all reco particle types.
        Generates the particles & their feature groups sequentially.
        Each feature group is generated conditioned on the groups generated so far for that particle.

        Args:
            - hard_data (List[torch.Tensor]): Hard-level data.
            - hard_mask_exist (List[torch.Tensor]): Existence masks for hard-level particles.
            - reco_data (List[torch.Tensor]): Reco-level data.
            - reco_mask_exist (List[torch.Tensor]): Existence masks for reco-level particles.
            - N_sample (int): Number of samples to generate for each object.
            - steps (int): Number of integration steps to perform.
            - store_trajectories (bool): If True, it stores and returns the intermediate states for plotting on trajecotries plots.

        Returns:
            - samples(List[torch.Tensor]): Tensor of samples for each reco particle type, shape [N_sample, B, particles, features].
            - trajectories (torch.Tensor): Optional, tensor storing intermediate states for samples, shape [N_sample, steps+1, B, total_particles, features].
        """
        samples = []  # Will collect one sample (i.e. a list-of-reco-tensors) per N_sample
        traj_all = [] if store_trajectories else None

        for sample_idx in range(N_sample):
            B = hard_data[0].shape[0]
            device = hard_data[0].device

            # Encoder: Process hard data
            hard_mask_cat = torch.cat(hard_mask_exist, dim=1)
            hard_emb = torch.cat(
                [self.hard_embeddings[i](hard_data[i]) for i in range(len(self.hard_embeddings))],
                dim=1,
            )
            hard_emb = hard_emb * hard_mask_cat[..., None]
            if self.hard_mask_attn is None:
                hard_mask_attn = hard_mask_cat
            else:
                hard_mask_attn = torch.logical_or(self.hard_mask_attn.to(device), hard_mask_cat)
            if hard_mask_attn.dtype != torch.bool:
                hard_mask_attn = hard_mask_attn > 0
            encoder_output = self.transformer.encoder(hard_emb, src_key_padding_mask=hard_mask_attn)

            # Global Autoregressive Initialization
            d_model = self.embed_dim
            null_token = torch.ones((B, 1, d_model), device=device) * -1
            # Global tokens: these are the tokens for complete particles
            generated_tokens = [null_token]  
            particle_states = []  # To store the final state (feature vector) for each particle
            traj_particles_all = [] if store_trajectories else None

            # Determine particle ordering and group specifications.
            particle_order = []
            for type_idx, num in enumerate(self.n_reco_particles_per_type):
                particle_order.extend([type_idx] * num)

            # For each reco type, determine the group sizes.
            group_specs = {}    # Maps type_idx -> list of group sizes.
            effective_dims = {}  # Maps type_idx -> total effective feature dim.
            for type_idx, flow_feats in enumerate(self.flow_input_features):
                groups = [2 if feat == "phi" else 1 for feat in flow_feats]
                group_specs[type_idx] = groups
                effective_dims[type_idx] = sum(groups)

            # Autoregressive Generation Loop
            for type_idx in particle_order:
                # For the current particle, we generate its feature groups sequentially.
                local_group_tokens = []  # Will hold tokens for the groups of this particle.
                group_states = []  # To store the final state of each group.
                groups = group_specs[type_idx]
                # For each feature group:
                for group_size in groups:
                    # Build the current local autoregressive sequence:
                    if local_group_tokens:
                        # Concatenate global tokens with already-generated group tokens.
                        current_seq = torch.cat(generated_tokens + local_group_tokens, dim=1)
                    else:
                        current_seq = torch.cat(generated_tokens, dim=1)
                    tgt_mask = self.transformer.generate_square_subsequent_mask(current_seq.size(1)).to(device)
                    dec_out = self.transformer.decoder(current_seq, encoder_output, tgt_mask=tgt_mask)
                    # Use the last token as the condition for this group.
                    token_context = dec_out[:, -1:, :]  # [B, 1, d_model]

                    # Initialize the group state with random noise.
                    current_group = torch.randn(B, group_size, device=device)
                    dt = 1.0 / steps
                    traj_group = [] if store_trajectories else None
                    for step in range(steps):
                        t_val = torch.full((B,), step * dt, device=device)
                        current_group = self.rk4_step_group(token_context, current_group, t_val, dt)
                        if store_trajectories:
                            traj_group.append(current_group.detach().cpu())
                    if store_trajectories:
                        traj_group = torch.stack(traj_group, dim=0)  # [steps, B, group_size]
                    # Compute a token for this group using the appropriate partial embed.
                    if group_size == 1:
                        group_token = self.partial_embed_scalar(current_group)  # [B, embed_dim]
                    elif group_size == 2:
                        group_token = self.partial_embed_circular(current_group)  # [B, embed_dim]
                    else:
                        raise ValueError("Unexpected group size")
                    group_token = group_token.unsqueeze(1)  # [B, 1, embed_dim]
                    local_group_tokens.append(group_token)
                    group_states.append(current_group)
                # End of groups for this particle.
                # Concatenate group states along the feature axis.
                particle_state = torch.cat(group_states, dim=1)  # [B, effective_dim]
                # Pad if necessary to match self.len_flow_feats.
                if effective_dims[type_idx] < self.len_flow_feats:
                    pad_width = self.len_flow_feats - effective_dims[type_idx]
                    pad_tensor = torch.zeros(B, pad_width, device=device)
                    particle_state = torch.cat([particle_state, pad_tensor], dim=1)
                particle_states.append(particle_state.unsqueeze(1))  # [B, 1, len_flow_feats]
                # Update the global autoregressive sequence: compute a full particle token.
                full_token = self.flow_to_embed(particle_state)  # [B, embed_dim]
                generated_tokens.append(full_token.unsqueeze(1))
                # (Optionally, you can also store the local group trajectories for analysis.)
                if store_trajectories:
                    traj_particles_all.append(local_group_tokens)

            # Repackage Generated Particle States
            # Concatenate particle states: [B, total_particles, len_flow_feats]
            x_t_full = torch.cat(particle_states, dim=1)
            # Use the existing unpacking function to recover per-reco-type tensors.
            reco_samples = self.unpack_reco_samples(x_t_full, reco_mask_exist)
            samples.append(reco_samples)

            # Trajectory repackaging (placeholder)
            if store_trajectories:
                traj_all = None

        # Rearrange samples so that we have one tensor per reco type with shape:
        # [N_sample, B, particles, features]
        if samples and samples[0]:
            N_total = len(samples)
            N_reco = len(samples[0])
            samples_by_type = [
                torch.stack([samples[s][r] for s in range(N_total)], dim=0)
                for r in range(N_reco)
            ]
        else:
            raise RuntimeError("No samples generated. Check sampling logic.")

        if store_trajectories:
            return samples_by_type, traj_all
        else:
            return samples_by_type



class StandardCFM(BaseCFM):
    """
    Child class which uses a standard linear bridging distribution for interpolation:
        x(t) = (1 - t) * x0 + t * x1 + σ * ε

    ε is a random noise variable sampled from ε ~ N(0,1). Adds stochasticity into the bridging distribution,
    helping it stablise in training and generalise to unseen cases.

    σ adjusts the scaling of the noise.
    """
    def bridging_distribution(
            self,
            x0: torch.Tensor,
            x1: torch.Tensor,
            t: torch.Tensor
    ) -> torch.Tensor:
        """
        Linear interpolation between x0 and x1 with added Gaussian noise.

        Args:
            - x0 (torch.Tensor): Initial state (Gaussian noise).
            - x1 (torch.Tensor): Target state.
            - t (torch.Tensor): Time tensor.

        Returns:
            - x_t (torch.Tensor): Intermediate state.
        """
        eps = torch.randn_like(x0) # Random gaussian noise
        t = pad_t_like_x(t, x0) # Expand t if needed to match x1 shape
        xt = (1.0 - t) * x0 + t * x1 + self.sigma * eps # Standard linear bridging

        return xt



class OptimalTransportCFM(BaseCFM):
    """
    Child class for Optimal Transport (OT) CFM method. Uses the OT-methods from `optimal_transport.py`
    to re-pair x0 <-> x1. Overides the bridging function.

    Parameters:
        - sigma (Union[float, int]): Standard deviation for noise in the bridging process.
        - ot_method (str): Method for computing the Optimal Transport plan, options:
            - 'exact': Exact OT by solving the LP associated with the Wasserstein distance.
            - 'sinkhorn': Entropy regularisation to speed up transport plan calculations.
            - 'unbalanced': Relaxes mass conservation, can transport variable mass.
            - 'partial': Transports a fraction of total mass.
        - ot_reg (float): Regularization strength for the OT computation.
        - normalize_cost (bool): Whether or not to normalize the cost matrix before computing OT.
    """
    def __init__(
            self,
            *args: Any,
            sigma: Union[float, int] = 0.0,
            ot_method: str = "exact",
            ot_reg: float = 0.05,
             normalize_cost: bool = False,
            **kwargs: Any,
    ):
        super().__init__(*args, sigma=sigma, **kwargs)
        # Create an OTPlanSampler. Use exact method by default
        self.ot_sampler = OTPlanSampler(method=ot_method, reg=ot_reg, normalize_cost=normalize_cost)


    def get_bridging_pair(
            self,
            x0: torch.Tensor,
            x1: torch.Tensor
     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Re-pair x0 and x1 using a custom OT plan.

        Args:
            x0 (torch.Tensor): Initial noise state.
            x1 (torch.Tensor): Target state.

        Returns:
            paired_x0 (torch.Tensor): Re-paired initial state.
            paired_x0 (torch.Tensor): Re-paired target state.
        """
        paired_x0, paired_x1 = self.ot_sampler.sample_plan(x0, x1, replace=True)

        return paired_x0, paired_x1



class TargetBridgingCFM(BaseCFM):
    """
    A subclass which implements Lipman et al. (ICLR 2023) style bridging. Sets the bridging distribution to:
        x(t) ~ N( t * x1, [1 - (1 - sigma)* t]^2 ).
    It disregards x0 in the bridging step, focusing solely on x1 and time t. Variance shrinks over time.
    """
    def velocity_target(
            self,
            x0: torch.Tensor,
            x1: torch.Tensor,
            x_t: torch.Tensor,
            t: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the target veloctiy for the target bridging function. Bridging is defined exclusively by x1 so:
            v_true = x1 - (1 - sigma) * eps,
        where epsiolon was sampled during the bridging distribution.

        Args:
            - x0 (torch.Tensor): Unused.
            - x1 (torch.Tensor): Target state.
            - x_t (torch.Tensor): Current state.
            - t (torch.Tensor): Current time.

        Returns:
            - v_true (torch.Tensor): New target velocity vector.
        """
        dsigma_dt = -(1.0 - self.sigma)
        v_true = x1 + dsigma_dt * self.last_eps

        return v_true


    def bridging_distribution(
            self,
            x0: torch.Tensor,
            x1: torch.Tensor,
            t: torch.Tensor
    ) -> torch.Tensor:
        """
        Target bridging distribution, relying on x1 exclusively (no dependence on x0):
            x(t) = t * x1 + sigma_t * eps
        where:
            sigma_t = 1 - (1 - sigma) * t.

        Args:
            - x0 (torch.Tensor): Not used.
            - x1 (torch.Tensor): Target state.
            - t (torch.Tensor): Current time.

        Returns:
            - x_t (torch.Tensor): Interpolated state x_t.
        """
        # Ignore x0 here
        t = pad_t_like_x(t, x0)
        self.last_eps = torch.randn_like(x1) # Sample noise and store it in self.last_eps for use in the velocity target.
        sigma_t = 1.0 - (1.0 - self.sigma) * t # Compute time-dependent std
        x_t = t * x1 + sigma_t * self.last_eps # Bridging distribution

        return x_t



class SchrodingerBridgeCFM(BaseCFM):
    """
    Child class for Schrödinger bridge CFM. Uses entropy-regularised version of OT where the interpolation from x0 to x1
    follows a stochastic path governed by a Schrödinger-type diffusion process. Introduces time varying noise.

    Parameters:
        - sigma (Union[float, int]): Noise standard deviation.
        - ot_method (str): Method for computing the Optimal Transport plan, options:
            - 'exact': Exact OT by solving the LP associated with the Wasserstein distance.
            - 'sinkhorn': Entropy regularisation to speed up transport plan calculations.
            - 'unbalanced': Relaxes mass conservation, can transport variable mass.
            - 'partial': Transports a fraction of total mass.
    """
    def __init__(
            self,
            *args: Any,
            sigma: Union[float, int] = 0.1,
            ot_method: str = 'exact',
            **kwargs: Any,
    ) -> None:
        super().__init__(*args, sigma=sigma, **kwargs)

        if sigma <= 0:
            raise ValueError(f"Sigma must be strictly positive, got {sigma}.")
        elif sigma < 1e-3:
            warnings.warn("Small sigma values may lead to numerical instability.")

        self.ot_method = ot_method
        self.ot_sampler = OTPlanSampler(method=ot_method, reg=2 * self.sigma**2)


    def get_bridging_pair(
            self,
            x0: torch.Tensor,
            x1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Re-pair x0 and x1 using a custom OT plan.

        Args:
            x0 (torch.Tensor): Initial noise state.
            x1 (torch.Tensor): Target state.

        Returns:
            paired_x0 (torch.Tensor): Re-paired initial state.
            paired_x0 (torch.Tensor): Re-paired target state.
        """
        paired_x0, paired_x1 = self.ot_sampler.sample_plan(x0, x1)

        return paired_x0, paired_x1


    def velocity_target(
            self,
            x0: torch.Tensor,
            x1: torch.Tensor,
            x_t: torch.Tensor,
            t: torch.Tensor
    ) -> torch.Tensor:
        """
        Target velocity for the Schrödinger Bridge method. Combines the derivative of the deterministic part (x1 - x0) with the derivative
        of the noise term, adjusted for the time-dependent variance.

        Args:
            - x0 (torch.Tensor): Initial state.
            - x1 (torch.Tensor): Target state.
            - x_t (torch.Tensor): Current state.
            - t (torch.Tensor): Current time.

        Returns:
            - v_true (torch.Tensor): True velocity vector.
        """
        t = pad_t_like_x(t, x0)
        # Compute the derivative of the deterministic part
        v_det = x1 - x0
        # Compute the derivative of the noise term, adding a small epsilon for stability
        denom = 2 * torch.sqrt(t * (1.0 - t) + 1e-6)
        v_noise = self.sigma * self.last_eps * (1.0 - 2.0 * t) / denom
        v_true = v_det + v_noise
        return v_true

    def bridging_distribution(
            self,
            x0: torch.Tensor,
            x1: torch.Tensor,
            t: torch.Tensor
    ) -> torch.Tensor:
        """
        Bridging distribution for the Schrödinger Bridge method.
        The mean is a linear interpolation between x0 and x1, and the variance scales as:
            sqrt(t*(1-t))*sigma.

        Args:
            - x0 (torch.Tensor): Initial state.
            - x1 (torch.Tensor): Target state.
            - t (torch.Tensor): Current time.

        Returns:
            - x_t (torch.Tensor): Intermediate state.
        """

        t = pad_t_like_x(t, x0)
        self.last_eps = torch.randn_like(x0)
        mu_t = (1.0 - t) * x0 + t * x1
        sigma_t = torch.sqrt(t * (1.0 - t)) * self.sigma
        x_t = mu_t + sigma_t * self.last_eps
        return x_t


class VariancePreservingCFM(BaseCFM):
    """
    Variance preserving bridging method using trigonometric interpolants:
        x(t) = cos((pi/2)*t) * x0 + sin((pi/2)*t) * x1 + sigma * eps.
    Ensures the variance of the interpolated distribution is preserved throughout time, following:
        Stochastic Interpolants: A Unifying Framework for Flows and Diffusions, Albergo et al.
    """

    def velocity_target(
            self,
            x0: torch.Tensor,
            x1: torch.Tensor,
            x_t: torch.Tensor,
            t: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the true velocity for the variance preserving bridging:
            d/dt[x(t)] = - (pi/2) * sin((pi/2)*t) * x0 + (pi/2) * cos((pi/2)*t) * x1.

        Args:
            - x0 (torch.Tensor): Initial state.
            - x1 (torch.Tensor): Target state.
            - x_t (torch.Tensor): Current state.
            - t (torch.Tensor): Current time.

        Returns:
            - v_true (torch.Tensor): True velocity vector.
        """
        t = pad_t_like_x(t, x0)  # Ensure t has the proper shape.
        d_cos_dt = - (math.pi / 2) * torch.sin((math.pi / 2) * t)
        d_sin_dt = (math.pi / 2) * torch.cos((math.pi / 2) * t)
        v_true = d_cos_dt * x0 + d_sin_dt * x1

        return v_true


    def bridging_distribution(
            self,
            x0: torch.Tensor,
            x1: torch.Tensor,
            t: torch.Tensor
    ) -> torch.Tensor:
        """
        Variance preserving bridging distribution using trigonometric functions.

        Args:
            x0 (torch.Tensor): Initial state.
            x1 (torch.Tensor): Target state.
            t (torch.Tensor): Current time.

        Returns:
            - x_t (torch.Tensor): Intermediate state.
        """
        t = pad_t_like_x(t, x0)
        eps = torch.randn_like(x0)
        cos_part = torch.cos((math.pi / 2) * t)
        sin_part = torch.sin((math.pi / 2) * t)
        x_t = cos_part * x0 + sin_part * x1 + self.sigma * eps

        return x_t