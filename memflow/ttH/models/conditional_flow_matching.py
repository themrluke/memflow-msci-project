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



# Just a convenience function as in your original code
def pad_t_like_x(t, x):
    """
    Utility that reshapes (batch,) → (batch, 1, ..., 1) so that
    t can broadcast with x of shape (B, P, F).
    If t has shape (B,) originally, we expand it to (B,1,1).
    """
    while t.dim() < x.dim():
        t = t.unsqueeze(-1)
    return t


class TransferCFM(L.LightningModule):
    """
    A fully 'global' Conditional Flow-Matching model, but now with a
    `conditioning()` function adapted from TransferFlow.
    """

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
        dropout=0.0,
        process_names=None,
        transformer_args=None,
        sigma=0.1,
        optimizer=None,
        scheduler_config=None,
        onehot_encoding=False,        # <--- If you want one-hot, set True
    ):
        super().__init__()
        if transformer_args is None:
            transformer_args = {}

        # Store config
        self.dropout = dropout
        self.embed_dims = embed_dims if isinstance(embed_dims, list) else [embed_dims]
        self.embed_dim = self.embed_dims[-1]
        self.embed_act = embed_act

        self.n_hard_particles_per_type = n_hard_particles_per_type
        self.hard_input_features_per_type = hard_input_features_per_type
        self.hard_particle_type_names = hard_particle_type_names

        self.n_reco_particles_per_type = n_reco_particles_per_type
        self.reco_input_features_per_type = reco_input_features_per_type
        self.reco_particle_type_names = reco_particle_type_names

        self.flow_input_features = flow_input_features
        self.len_flow_feats = max(len(flow_feats) for flow_feats in self.flow_input_features)
        self.process_names = process_names

        self.sigma = sigma
        self._optimizer = optimizer
        self._scheduler_config = scheduler_config

        self.onehot_encoding = onehot_encoding
        if self.onehot_encoding:
            # Initialize one-hot tensors for each reco type
            # Assuming each reco type has a known number of categories
            # For illustration, assuming two reco types with 3 and 2 categories respectively
            # Adjust accordingly based on your reco types
            self.onehot_tensors = nn.ParameterList([
                nn.Parameter(torch.eye(len(features)), requires_grad=False) for features in self.flow_input_features
            ])

        # Handle attention masks
        if reco_mask_attn is None:
            self.reco_mask_attn = None
            print("No reco attention mask provided; will use existence mask only.")
        else:
            # Add a single True for the null token
            self.reco_mask_attn = torch.cat((torch.tensor([True]), reco_mask_attn), dim=0)
        if hard_mask_attn is None:
            print("No hard attention mask provided; will use existence mask only.")
        self.hard_mask_attn = hard_mask_attn

        # Build embeddings
        self.hard_embeddings = self.make_embeddings(self.hard_input_features_per_type)
        self.reco_embeddings = self.make_embeddings(self.reco_input_features_per_type)

        # Build Transformer
        transformer_args = dict(transformer_args)
        transformer_args["d_model"] = self.embed_dim
        if "dropout" not in transformer_args:
            transformer_args["dropout"] = self.dropout

        self.transformer = nn.Transformer(batch_first=True, **transformer_args)

        # Here we assume sum(n_reco_particles_per_type)+1 is the same for all events
        # If that’s guaranteed, you can create one global tgt_mask:
        self.max_reco_len = sum(self.n_reco_particles_per_type) + 1
        self.tgt_mask = self.transformer.generate_square_subsequent_mask(self.max_reco_len)

        # Velocity net for bridging
        d_in = self.embed_dim + self.len_flow_feats + 1  # [context + x(t) + t]

        d_hid = 128
        self.velocity_net = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.SiLU(),
            nn.Linear(d_hid, d_hid),
            nn.SiLU(),
            nn.Linear(d_hid, self.len_flow_feats),
        )

        self.flow_indices = []
        self.global_flow_features = []
        for i,(n,reco_features,flow_features) in enumerate(zip(self.n_reco_particles_per_type,self.reco_input_features_per_type,self.flow_input_features)):
            assert len(set(flow_features).intersection(set(reco_features))) == len(flow_features), f'Not all flow features {flow_features} found in reco_features for particle set #{i} ({reco_features})'
            indices = [reco_features.index(feature) for feature in flow_features]
            self.flow_indices.append(indices)
            self.global_flow_features.extend([feat for feat in flow_features if feat not in self.global_flow_features])


    def make_embeddings(self, input_features_per_type):
        """
        Build an MLP embedding for each type's raw features.
        The final dimension = self.embed_dim.
        """
        embs = nn.ModuleList()
        for feat_list in input_features_per_type:
            layers = []
            for i in range(len(self.embed_dims)):
                in_dim = len(feat_list) if i == 0 else self.embed_dims[i-1]
                out_dim = self.embed_dims[i]
                layers.append(nn.Linear(in_dim, out_dim))
                if i < len(self.embed_dims) - 1 and self.embed_act is not None:
                    layers.append(self.embed_act())
                if self.dropout > 0.0:
                    layers.append(nn.Dropout(self.dropout))
            embs.append(nn.Sequential(*layers))
        return embs


    def set_optimizer(self, optimizer):
        self._optimizer = optimizer


    def set_scheduler_config(self, scheduler_config):
        self._scheduler_config = scheduler_config


    def configure_optimizers(self):
        if self._optimizer is None:
            opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        else:
            opt = self._optimizer

        if self._scheduler_config is None:
            return opt
        else:
            return {
                "optimizer": opt,
                "lr_scheduler": self._scheduler_config,
            }


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

        # # Split condition per particle type to match reco_data segmentation #
        # slices = np.r_[0,np.array(self.n_reco_particles_per_type).cumsum()]
        # conditions = [condition[:,ni:nf,:] for ni,nf in zip(slices[:-1],slices[1:])]

        return condition


    def bridging_distribution(self, x0, x1, t):
        """
        x(t) = (1 - t)* x0 + t * x1 + sigma * eps.
        Shape: x0,x1 are (B, P, F). We broadcast t => shape (B,1,1).
        """
        eps = torch.randn_like(x0)
        t_ = pad_t_like_x(t, x0)
        bridging = (1.0 - t_) * x0 + t_ * x1 + self.sigma * eps

        return bridging


    def compute_velocity(self, context, x_t, t_):
        """
        Compute velocity using the velocity_net.

        Args:
            context: Tensor of shape [B, sum_reco, embed_dim]
            x_t: Tensor of shape [B, sum_reco, len_flow_feats]
            t_: Scalar or tensor representing the current time step

        Returns:
            v_t: Tensor of shape [B, sum_reco, len_flow_feats]
        """
        B, sum_reco_tokens, _ = context.shape
        # Prepare net_in
        net_in = torch.cat([
            context.reshape(B * sum_reco_tokens, -1),          # [B*sum_reco, embed_dim]
            x_t.reshape(B * sum_reco_tokens, self.len_flow_feats), # [B*sum_reco, len_flow_feats]
            t_.repeat_interleave(sum_reco_tokens).unsqueeze(-1),# [B*sum_reco, 1]
        ], dim=1)  # [B*sum_reco, embed_dim + len_flow_feats + 1]

        # Predict velocity
        v_pred = self.velocity_net(net_in).reshape(B, sum_reco_tokens, self.len_flow_feats)  # [B, sum_reco, len_flow_feats]

        return v_pred


    def cfm_loss(self, batch):
        """
        Compute the Conditional Flow-Matching loss.

        Steps:
        1) Transformer => shape (B, sum_reco+1, embed_dim)
        2) Remove null token => shape (B, sum_reco, embed_dim)
        3) Pack reco features => x_real: [B, sum_reco, len_flow_feats], feat_mask: [B, sum_reco, len_flow_feats]
        4) Initialize bridging distribution: x0, x1, t
        5) Compute velocity_net input and predict velocity
        6) Euler stepping
        7) Unpack samples and compute loss
        """
        hard_data = batch["hard"]["data"]
        hard_mask = batch["hard"]["mask"]
        reco_data = batch["reco"]["data"]
        reco_mask = batch["reco"]["mask"]

        # 1. Transformer Output
        transformer_out = self.conditioning(hard_data, hard_mask, reco_data, reco_mask)  # [B, sum_reco+1, embed_dim]

        # 2. Remove Null Token
        context = transformer_out[:, 1:, :]  # [B, sum_reco, embed_dim]

        # 3. Pack Reco Features
        x_real, feat_mask, sum_reco = self.pack_reco_features(reco_data, reco_mask)  # [B, sum_reco, len_flow_feats]
        B, sum_reco_tokens, len_flow_feats = x_real.shape

        # 4. Initialize Bridging Distribution
        x0 = torch.randn_like(x_real)  # [B, sum_reco, len_flow_feats]
        x1 = x_real
        t = torch.rand(B, device=x_real.device)  # [B]

        x_t = self.bridging_distribution(x0, x1, t)  # [B, sum_reco, len_flow_feats]

        # 5. Compute Velocity
        v_pred = self.compute_velocity(context, x_t, t)  # [B, sum_reco, len_flow_feats]

        # 6. Compute True Velocity
        v_true = x1 - x0  # [B, sum_reco, len_flow_feats]

        # 7. Compute Loss
        diff = (v_pred - v_true) ** 2 * feat_mask  # [B, sum_reco, len_flow_feats]
        loss_per_event = diff.mean(dim=(1, 2))  # [B]
        loss = loss_per_event.mean(dim=0)  # Scalar

        return loss


    def pack_reco_features(self, reco_data, reco_mask):
        """
        Pack reco features into a flat tensor with masking.

        Args:
            reco_data: List of tensors, one per reco type, shape [B, P_j, F_j]
            reco_mask: List of tensors, one per reco type, shape [B, P_j]

        Returns:
            x_real: Tensor of shape [B, sum_reco, len_flow_feats]
            feat_mask: Tensor of shape [B, sum_reco, len_flow_feats]
            sum_reco: Total number of reco particles
        """
        B = reco_data[0].shape[0]
        n_tokens_each = [rd.shape[1] for rd in reco_data]
        sum_reco = sum(n_tokens_each)

        # Flattened list of all flow features
        flow_features_flat = self.flow_input_features  # List of lists
        len_flow_feats = self.len_flow_feats  # Maximum number of flow features across reco types

        # Initialize tensors
        x_real = torch.zeros((B, sum_reco, len_flow_feats), device=reco_data[0].device)
        feat_mask = torch.zeros((B, sum_reco, len_flow_feats), device=reco_data[0].device)

        offset = 0
        for type_i, n in enumerate(self.n_reco_particles_per_type):
            rd_i = reco_data[type_i]      # shape: [B, P_j, F_j]
            mask_i = reco_mask[type_i]    # shape: [B, P_j]
            feat_list_i = self.flow_input_features[type_i]  # e.g., ["pt", "phi"]

            for feat_j, feat_name in enumerate(self.flow_input_features[type_i]):
                if feat_name in feat_list_i:
                    col_idx = feat_list_i.index(feat_name)
                    # Assign the feature value to the corresponding position
                    # Since flow_input_features is per reco type, map to global position
                    # Here, assuming len_flow_feats >= len(feat_list_i)
                    x_real[:, offset:offset + n, feat_j] = rd_i[:, :, col_idx]
                    feat_mask[:, offset:offset + n, feat_j] = mask_i

            # For features not present in this reco type, they remain zero and masked
            offset += n

        return x_real, feat_mask, sum_reco

    def unpack_reco_samples(self, x_t, reco_mask_exist):
        """
        Unpack the flat x_t tensor into per reco type tensors.

        Args:
            x_t: Tensor of shape [B, sum_reco, len_flow_feats]
            reco_mask_exist: List of tensors, one per reco type, shape [B, P_j]

        Returns:
            reco_samples: List of tensors, one per reco type, shape [B, P_j, F_j]
        """
        reco_samples = []
        offset = 0
        for type_i, n in enumerate(self.n_reco_particles_per_type):
            P_j = n
            flow_features = self.flow_input_features[type_i]
            F_j = len(flow_features)

            # Extract features for this reco type
            sample_i = x_t[:, offset:offset + P_j, :F_j]  # [B, P_j, F_j]
            reco_samples.append(sample_i)

            # Update offset
            offset += P_j

        return reco_samples


    def forward(self, batch):
        return self.cfm_loss(batch)


    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss


    def sample(self, hard_data, hard_mask_exist, reco_data, reco_mask_exist, N_sample=1, steps=10):
        """
        Generate N_sample independent samples.

        Args:
            hard_data: List of tensors, one per hard particle type, shape [B, P_i, F]
            hard_mask_exist: List of tensors, one per hard particle type, shape [B, P_i]
            reco_data: List of tensors, one per reco particle type, shape [B, P_j, F_j]
            reco_mask_exist: List of tensors, one per reco particle type, shape [B, P_j]
            N_sample: Number of samples to generate
            steps: Number of Euler steps for bridging

        Returns:
            samples: List of tensors, one per reco type, shape [N_sample, B, P_j, F_j]
        """
        N_reco = len(reco_data)
        B = reco_data[0].shape[0]
        samples = [ [] for _ in range(N_reco) ]  # Initialize list for each reco type

        for s in range(N_sample):
            # 1. Obtain Conditioning
            conditions = self.conditioning(hard_data, hard_mask_exist, reco_data, reco_mask_exist)  # [B, sum_reco+1, embed_dim]

            # 2. Remove Null Token
            context = conditions[:, 1:, :]  # [B, sum_reco, embed_dim]

            # 3. Pack Reco Features
            x_real, feat_mask, sum_reco = self.pack_reco_features(reco_data, reco_mask_exist)  # [B, sum_reco, len_flow_feats]
            B, sum_reco_tokens, len_flow_feats = x_real.shape

            # 4. Initialize Bridging Distribution
            x0 = torch.randn_like(x_real)  # [B, sum_reco, len_flow_feats]
            x1 = x_real
            t = torch.rand(B, device=x_real.device)  # [B]

            x_t = self.bridging_distribution(x0, x1, t)  # [B, sum_reco, len_flow_feats]

            # 5. Compute Velocity
            v_pred = self.compute_velocity(context, x_t, t)  # [B, sum_reco, len_flow_feats]

            # 6. Euler Stepping for Bridging
            dt = 1.0 / steps
            for step_i in range(steps):
                t_value = step_i * dt
                t_ = torch.full((B,), t_value, device=x_t.device)  # [B]
                v_t = self.compute_velocity(context, x_t, t_)  # [B, sum_reco, len_flow_feats]
                x_t = x_t + dt * v_t  # [B, sum_reco, len_flow_feats]

            # 7. Unpack Samples per Reco Type
            reco_samples = self.unpack_reco_samples(x_t, reco_mask_exist)  # List of [B, P_j, F_j]

            # 8. Append Samples
            for i in range(N_reco):
                samples[i].append(reco_samples[i])  # Append [B, P_j, F_j] to list for reco type i

        # 9. Stack Samples per Reco Type
        samples = [torch.stack(s, dim=0) for s in samples]  # List of [N_sample, B, P_j, F_j]

        return samples