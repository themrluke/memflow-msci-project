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
        self.process_names = process_names

        self.sigma = sigma
        self._optimizer = optimizer
        self._scheduler_config = scheduler_config

        self.onehot_encoding = onehot_encoding
        # If you do want actual one-hot vectors, you can build them here:
        # (Below is just an illustration; you’d need the sum of reco dims, etc.)
        # self.onehot_tensors = [...]
        # <You can implement exactly as in TransferFlow if needed.>

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
        d_in = self.embed_dim + len(self.flow_input_features) + 1  # [context + x(t) + t]

        d_hid = 128
        self.velocity_net = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.SiLU(),
            nn.Linear(d_hid, d_hid),
            nn.SiLU(),
            nn.Linear(d_hid, len(self.flow_input_features)),
        )


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

    def cfm_loss(self, batch):
        """
        1) Transformer => shape (B, sum_reco+1, embed_dim)
        2) Flatten all reco tokens (except the null) => shape (B, sum_reco, embed_dim)
        3) Gather only 'flow_input_features' from each reco token => bridging from prior => real
        4) velocity_net => MSE vs. target velocity
        5) If some features don't exist in a token type, mask them out.
        6) Optionally log per-process or other metrics.
        """
        hard_data = batch["hard"]["data"]
        hard_mask = batch["hard"]["mask"]
        reco_data = batch["reco"]["data"]
        reco_mask = batch["reco"]["mask"]

        # 1) Transformer => shape (B, sum_reco+1, embed_dim)
        transformer_out = self.conditioning(hard_data, hard_mask, reco_data, reco_mask)

        # 2) The 1st token in the decoder is "null"; skip it
        context = transformer_out[:, 1:, :]

        # print(f"DEBUG: context: shape {context.shape}, min {context.min()}, max {context.max()}")
        # if torch.isnan(context).any():
        #     print("NaNs detected in context")
        # if torch.isinf(context).any():
        #     print("Infs detected in context")

            # 3) Build the bridging distribution
        x_real, feat_mask, sum_reco = self.pack_reco_features(reco_data, reco_mask)
        B, sum_reco_tokens, len_flow_feats = x_real.shape
        # if torch.isnan(x_real).any():
        #     print("NaNs detected in x_real")
        # if torch.isinf(x_real).any():
        #     print("Infs detected in x_real")

        x0 = torch.randn_like(x_real)
        x1 = x_real
        t = torch.rand(B, device=x_real.device)
        x_t = self.bridging_distribution(x0, x1, t)
        # if torch.isnan(x_t).any():
        #     print("NaNs detected in x_t")
        # if torch.isinf(x_t).any():
        #     print("Infs detected in x_t")

        # 4) velocity net input = [context, x_t, t]
        # print(f"DEBUG: embed_dim = {self.embed_dim}")
        # print(f"DEBUG: len_flow_feats = {len_flow_feats}")
        # print(f"DEBUG: x_t shape = {x_t.shape}")  # Expected: (B, sum_reco, len_flow_feats)

        net_in = torch.cat([
            context.reshape(B * sum_reco_tokens, -1),
            x_t.reshape(B * sum_reco_tokens, len_flow_feats),
            t.repeat_interleave(sum_reco_tokens).unsqueeze(-1),
        ], dim=1)
        # print(f"DEBUG: net_in: shape {net_in.shape}, min {net_in.min()}, max {net_in.max()}")
        # if torch.isnan(net_in).any():
        #     print("NaNs detected in net_in")
        # if torch.isinf(net_in).any():
        #     print("Infs detected in net_in")

        v_pred = self.velocity_net(net_in).reshape(B, sum_reco_tokens, len_flow_feats)
        # print(f"DEBUG: v_pred: shape {v_pred.shape}, min {v_pred.min()}, max {v_pred.max()}")
        # if torch.isnan(v_pred).any():
        #     print("NaNs detected in v_pred")
        # if torch.isinf(v_pred).any():
        #     print("Infs detected in v_pred")

        v_true = x1 - x0
        diff = (v_pred - v_true)**2 * feat_mask

        loss_per_event = diff.mean(dim=(1,2))
        loss = loss_per_event.mean(dim=0)

        return loss


    def pack_reco_features(self, reco_data, reco_mask):
        B = reco_data[0].shape[0]
        n_tokens_each = [rd.shape[1] for rd in reco_data]
        sum_reco = sum(n_tokens_each)

        # Flattened list of all flow features
        flow_features_flat = self.flow_input_features  # ["pt", "eta", "phi"]
        len_flow_feats = len(flow_features_flat)  # 3

        # Initialize tensors
        x_real = torch.zeros((B, sum_reco, len_flow_feats), device=reco_data[0].device)
        feat_mask = torch.zeros((B, sum_reco, len_flow_feats), device=reco_data[0].device)

        offset = 0
        for type_i in range(len(self.n_reco_particles_per_type)):
            rd_i = reco_data[type_i]      # shape: (B, n_i, F_i)
            mask_i = reco_mask[type_i]    # shape: (B, n_i)
            feat_list_i = self.reco_input_features_per_type[type_i]  # e.g., ["pt", "phi"]

            for feat_k, feat_name in enumerate(flow_features_flat):
                if feat_name in feat_list_i:
                    col_idx = feat_list_i.index(feat_name)
                    x_real[:, offset:offset + rd_i.shape[1], feat_k] = rd_i[:, :, col_idx]
                    feat_mask[:, offset:offset + rd_i.shape[1], feat_k] = mask_i
                else:
                    # Feature not present for this reco type; leave as zero and mask=0
                    pass

            offset += rd_i.shape[1]

        return x_real, feat_mask, sum_reco



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

    def compute_velocity(self, context, x_t, t_):
        B, sum_reco, F = x_t.shape
        net_in = torch.cat([
            context.reshape(B * sum_reco, -1),
            x_t.reshape(B * sum_reco, F),
            torch.full((B * sum_reco, 1), t_, device=x_t.device),
        ], dim=1)
        v_flat = self.velocity_net(net_in)
        return v_flat.view(B, sum_reco, F)

    def sample(self, batch, steps=10):
        """
        Euler stepping from x(0)=N(0,1) => x(1)=reco features,
        using your velocity field.
        """
        hard_data = batch["hard"]["data"]
        hard_mask = batch["hard"]["mask"]
        reco_data = batch["reco"]["data"]
        reco_mask = batch["reco"]["mask"]

        transformer_out = self.conditioning(hard_data, hard_mask, reco_data, reco_mask)
        context = transformer_out[:, 1:, :]

        x_real, feat_mask, sum_reco = self.pack_reco_features(reco_data, reco_mask)
        B, sum_reco_tokens, len_flow_feats = x_real.shape
        x_t = torch.randn_like(x_real)

        dt = 1.0 / steps
        for step_i in range(steps):
            t_ = step_i * dt
            v_t = self.compute_velocity(context, x_t, t_)
            x_t = x_t + dt * v_t

        return x_t