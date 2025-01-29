import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import math
import numpy as np

from typing import Optional, Dict, Any, List
from zuko.distributions import DiagNormal

from memflow.models.utils import lowercase_recursive

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L

def pad_t_like_x(t, x):
    """
    Utility that reshapes (batch,) → (batch, 1, ..., 1) so that
    t can broadcast with x of shape (B, P, F) as needed.
    If t has shape (B,) originally, we expand it to (B,1,1).
    """
    while t.dim() < x.dim():
        t = t.unsqueeze(-1)
    return t

class TransferCFM(L.LightningModule):
    """
    A fully 'global' Conditional Flow-Matching model:
      - The entire set of hard particles is the Transformer-encoder input.
      - The entire set of reco particles is the Transformer-decoder input.
      - We add exactly one global null token at the start of the reco sequence.
      - We define a single bridging distribution from a simple prior (e.g. Normal(0,1))
        to the real reco features, conditioned on the transformer's output.
      - We have ONE velocity net that is used across all reco tokens (no 1:1 bridging).
    """

    def __init__(
        self,
        # Original TransferFlow-like arguments:
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
        process_names = None,
        transformer_args=None,
        sigma=0.1,
        optimizer=None,
        scheduler_config=None,
    ):
        super().__init__()

        if transformer_args is None:
            transformer_args = {}

        # Store config:
        self.dropout = dropout
        self.embed_dims = embed_dims if isinstance(embed_dims, list) else [embed_dims]
        self.embed_dim = self.embed_dims[-1]
        self.embed_act = embed_act

        self.n_hard_particles_per_type = n_hard_particles_per_type
        self.hard_particle_type_names = lowercase_recursive(hard_particle_type_names)
        self.hard_input_features_per_type = lowercase_recursive(hard_input_features_per_type)

        self.n_reco_particles_per_type = n_reco_particles_per_type
        self.reco_particle_type_names = lowercase_recursive(reco_particle_type_names)
        self.reco_input_features_per_type = lowercase_recursive(reco_input_features_per_type)
        self.flow_input_features = lowercase_recursive(flow_input_features)
        self.process_names = process_names

        self.sigma = sigma  # bridging-dist std
        self._optimizer = optimizer
        self._scheduler_config = scheduler_config

        if reco_mask_attn is None:
            self.reco_mask_attn = None
            print ('No reco attention mask provided, will use the exist mask for the attention')
        else:
            self.reco_mask_attn = torch.cat((torch.tensor([True]),reco_mask_attn),dim=0) # Adding True at index=0 for null token
        if hard_mask_attn is None:
            print ('No hard attention mask provided, will use the exist mask for the attention')
        self.hard_mask_attn  = hard_mask_attn

        # Build embeddings for Hard and Reco
        self.hard_embeddings = self.make_embeddings(self.hard_input_features_per_type)
        self.reco_embeddings = self.make_embeddings(self.reco_input_features_per_type)

        # Build the Transformer
        transformer_args = dict(transformer_args)  # shallow copy
        transformer_args["d_model"] = self.embed_dim
        if "dropout" not in transformer_args:
            transformer_args["dropout"] = self.dropout

        self.transformer = nn.Transformer(batch_first=True, **transformer_args)
        max_reco_len = sum(self.n_reco_particles_per_type) + 1  # +1 for a single global null token
        self.tgt_mask = self.transformer.generate_square_subsequent_mask(max_reco_len)
        print(f"Max Reco Length: {max_reco_len}")
        print(f"Target Mask Shape: {self.tgt_mask.shape}")

        # A single velocity net for bridging the chosen features
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

    #def conditioning(self, hard_data, hard_mask, reco_data, reco_mask):
        """
        1) Encode all hard data => [B, sum_hard, embed_dim]
        2) Decode all reco data (+1 null token) => [B, sum_reco+1, embed_dim]
        3) Return the transformer's output (same shape).
        4) We also apply optional attention masks (self.hard_mask_attn, self.reco_mask_attn).
        """
        B = hard_data[0].shape[0]

        # (a) Hard side embed + concat
        hard_embed_list = []
        hard_mask_list = []
        for emb, x, m in zip(self.hard_embeddings, hard_data, hard_mask):
            e = emb(x) * m.unsqueeze(-1)  # zero out if missing
            hard_embed_list.append(e)
            hard_mask_list.append(m)
        hard_embed_cat = torch.cat(hard_embed_list, dim=1)  # (B, sum_hard, embed_dim)
        hard_mask_cat  = torch.cat(hard_mask_list, dim=1)  # (B, sum_hard)

        # (b) Reco side embed + 1 global null token
        reco_embed_list = []
        reco_mask_list = []
        for emb, x, m in zip(self.reco_embeddings, reco_data, reco_mask):
            e = emb(x) * m.unsqueeze(-1)
            reco_embed_list.append(e)
            reco_mask_list.append(m)
        reco_embed_cat = torch.cat(reco_embed_list, dim=1)  # (B, sum_reco, embed_dim)
        reco_mask_cat  = torch.cat(reco_mask_list, dim=1)  # (B, sum_reco)

        null_token = torch.full((B, 1, self.embed_dim), -1.0,
                                device=hard_embed_cat.device, dtype=hard_embed_cat.dtype)
        null_mask  = torch.ones(B, 1, device=hard_embed_cat.device, dtype=reco_mask_cat.dtype)

        reco_embed_cat = torch.cat([null_token, reco_embed_cat], dim=1)  # shape (B, sum_reco+1, embed_dim)
        reco_mask_cat  = torch.cat([null_mask,  reco_mask_cat],  dim=1)  # shape (B, sum_reco+1)

        # (c) Convert to key_padding_mask => True=ignore
        # If mask=1 => particle exists => key_padding=False => ~mask.bool()
        if self.hard_mask_attn is None:
            hard_key_pad = ~hard_mask_cat.bool()
        else:
            # Combine user-supplied attention mask with existence mask
            # E.g. if self.hard_mask_attn is shape [sum_hard] or [1, sum_hard],
            # you might need to tile or ensure the shapes match. Here we assume
            # self.hard_mask_attn is shape (sum_hard,) => broadcast.
            # Then final => True=ignore => OR => means "any are True => ignore"
            # NOTE: You might want to invert your self.hard_mask_attn if you stored it differently.
            # Adjust as needed:
            _mask = self.hard_mask_attn.to(hard_mask_cat.device)  # (sum_hard,) or (1,sum_hard)
            # broadcast to (B,sum_hard):
            while _mask.dim() < hard_mask_cat.dim():
                _mask = _mask.unsqueeze(0)
            hard_key_pad = torch.logical_or(~hard_mask_cat.bool(), _mask.bool())

        if self.reco_mask_attn is None:
            reco_key_pad = ~reco_mask_cat.bool()
        else:
            _mask = self.reco_mask_attn.to(reco_mask_cat.device)
            while _mask.dim() < reco_mask_cat.dim():
                _mask = _mask.unsqueeze(0)
            reco_key_pad = torch.logical_or(~reco_mask_cat.bool(), _mask.bool())

        tgt_seq_len = reco_embed_cat.size(1)  # This is sum_reco+1 for *this batch*
        dynamic_tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len)

        # (d) run Transformer
        out = self.transformer(
            src=hard_embed_cat,
            tgt=reco_embed_cat,
            src_key_padding_mask=hard_key_pad,
            tgt_key_padding_mask=reco_key_pad,
            memory_key_padding_mask=hard_key_pad,
            tgt_mask=dynamic_tgt_mask.to(reco_embed_cat.device),
        )
        return out

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

    def bridging_distribution(self, x0, x1, t):
        """
        x(t) = (1 - t)* x0 + t * x1 + sigma * eps.
        Shape: x0,x1 are (B, P, F). We broadcast t => shape (B,1,1).
        """
        eps = torch.randn_like(x0)
        t_ = pad_t_like_x(t, x0)
        return (1.0 - t_) * x0 + t_ * x1 + self.sigma * eps


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
        # e.g. batch["process"] if you store that

        # (a) build transformer context
        transformer_out = self.conditioning(hard_data, hard_mask, reco_data, reco_mask)
        B = transformer_out.shape[0]
        # skip the first null token => real tokens => shape (B, sum_reco, embed_dim)
        context = transformer_out[:, 1:, :]

        # (b) build a [B, sum_reco, len(flow_input_features)] array
        #     also build a mask for each feature => 1=exists, 0=missing
        x_real, feat_mask, sum_reco = self.pack_reco_features(reco_data, reco_mask)
        # x_real => shape (B, sum_reco, len_flow_feats)
        # feat_mask => shape (B, sum_reco, len_flow_feats), 1=exists

        # (c) bridging from x0 ~ Normal(0,1) => x_real
        x0 = torch.randn_like(x_real)
        x1 = x_real
        t = torch.rand(B, device=x_real.device)
        x_t = self.bridging_distribution(x0, x1, t)

        # (d) velocity => flatten across tokens
        sum_reco_tokens = context.shape[1]
        len_flow_feats = x_real.shape[2]
        net_in = torch.cat([
            context.reshape(B * sum_reco_tokens, -1),           # embed_dim
            x_t.reshape(B * sum_reco_tokens, len_flow_feats),   # bridging dims
            t.repeat_interleave(sum_reco_tokens).unsqueeze(-1), # shape (B*sum_reco_tokens,1)
        ], dim=1)
        v_pred = self.velocity_net(net_in).reshape(B, sum_reco_tokens, len_flow_feats)

        # (e) target velocity => (x1 - x0), apply feature mask => MSE
        v_true = x1 - x0
        diff = (v_pred - v_true)**2 * feat_mask
        loss_per_event = diff.mean(dim=(1,2))  # average over tokens+features
        loss = loss_per_event.mean(dim=0)      # average over batch

        # (f) Optionally track per-process
        # if "process" in batch:
        #     for pid in torch.unique(batch["process"]):
        #         pid_idx = (batch["process"]==pid).nonzero(as_tuple=True)[0]
        #         if pid_idx.numel()>0:
        #             sub_loss = loss_per_event[pid_idx].mean()
        #             proc_name = (self.process_names[pid.item()] 
        #                          if self.process_names is not None else f"proc_{pid}")
        #             self.log(f"loss_{proc_name}", sub_loss, prog_bar=False)

        return loss


    def pack_reco_features(self, reco_data, reco_mask):
        """
        Build a single (B, sum_reco, len(self.flow_input_features)) array
        plus a mask for each feature => shape (B, sum_reco, len_flow_feats).

        If a token type lacks a certain feature, fill that feature with 0
        and mark mask=0. Otherwise, fill real value & mask=1.

        This code assumes each 'reco_data[i]' has shape (B, n_i, #features_for_type_i).
        We'll flatten them into sum_reco along the particle axis.
        """
        B = reco_data[0].shape[0]
        # total tokens
        n_tokens_each = [rd.shape[1] for rd in reco_data]
        sum_reco = sum(n_tokens_each)

        len_flow_feats = len(self.flow_input_features)
        x_real = torch.zeros((B, sum_reco, len_flow_feats), device=reco_data[0].device)
        feat_mask = torch.zeros((B, sum_reco, len_flow_feats), device=reco_data[0].device)

        offset = 0
        for type_i in range(len(self.n_reco_particles_per_type)):
            # retrieve e.g. shape => (B, n_i, F_i)
            rd_i = reco_data[type_i]
            mask_i = reco_mask[type_i]  # shape (B, n_i)
            F_i = rd_i.shape[2]
            # for each requested feature f_k, check if it is in the type_i's feature set
            feat_list_i = self.reco_input_features_per_type[type_i]
            for feat_k, feat_name in enumerate(self.flow_input_features):
                if feat_name in feat_list_i:
                    col_idx = feat_list_i.index(feat_name)
                    # fill x_real for tokens in [offset : offset + n_i]
                    x_real[:, offset:offset + rd_i.shape[1], feat_k] = rd_i[:, :, col_idx]
                    # set feat_mask=1 where mask_i=1
                    feat_mask[:, offset:offset + rd_i.shape[1], feat_k] = mask_i
                else:
                    # not present => remain 0 => mask=0
                    pass
            offset += rd_i.shape[1]

        return x_real, feat_mask, sum_reco


    def forward(self, batch):
        """
        For training, we define forward(batch) => returns the CFM loss.
        """
        return self.cfm_loss(batch)

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def sample(self, batch, steps=10):
        """
        Euler stepping from x(0)=N(0,1) => x(1)=reco features, but we only produce
        the bridging features. If you want the full "unpacked" features, you would
        do an 'inverse' or something else. This is just a simple demonstration.
        """
        hard_data = batch["hard"]["data"]
        hard_mask = batch["hard"]["mask"]
        reco_data = batch["reco"]["data"]
        reco_mask = batch["reco"]["mask"]

        # 1) Transformer context => shape (B, sum_reco+1, embed_dim)
        transformer_out = self.conditioning(hard_data, hard_mask, reco_data, reco_mask)
        context = transformer_out[:, 1:, :]  # skip null => (B, sum_reco, embed_dim)

        # 2) x_t from prior => Normal(0,1)
        x_real, feat_mask, sum_reco = self.pack_reco_features(reco_data, reco_mask)
        B, sum_reco_tokens, len_flow_feats = x_real.shape
        x_t = torch.randn_like(x_real)

        dt = 1.0 / steps
        for step_i in range(steps):
            t_ = step_i * dt
            v_t = self.compute_velocity(context, x_t, t_)
            x_t = x_t + dt * v_t

        # We now have a final x_t that is an approximate sample from the bridging distribution.
        # If you want to "unpack" them into separate jets/met arrays, you'd do the inverse
        # of pack_reco_features.
        return x_t

    def compute_velocity(self, context, x_t, t_):
        """
        Evaluate velocity_net.  context: (B, sum_reco, embed_dim)
        x_t: (B, sum_reco, len_flow_feats)
        t_: float in [0,1].
        """
        B, sum_reco, F = x_t.shape
        net_in = torch.cat([
            context.reshape(B * sum_reco, -1),
            x_t.reshape(B * sum_reco, F),
            torch.full((B * sum_reco, 1), t_, device=x_t.device),
        ], dim=1)
        v_flat = self.velocity_net(net_in)
        return v_flat.view(B, sum_reco, F)