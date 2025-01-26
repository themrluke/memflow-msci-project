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
    A Conditional Flow Matching model that parallels the 'TransferFlow' structure:
      - Takes lists of (hard_data, reco_data) per particle type,
      - Embeds them via small MLP layers,
      - Runs a Transformer for context conditioning,
      - Then, instead of normalizing flow heads, uses bridging distribution x(t)
        and velocity nets to compute a CFM loss.

    Data structure assumptions:
      - hard_data: list of length len(n_hard_particles_per_type)
        each element has shape (batch, nHard_i, #features_for_that_type).
      - reco_data: same structure for reco particles.
      - hard_mask_exist[i], reco_mask_exist[i]: bool or float masks
        indicating which particles exist in each event.
      - We also have "flow_input_features[i]" to pick which features
        (pt, eta, phi, mass, etc.) are included in the bridging distribution
        for type i.

    The main difference from TransferFlow is that we do not loop feature-by-feature
    with normalizing flows, but rather a single velocity net per type
    (or per particle, if you like).
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
        transformer_args=None,
        onehot_encoding=False,
        process_names=None,
        # CFM-specific arguments:
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
        self.hard_mask_attn = hard_mask_attn
        self.reco_mask_attn = reco_mask_attn
        self.onehot_encoding = onehot_encoding
        self.process_names = process_names

        self.sigma = sigma  # bridging-dist std
        self._optimizer = optimizer
        self._scheduler_config = scheduler_config

        # Basic checks:
        assert len(n_reco_particles_per_type) == len(reco_input_features_per_type), (
            f"Got {len(n_reco_particles_per_type)} sets of reco particles vs. "
            f"{len(reco_input_features_per_type)} sets of reco features."
        )
        assert len(n_hard_particles_per_type) == len(hard_input_features_per_type), (
            f"Got {len(n_hard_particles_per_type)} sets of hard particles vs. "
            f"{len(hard_input_features_per_type)} sets of hard features."
        )
        assert len(flow_input_features) == len(reco_input_features_per_type), (
            f"flow_input_features has length {len(flow_input_features)} but "
            f"we have {len(reco_input_features_per_type)} reco feature sets."
        )

        # Prepend True for null token in reco_mask_attn if present
        if self.reco_mask_attn is not None:
            self.reco_mask_attn = torch.cat(
                [torch.tensor([True]), self.reco_mask_attn],
                dim=0
            )

        # Build embeddings for Hard and Reco
        self.hard_embeddings = self.make_embeddings(self.hard_input_features_per_type)
        self.reco_embeddings = self.make_embeddings(self.reco_input_features_per_type)

        # Build the Transformer
        transformer_args = dict(transformer_args)  # shallow copy
        transformer_args["d_model"] = self.embed_dim
        if "dropout" not in transformer_args:
            transformer_args["dropout"] = self.dropout

        self.transformer = nn.Transformer(batch_first=True, **transformer_args)
        max_reco_len = sum(self.n_reco_particles_per_type) + 1  # +1 for null
        self.tgt_mask = self.transformer.generate_square_subsequent_mask(max_reco_len)

        # Build velocity nets (one per type):
        self.vel_nets = nn.ModuleList()
        self.flow_indices = []
        for i in range(len(self.n_reco_particles_per_type)):
            # find which features we use for type i
            feats = self.flow_input_features[i]
            hi_feats = self.hard_input_features_per_type[i]  # e.g. ["pt","eta","phi","mass"]
            indices_hard = []
            for f_ in feats:
                idx_ = hi_feats.index(f_)
                indices_hard.append(idx_)
            self.flow_indices.append(indices_hard)

            # small MLP: input = (transformer_context + x(t) + t), output = velocity for #feats
            d_in = self.embed_dim + len(feats) + 1
            d_hid = 128
            net = nn.Sequential(
                nn.Linear(d_in, d_hid),
                nn.SiLU(),
                nn.Linear(d_hid, d_hid),
                nn.SiLU(),
                nn.Linear(d_hid, len(feats)),
            )
            self.vel_nets.append(net)

    def make_embeddings(self, input_features_per_type):
        """
        Build an MLP embedding for each type's raw features,
        just like in TransferFlow. The final dimension = self.embed_dim.
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
        """
        Lightning method that returns optimizer (and optional scheduler).
        You can also just define this inline or pass in self._optimizer externally.
        """
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

    def conditioning(self, hard_data, hard_mask_exist, reco_data, reco_mask_exist):
        """
        Runs the Transformer “context” step:
          1) Embed all hard_data → cat across types → pass as Transformer encoder input
          2) Embed all reco_data, but prepend a null token for the first type
          3) Output the final decoder: (B, sum(n_reco) + 1, embed_dim)

        The “mask_attn” logic is inherited from the original TransferFlow approach.
        """
        # 1) Hard side
        hard_embed = []
        hard_mask_all = []
        for i in range(len(hard_data)):
            he = self.hard_embeddings[i](hard_data[i])  # shape (B, nHard_i, embed_dim)
            # multiply by exist mask
            he = he * hard_mask_exist[i].unsqueeze(-1)
            hard_embed.append(he)
            hard_mask_all.append(hard_mask_exist[i])

        hard_embed_cat = torch.cat(hard_embed, dim=1)   # shape (B, sum(nHard_i), embed_dim)
        hard_mask_cat  = torch.cat(hard_mask_all, dim=1)  # shape (B, sum(nHard_i))

        # 2) Reco side (null token in the first type):
        emb_list = []
        mask_list = []
        if len(reco_data) > 0:
            # first type
            bsz, n_reco_0, feat_dim = reco_data[0].shape
            null_token = torch.full(
                (bsz, 1, feat_dim), -1.0,
                device=reco_data[0].device,
                dtype=reco_data[0].dtype,
            )
            # cat
            x0 = torch.cat([null_token, reco_data[0]], dim=1)
            # embed
            e0 = self.reco_embeddings[0](x0)
            # mask: True for null token
            m0 = torch.cat([
                torch.ones_like(reco_mask_exist[0][:, :1]),  # shape (B,1)
                reco_mask_exist[0]
            ], dim=1)
            e0 = e0 * m0.unsqueeze(-1)
            emb_list.append(e0)
            mask_list.append(m0)

            # subsequent reco types
            for i in range(1, len(reco_data)):
                e_ = self.reco_embeddings[i](reco_data[i]) \
                     * reco_mask_exist[i].unsqueeze(-1)
                emb_list.append(e_)
                mask_list.append(reco_mask_exist[i])
            reco_embed_cat = torch.cat(emb_list, dim=1)  # shape (B, sum(nReco_i)+1, embed_dim)
            reco_mask_cat  = torch.cat(mask_list, dim=1) # shape (B, sum(nReco_i)+1)
        else:
            # corner case if no reco data
            reco_embed_cat = torch.zeros_like(hard_embed_cat)
            reco_mask_cat  = torch.zeros_like(hard_mask_cat)

        # 3) Convert mask to “key_padding_mask=True => ignore”
        if self.hard_mask_attn is not None:
            keep_hard = torch.logical_or(
                self.hard_mask_attn.to(hard_mask_cat.device), hard_mask_cat.bool()
            )
        else:
            keep_hard = hard_mask_cat.bool()
        if self.reco_mask_attn is not None:
            keep_reco = torch.logical_or(
                self.reco_mask_attn.to(reco_mask_cat.device), reco_mask_cat.bool()
            )
        else:
            keep_reco = reco_mask_cat.bool()

        hard_key_pad = ~keep_hard  # True=ignore in PyTorch
        reco_key_pad = ~keep_reco

        # 4) Pass through Transformer
        out = self.transformer(
            src=hard_embed_cat,
            tgt=reco_embed_cat,
            src_key_padding_mask=hard_key_pad,
            tgt_key_padding_mask=reco_key_pad,
            memory_key_padding_mask=hard_key_pad,
            tgt_mask=self.tgt_mask.to(reco_embed_cat.device),
        )
        # out => shape (B, sum(n_reco)+1, embed_dim)
        return out

    def bridging_distribution(self, x0, x1, t):
        """
        x(t) = (1 - t)* x0 + t * x1 + sigma * eps.
        Shape: x0,x1 are (B, P, F). We broadcast t => shape (B,1,1).
        """
        eps = torch.randn_like(x0)
        t_ = pad_t_like_x(t, x0)
        x_t = (1.0 - t_) * x0 + t_ * x1 + self.sigma * eps
        return x_t

    def cfm_loss(self, batch):
        """
        Compute the MSE loss for conditional flow matching:
          1) Get Transformer context
          2) For each type i, slice out the portion of context
             (account for the single null token in type0).
          3) bridging distribution x(t)
          4) velocity net => v_pred
          5) MSE vs. (x1 - x0), masked
        """
        hard_data = batch["hard"]["data"]
        hard_mask = batch["hard"]["mask"]
        reco_data = batch["reco"]["data"]
        reco_mask = batch["reco"]["mask"]

        # 1) condition
        condition = self.conditioning(hard_data, hard_mask, reco_data, reco_mask)
        # shape => (B, sum(n_reco_particles)+1, embed_dim)

        # 2) We'll slice condition into type i chunks
        # The first chunk has length (n_reco_particles_for_type0+1), subsequent ones have n_i
        slices = np.cumsum([1] + self.n_reco_particles_per_type)
        B = condition.size(0)
        # print("condition.shape =", condition.shape)
        # print("slices =", slices)

        # random t in [0,1] for bridging
        t = torch.rand(B, device=condition.device)

        total_loss = 0.0
        total_count = 0

        for i in range(len(self.n_reco_particles_per_type)):
            # slice out the context
            c_i = condition[:, slices[i] : slices[i+1], :]
            # print(f"Type {i}, after slice =>", c_i.shape)
            if i == 0:
                c_i = c_i
                # print(f"Type {i}, after removing null =>", c_i.shape)

            # get x0_i, x1_i from the chosen flow_input_features
            idxs = self.flow_indices[i]  # list of indices
            x0_i = hard_data[i][..., idxs]  # shape (B, nHard_i, fSel)
            x1_i = reco_data[i][..., idxs]  # shape (B, nReco_i, fSel)
            # Typically we assume nHard_i == nReco_i if you're matching 1:1 quarks→jets, etc.
            # If they differ, you'll need additional logic.

            # bridging distribution
            x_t = self.bridging_distribution(x0_i, x1_i, t)  # shape (B, n_i, fSel)

            # flatten for velocity net
            B_i, n_i, fSel = x_t.shape
            net_input = torch.cat([
                c_i.reshape(B_i*n_i, -1),        # (B*n_i, embed_dim)
                x_t.reshape(B_i*n_i, fSel),      # (B*n_i, fSel)
                t.unsqueeze(-1).expand(B_i, n_i).reshape(B_i*n_i, 1),
            ], dim=1)

            v_pred = self.vel_nets[i](net_input)  # shape (B*n_i, fSel)
            v_pred = v_pred.reshape(B_i, n_i, fSel)

            # target velocity
            v_true = x1_i - x0_i  # shape (B, n_i, fSel)

            # MSE with mask
            mask_2d = reco_mask[i].float()  # shape (B, n_i)
            mask_3d = mask_2d.unsqueeze(-1) # shape (B, n_i, 1)
            diff = (v_pred - v_true)**2 * mask_3d
            loss_i_per_event = diff.mean(dim=(1, 2))  # average over particles/features
            total_loss += loss_i_per_event.sum()
            total_count += B_i

        cfm_loss = total_loss / float(total_count)
        return cfm_loss

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

    def gen_only_context(self, hard_data_list, hard_mask_list):
        """
        Build a Transformer ENCODER context for each hard-data type
        without depending on the real reco data.
        - We embed each hard type: shape => (B, nHard_i, feats) -> (B, nHard_i, embed_dim)
        - Concatenate along the particle axis => (B, sum(nHard_i), embed_dim)
        - Build a key_padding_mask => shape (B, sum(nHard_i)), True=ignore
        - Pass through self.transformer.encoder(...) to get the ENCODER output
        - Slice back into each type chunk => (B, nHard_i, embed_dim)

        Returns:
        context_list: a list of length == len(hard_data_list),
                        each => (B, nHard_i, embed_dim)
        """
        device = hard_data_list[0].device
        # 1) embed each type -> shape (B, nHard_i, embed_dim), zero out missing with mask
        embedded_types = []
        mask_types     = []
        for i in range(len(hard_data_list)):
            x_i = hard_data_list[i]             # shape (B, nHard_i, feats)
            m_i = hard_mask_list[i]            # shape (B, nHard_i)  => 1=exists,0=missing
            e_i = self.hard_embeddings[i](x_i)  # => (B, nHard_i, embed_dim)
            e_i = e_i * m_i.unsqueeze(-1)       # zero out where mask=0
            embedded_types.append(e_i)
            mask_types.append(m_i)

        # 2) concatenate along the particle axis
        embed_cat = torch.cat(embedded_types, dim=1)  # (B, sum_nHard, embed_dim)
        mask_cat  = torch.cat(mask_types, dim=1)      # (B, sum_nHard)

        # 3) build key_padding_mask => True=ignore in PyTorch Transformers
        # If mask_cat=1 => means "exists," so we do "False=keep" => ~mask_cat.bool()
        key_pad = ~mask_cat.bool()  # shape (B, sum_nHard)

        # 4) run the ENCODER
        # We do NOT call the full self.transformer(...) since that also does a decoder pass.
        # Instead, we call self.transformer.encoder(...) directly:
        encoder_out = self.transformer.encoder(
            src=embed_cat,                  # shape (B, sum_nHard, embed_dim)
            src_key_padding_mask=key_pad    # shape (B, sum_nHard)
        )
        # encoder_out => shape (B, sum_nHard, embed_dim)

        # 5) slice them back into each type
        slices = np.cumsum([0] + self.n_hard_particles_per_type)  # e.g. [0,6,7] if [6,1]
        context_list = []
        for i in range(len(hard_data_list)):
            start = slices[i]
            end   = slices[i+1]
            c_i   = encoder_out[:, start:end, :]  # shape => (B, nHard_i, embed_dim)
            context_list.append(c_i)
        return context_list

    def sample(self, batch, steps=10):
        """
        Euler bridging from x(0)=hard data => x(1). 
        We'll produce final "reco-like" data for each type. 
        """
        hard_data_list = batch["hard"]["data"]  # list of length=2 if you have [partons, neutrinos]
        hard_mask_list = batch["hard"]["mask"]

        # 1) get the encoder context for each type i => shape (B, nHard_i, embed_dim)
        context_list = self.gen_only_context(hard_data_list, hard_mask_list)

        x_t_list = []
        for i, x0 in enumerate(hard_data_list):
            x_t = x0.clone()       # shape (B, n_i, feats)
            c_i = context_list[i]  # shape (B, n_i, embed_dim)
            dt = 1.0 / steps

            for step_i in range(steps):
                t_ = step_i * dt
                v_t = self.compute_velocity_for_type(i, x_t, t_, c_i)
                idxs = self.flow_indices[i]  # select only flow features
                x_t[..., idxs] = x_t[..., idxs] + dt * v_t  # update only the relevant features

            x_t_list.append(x_t)
        return x_t_list

    def compute_velocity_for_type(self, i, x_t, t_, c_i):
        """
        i: which type index
        x_t: (B, n_i, feats)
        t_: scalar float
        c_i: (B, n_i, embed_dim)
        => returns velocity => shape (B, n_i, feats)
        """
        # 1) gather only the flow feats
        idxs = self.flow_indices[i]  # e.g. [0,1,2] for (pt,eta,phi)
        x_t_select = x_t[..., idxs]   # shape => (B, n_i, 3)
        B, n_i, feats = x_t_select.shape

        # 2) flatten
        c_flat  = c_i.reshape(B*n_i, self.embed_dim) # (B*n_i, 64)
        x_flat  = x_t_select.reshape(B*n_i, feats) # (B*n_i, 3)
        t_tensor= torch.full((B*n_i, 1), t_, device=x_t.device)

        # cat => shape => (B*n_i, embed_dim + feats + 1)
        inp = torch.cat([c_flat, x_flat, t_tensor], dim=1)  
        # e.g. if embed_dim=64, feats=3 => 64+3+1=68

        v_flat = self.vel_nets[i](inp)  # => (B*n_i, feats)
        v_t = v_flat.view(B, n_i, feats)
        return v_t