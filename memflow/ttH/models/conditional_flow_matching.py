import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import math
import numpy as np

from typing import Optional, Dict, Any, List
from zuko.distributions import DiagNormal

def pad_t_like_x(t, x):
    """
    Utility that reshapes (batch,) → (batch, 1, ..., 1) so that
    t can broadcast with x (B, P, F) as needed.
    """
    while t.dim() < x.dim():
        t = t.unsqueeze(-1)
    return t

class TransferCFM(L.LightningModule):
    """
    A CFM approach that uses the same style of data + embeddings + Transformer
    as the TransferFlow, but replaces the normalizing-flow heads with
    a time-dependent velocity field for bridging distribution x(t).

    x0 = 'hard' data
    x1 = 'reco' data
    x(t) = (1 - t)*x0 + t*x1 + sigma * eps
    The velocity is predicted via a neural net that sees the Transformer context,
    x(t), and t. Then we minimize MSE[ v(t,x(t)) - (x1 - x0) ] (CFM loss).
    """

    def __init__(
        self,
        # --- same as in TransferFlow: ---
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
        transformer_args={},
        onehot_encoding=False,
        process_names=None,
        # ---- new / replaced items: ----
        sigma=0.1,
        optimizer=None,
        scheduler_config=None,
    ):
        super().__init__()

        # Basic stored config
        self.embed_dims = embed_dims if isinstance(embed_dims, list) else [embed_dims]
        self.embed_dim = self.embed_dims[-1]
        self.embed_act = embed_act
        self.dropout   = dropout

        self.n_hard_particles_per_type = n_hard_particles_per_type
        self.hard_particle_type_names  = hard_particle_type_names
        self.hard_input_features_per_type = hard_input_features_per_type

        self.n_reco_particles_per_type = n_reco_particles_per_type
        self.reco_particle_type_names  = reco_particle_type_names
        self.reco_input_features_per_type = reco_input_features_per_type

        self.flow_input_features = flow_input_features  # which features we eventually match (like "pt","eta","phi", etc.)

        self.hard_mask_attn = hard_mask_attn
        self.reco_mask_attn = reco_mask_attn
        self.onehot_encoding = onehot_encoding
        self.process_names   = process_names

        self.sigma = sigma  # bridging-dist std
        self._optimizer = optimizer
        self._scheduler_config = scheduler_config

        # FIX: prepend a True entry to reco_mask_attn for the null token
        if reco_mask_attn is not None:
            # Suppose your original reco_mask_attn was shape (X,).
            # By prepending [True], it becomes shape (X+1,).
            # That ensures it matches the dimension for your “null token”.
            reco_mask_attn = torch.cat(
                [torch.ones(1, dtype=torch.bool), reco_mask_attn],
                dim=0
            )
        self.reco_mask_attn = reco_mask_attn

        # If you also have a null token for hard data, do something similar:
        if hard_mask_attn is not None:
            # Possibly do the same for the hard side if you also prepend a null
            # token there. If not, you can leave it alone.
            pass
        self.hard_mask_attn = hard_mask_attn
        # ------------------------------------------------------

        # --- 1) Build the embeddings for Hard and Reco (same idea as in TransferFlow) ---
        self.hard_embeddings = self.make_embeddings(self.hard_input_features_per_type)
        self.reco_embeddings = self.make_embeddings(self.reco_input_features_per_type)

        # Optionally add “onehot” position encoding (like in TransferFlow)
        self.onehot_tensors = None
        if self.onehot_encoding:
            # (You can adapt from your TransferFlow code that builds onehot_tensors)
            raise NotImplementedError("Implement your onehot logic if needed")

        # --- 2) Build the Transformer ---
        if "d_model" in transformer_args:
            # override
            transformer_args["d_model"] = self.embed_dim
        else:
            transformer_args["d_model"] = self.embed_dim
        if "dropout" not in transformer_args:
            transformer_args["dropout"] = self.dropout

        self.transformer = nn.Transformer(
            batch_first=True,
            **transformer_args
        )

        # We add a standard subsequent mask for the “decoder”
        #   e.g. sum(n_reco_particles) + 1 for the null token
        max_reco_particles = sum(self.n_reco_particles_per_type) + 1
        self.tgt_mask = self.transformer.generate_square_subsequent_mask(
            max_reco_particles
        )

        # --- 3) Build a “velocity head” that predicts v(t,x(t)) from the context + x(t) + t. ---
        # The dimension of x(t) is the total number of features across the chosen “flow_input_features”.
        # Typically, that is 3 for jets (“pt, eta, phi”) or 2 for MET, etc.
        # We can get the total dimension from sum of all features, or do it per type.
        # For simplicity, let's do a single large velocity network that acts on each “particle” row.
        # We'll embed x(t) + the transformer context, then produce velocity for each feature.
        self.vel_head = nn.Sequential(
            nn.Linear(self.embed_dim + 1 + 8, 128),  # (embedding + time + some guess at #features)
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 3),  # e.g. predict velocity for [pt, eta, phi] (adjust as needed)
        )
        # The above is just an example.  In practice, you might want to do something more elaborate
        # or do it per-type, etc.

    def make_embeddings(self, input_features_per_type):
        """
        Build a set of MLP embeddings, one for each “type” in the dataset
        (just like in TransferFlow).
        """
        embeddings = nn.ModuleList()
        for features in input_features_per_type:
            in_dim = len(features)
            layers = []
            for i in range(len(self.embed_dims)):
                d_in = in_dim if i == 0 else self.embed_dims[i - 1]
                d_out = self.embed_dims[i]
                layers.append(nn.Linear(d_in, d_out))
                if i < len(self.embed_dims) - 1 and self.embed_act is not None:
                    layers.append(self.embed_act)
                if self.dropout != 0.0:
                    layers.append(nn.Dropout(self.dropout))
            embeddings.append(nn.Sequential(*layers))
        return embeddings

    def conditioning(self, hard_data, hard_mask_exist, reco_data, reco_mask_exist):
        """
        Same logic as in TransferFlow:
        1) Concatenate hard objects (B, #hard, #feat) → embeddings
        2) Insert a null token at the front of reco
        3) Concatenate reco objects (B, #reco, #feat) → embeddings
        4) Pass them to Transformer (src=hard, tgt=reco)
        5) Return “memory” or “decoder” (the condition).
        """
        # 1) Hard
        hard_mask = torch.cat(hard_mask_exist, dim=1)  # (B, total_hard_particles)
        hard_embed = torch.cat(
            [
                emb_module(hard_data[i])
                for i, emb_module in enumerate(self.hard_embeddings)
            ],
            dim=1
        ) * hard_mask.unsqueeze(-1)  # zero out missing

        # 2) Reco (add null token)
        null_token = torch.full(
            (reco_data[0].shape[0], 1, reco_data[0].shape[2]), -1.0,
            device=reco_data[0].device
        )
        reco_mask_null = [
            torch.cat(
                [
                    torch.ones_like(reco_mask_exist[0][:, :1]),  # True for null
                    reco_mask_exist[0]
                ],
                dim=1
            )
        ] + reco_mask_exist[1:]

        reco_data_null = [
            torch.cat([null_token, reco_data[0]], dim=1)
        ] + reco_data[1:]

        # 3) Reco embedding
        reco_mask = torch.cat(reco_mask_null, dim=1)  # (B, total_reco_particles+1)
        reco_embed = torch.cat(
            [
                emb_module(reco_data_null[i])
                for i, emb_module in enumerate(self.reco_embeddings)
            ],
            dim=1
        ) * reco_mask.unsqueeze(-1)

        # 4) Convert bool mask → float mask for transformer
        #    The Transformer expects “padding_mask” with True=pad (−∞).
        #    We invert logic or pass zeros accordingly.
        if self.hard_mask_attn is not None:
            hard_attn_mask = torch.logical_or(
                self.hard_mask_attn.to(hard_mask.device), hard_mask.bool()
            )
        else:
            hard_attn_mask = hard_mask.bool()
        if self.reco_mask_attn is not None:
            reco_attn_mask = torch.logical_or(
                self.reco_mask_attn.to(reco_mask.device), reco_mask.bool()
            )
        else:
            reco_attn_mask = reco_mask.bool()

        # Turn True→0, False→−inf for “key_padding_mask”
        # Actually the standard in PyTorch is that `True` means “mask out”, i.e. “ignore token.”
        # If you want to do it the same as TransferFlow, adjust carefully:
        enc_key_pad  = ~hard_attn_mask  # True where we want to keep
        dec_key_pad  = ~reco_attn_mask
        # 5) Forward pass
        # Tgt mask is a standard subsequent mask
        out = self.transformer(
            src=hard_embed,
            tgt=reco_embed,
            src_key_padding_mask=enc_key_pad,
            tgt_key_padding_mask=dec_key_pad,
            memory_key_padding_mask=enc_key_pad,
            tgt_mask=self.tgt_mask.to(reco_embed.device),
        )
        # out: (B, total_reco_particles+1, d_model)
        # We'll drop the null token from the front, splitting it back into the original reco pieces if we want
        # or we can keep it as a single block for the velocity net.  Let’s keep it as a single block for simplicity:
        return out  # shape (B, sum(n_reco)+1, embed_dim)

    def bridging_distribution(self, x0, x1, t):
        """
        x(t) = (1 - t)*x0 + t*x1 + sigma * eps
        x0, x1, x(t) all have shape (B, P, F).
        t is shape (B,) or (B,1).
        We'll broadcast properly.
        """
        eps = torch.randn_like(x0)
        t_ = pad_t_like_x(t, x0)  # shape (B, 1, 1) or so
        x_t = (1 - t_)*x0 + t_*x1 + self.sigma * eps
        return x_t

    def forward(self, batch):
        """
        The main “forward” pass.  We will compute the MSE for CFM
        and return it as the “loss.”  Then in the training_step/validation_step
        we can log it, etc.
        """
        # 1) unpack data
        hard_data  = batch["hard"]["data"]
        hard_mask  = batch["hard"]["mask"]
        reco_data  = batch["reco"]["data"]
        reco_mask  = batch["reco"]["mask"]

        # 2) get the “condition” from the transformer
        # shape => (B, sum(n_reco_particles)+1, embed_dim)
        condition = self.conditioning(hard_data, hard_mask, reco_data, reco_mask)

        # 3) build x0, x1 Tensors that contain only the “flow_input_features”
        #    We do exactly what TransferFlow does: gather the selected features from `reco_data`.
        #    But for CFM we also need the “hard_data” features, so let's do:
        x0_list, x1_list, exist_mask = [], [], []
        for i, (hard_feats, reco_feats) in enumerate(zip(hard_data, reco_data)):
            # Grab the subset of features from `flow_input_features[i]`.
            fsel = self.flow_input_features[i]  # e.g. ['pt','eta','phi']
            # Indices for the relevant columns in the “hard_feats[i]” etc.
            # Suppose your dataset's order is exactly the same as in “hard_input_features_per_type[i]”.
            # or you can do something more robust.  For brevity:
            chosen_indices = []
            # E.g. find the index in the dataset for each “fsel”
            for f_ in fsel:
                idx_ = self.hard_input_features_per_type[i].index(f_)
                chosen_indices.append(idx_)
            chosen_indices = torch.tensor(chosen_indices, device=hard_feats.device)

            # shape (B, Npart, len(chosen_indices))
            x0_list.append(hard_feats[..., chosen_indices])
            x1_list.append(reco_feats[..., chosen_indices])
            # mask
            exist_mask.append(batch["reco"]["mask"][i])  # shape (B, Npart)

        # Concat them along the “particle” axis
        x0_cat = torch.cat(x0_list, dim=1)  # (B, sum(n_reco_particles), #features)
        x1_cat = torch.cat(x1_list, dim=1)
        mask_cat = torch.cat(exist_mask, dim=1)  # (B, sum(n_reco_particles))

        # 4) sample random t in [0,1]
        B = x0_cat.size(0)
        t = torch.rand(B, device=x0_cat.device)

        # 5) bridging distribution
        x_t = self.bridging_distribution(x0_cat, x1_cat, t)  # shape (B, sum(n_reco), #features)

        # 6) velocity net
        # We want to feed each particle’s condition from the Transformer plus x(t) plus t.
        # But the Transformer “condition” has shape (B, sum(n_reco)+1, embed_dim).
        # Typically, the first token is the “null,” so let’s remove it:
        condition_no_null = condition[:, 1:, :]  # (B, sum(n_reco), embed_dim)
        # Combine them
        # e.g. shape (B, sum(n_reco), embed_dim + #features + 1)
        # Then pass it into self.vel_head *per particle*
        # We'll just flatten out (B*Np, D) for one pass:
        B_np, P, F_ = x_t.shape
        # flatten
        c_flat = condition_no_null.reshape(B_np*P, self.embed_dim)
        x_flat = x_t.reshape(B_np*P, F_)
        t_flat = t.unsqueeze(-1).expand(B_np, P).reshape(-1, 1)

        combined = torch.cat([c_flat, x_flat, t_flat], dim=1)  # shape (B*Np, embed_dim + #features + 1)

        v_pred = self.vel_head(combined)  # (B*Np, #features)
        v_pred = v_pred.reshape(B_np, P, -1)  # (B, Np, #features)

        # 7) target velocity = (x1 - x0)
        v_target = x1_cat - x0_cat  # (B, sum(n_reco), #features)

        # 8) MSE, masked
        # mask_cat is (B, sum(n_reco)) with 1 = existing, 0 = not existing
        # broadcast over features
        mask_3d = mask_cat.unsqueeze(-1).float()  # shape (B, sum(n_reco), 1)
        mse = (v_pred - v_target)**2 * mask_3d
        loss_per_event = mse.mean(dim=(1, 2))  # average over particles+features
        loss = loss_per_event.mean()           # average over batch

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        if self._scheduler_config is None:
            return self._optimizer
        else:
            return {
                "optimizer": self._optimizer,
                "lr_scheduler": self._scheduler_config,
            }