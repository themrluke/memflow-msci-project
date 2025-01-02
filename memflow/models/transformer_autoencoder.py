import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import lightning as L

from memflow.models.basics import MLP

class TAE(L.LightningModule):
    def __init__(
        self,
        dim_input,
        dim_embeds,
        dim_latents,
        nhead,
        expansion_factor,
        activation,
        num_encoding_layers,
        num_decoding_layers,
        max_seq_len,
        decoder_window,
        reco_mask_attn=None,
        dropout = 0.,
        process_names = None,
        optimizer = None,
        scheduler_config = None,
    ):
        super().__init__()

        # Public attribute #
        self.process_names = process_names

        # Private attributes #
        self._optimizer = optimizer
        self._scheduler_config = scheduler_config
        self._loss_function = nn.MSELoss(reduction='none')

        # Dimensions #
        if not isinstance(dim_embeds,(list,tuple)):
            dim_embeds = [dim_embeds]
        dim_embed = dim_embeds[-1]
        dim_embeds = dim_embeds[:-1]
        if not isinstance(dim_latents,(list,tuple)):
            dim_latents = [dim_latents]
        dim_latent = dim_latents[-1]
        dim_latents = dim_latents[:-1]

        ##### ENCODER #####
        self.embedding_encoder = MLP(
            dim_in = dim_input,
            dim_out = dim_embed,
            neurons = dim_embeds,
            hidden_activation = activation,
            output_activation = None,
            dropout = dropout,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer = nn.TransformerEncoderLayer(
                d_model = dim_embed,
                nhead = nhead,
                dim_feedforward = dim_embed * expansion_factor,
                activation = activation(),
                batch_first = True,
                dropout = dropout,
            ),
            num_layers = num_encoding_layers,
            norm = nn.LayerNorm(dim_embed),
        )
        self.projection_encoder = MLP(
            dim_in = dim_embed,
            dim_out = dim_latent,
            neurons = dim_latents,
            hidden_activation = activation,
            output_activation = None,
            dropout = dropout,
        )

        ##### DECODER #####
        self.projection_decoder = MLP(
            dim_in = dim_latent,
            dim_out = dim_embed,
            neurons = dim_latents[::-1],
            hidden_activation = activation,
            output_activation = None,
            dropout = dropout,
        )
        self.decoder = nn.TransformerEncoder(
            encoder_layer = nn.TransformerEncoderLayer(
                d_model = dim_embed,
                nhead = nhead,
                dim_feedforward = dim_embed * 2,
                activation = activation(),
                batch_first = True,
                dropout = dropout,
            ),
            num_layers = num_decoding_layers,
            norm = nn.LayerNorm(dim_embed),
        )
        self.embedding_decoder = MLP(
            dim_in = dim_embed,
            dim_out = dim_input,
            neurons = dim_embeds[::-1],
            hidden_activation = activation,
            output_activation = None,
            dropout = dropout,
        )

        ##### Masks #####
        self.reco_mask_attn = reco_mask_attn

        self.decoder_mask = torch.full((max_seq_len+1,max_seq_len+1),fill_value=True)
        self.decoder_mask[:,0] = False # all attend to CLS token
        x = torch.arange(max_seq_len+1)
        for k in range(decoder_window):
            y = x-k
            self.decoder_mask[x[y>=0],y[y>=0]] = False
        print ('Decoder mask')
        print (self.decoder_mask)

        ##### Position embedding #####
        self.position_embedding = nn.Embedding(
            num_embeddings = max_seq_len+1,
            embedding_dim = dim_embed,
        )

    @staticmethod
    def make_CLS_token(x):
        return torch.ones((x.shape[0],1,x.shape[2])).to(x.device)

    @staticmethod
    def process_padding_mask(padding_mask,attn_mask=None,add_cls=True):
        # Handle mask type #
        if not padding_mask.dtype == torch.bool:
            # Adapt mask type
            padding_mask = padding_mask > 0
        if attn_mask is not None:
            # Use the reco_attn_mask to unmask particles we want the transformer to know are absent
            padding_mask = torch.logical_or(
                attn_mask.to(padding_mask.device),
                padding_mask,
            )

        # Add True for CLS token
        if add_cls:
            padding_mask = torch.cat(
                (
                    (torch.ones((padding_mask.shape[0],1)) > 0).to(padding_mask.device),
                    padding_mask,
                ),
                dim = 1,
            )
        # True -> not attended
        padding_mask = ~padding_mask

        return padding_mask


    def embedding(self,x,padding_mask):
        # Zero out the absent particles #
        x = x * padding_mask.unsqueeze(-1)
            # at this stage in padding_mask : present = True, absent = False

        # Add the CLS token #
        x = torch.cat(
            (
                self.make_CLS_token(x),
                x,
            ),
            dim = 1,
        )
        # Embed #
        x = self.embedding_encoder(x)

        return x

    def encode(self,x,padding_mask):
        # Embed #
        x = self.embedding(x,padding_mask)
        # Process mask #
        padding_mask = self.process_padding_mask(padding_mask,self.reco_mask_attn)

        # Encode #
        x_latent = self.encoder(
            src = x,
            src_key_padding_mask = padding_mask,
        )[:,0,:]
        x_latent = self.projection_encoder(x_latent)
        return x_latent

    def decode(self,latent,padding_mask):
        # Padding mask #
        seq_len = padding_mask.shape[1]
        padding_mask_present = ~self.process_padding_mask(padding_mask,add_cls=False)
            # we want to keep padding_mask to replace produced tokens for absent particles
        padding_mask_attn = self.process_padding_mask(padding_mask,self.reco_mask_attn).to(self.device)
            # contains both padding mask and attn_mask to consider absent particles in self-attention

        # Generate null token #
        null_token = self.embedding_encoder(
            torch.zeros(1,self.embedding_encoder.dim_in).to(self.device)
        )

        # expand dimension of latent
        latent = self.projection_decoder(latent).unsqueeze(1)

        # make reco with zeros but latent as first token
        decoder_input = torch.cat(
            (
                latent,
                torch.zeros((latent.shape[0],seq_len,latent.shape[2])).to(self.device),
            ),
            dim = 1,
        ).to(self.device)

        # Position embedding
        pos = torch.arange(padding_mask.shape[1]+1).unsqueeze(0).repeat(latent.shape[0],1).to(self.device)
        embed_pos = self.position_embedding(pos)

        # generation loop
        decoder_embedding_output = torch.zeros(
            (
                latent.shape[0],                # N batch
                padding_mask.shape[1],          # S seq len
                self.embedding_encoder.dim_in,  # F input features
            )
        ).to(self.device)
        for i in range(seq_len):
            src_key_padding_mask = torch.logical_not(
                torch.logical_and(
                    torch.logical_not(padding_mask_attn),
                    torch.logical_not(self.decoder_mask[i].to(self.device)),
                )
            )
            # Pass through decoder
            decoder_output = self.decoder(
                src = decoder_input + embed_pos,
                src_key_padding_mask = src_key_padding_mask,
            )[:,i,:] # Only need the generated i element

            # Make the initial reconstructed input #
            emb_out = self.embedding_decoder(decoder_output)
            # Zero-out in case particle is absent
            emb_out[~padding_mask_present[:,i]] = 0.
            # Record #
            decoder_embedding_output[:,i,:] = emb_out

            # Obtain the corresponding token #
            decoder_input[:,i+1,:] = self.embedding_encoder(emb_out)

        return decoder_embedding_output


    def forward(self,x,padding_mask):
        # Embed #
        x = self.embedding(x,padding_mask)
        # Process mask #
        padding_mask_present = ~self.process_padding_mask(padding_mask,add_cls=False)
            # keep track of what particles are present to zero out the others later
        padding_mask_src = self.process_padding_mask(padding_mask,self.reco_mask_attn)
            # inverted mask to match attention
            # + include CLS token
            # + include absent particles we want the transformer to know about

        # Encode #
        x_latent = self.encoder(
            src = x,
            src_key_padding_mask = padding_mask_src,
        )[:,0,:]
        x_latent = self.projection_encoder(x_latent)

        # Decode #
        x_latent = self.projection_decoder(x_latent).unsqueeze(1)
        x_reco = torch.cat(
            (
                x_latent,  # latent space = encoded+projected CLS token
                x[:,1:,:], # include the sequence (without CLS token)
            ),
            dim = 1,
        )
        x_pos = torch.arange(x_reco.shape[1]).unsqueeze(0).repeat(x.shape[0],1).to(self.device)
        x_reco = self.decoder(
            src = x_reco + self.position_embedding(x_pos),
            src_key_padding_mask = padding_mask_src,
            mask = self.decoder_mask.to(self.device),
        )
        x_reco = self.embedding_decoder(x_reco[:,:-1,:]) # unshift the CLS token
        # zero out absent particles
        x_reco = x_reco * padding_mask_present.unsqueeze(-1)
            # Note : they are ignored in the loss anyway

        return x_reco

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

    def shared_eval(self, batch, batch_idx, prefix, generative=False):
        x_init = torch.cat(batch['data'],dim=1)
        mask = torch.cat(batch['mask'],dim=1)

        # Full parallel reco #
        x_reco = self(x_init,mask)

        loss_per_particle = self._loss_function(x_init,x_reco).sum(dim=-1)
        loss_values = (loss_per_particle * mask).sum(dim=1) / mask.sum(dim=1)

        self.log(f"{prefix}/loss_tot", loss_values.mean(), prog_bar=True)

        if generative:
            x_latent = self.encode(x_init,mask)
            x_gen = self.decode(x_latent,mask)
            loss_gens = ((self._loss_function(x_init,x_gen).sum(dim=-1) * mask).sum(dim=1) / mask.sum(dim=1))
            self.log(f"{prefix}/loss_gen", loss_gens.mean(), prog_bar=True)

        # Log per process #
        if 'process' in batch.keys():
            for idx in torch.unique(batch['process']).sort()[0]:
                process_idx = torch.where(batch['process'] == idx)[0]
                process_name = self.process_names[idx] if self.process_names is not None else str(idx)
                self.log(
                    f"{prefix}/loss_{process_name}",
                    loss_values[process_idx].mean(),
                    prog_bar = False,
                )

                if generative:
                    self.log(
                        f"{prefix}/loss_gen_{process_name}",
                        loss_gens[process_idx].mean(),
                        prog_bar = False,
                    )


        return loss_values.mean()

    def training_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'val', generative=True)


