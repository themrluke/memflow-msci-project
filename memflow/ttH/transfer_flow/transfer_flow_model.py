from itertools import chain

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import lightning as L

from memflow.models.utils import lowercase_recursive
from transfer_flow.tools import *


class TransferFlow(AbsModel):
    def __init__(
        self,
        # models #
        encoder_embeddings,
        decoder_embeddings,
        transformer,
        flow,
        # parameters #
        hard_names = None,
        reco_names = None,
        process_names = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.save_hyperparameters(
            ignore=["optimizer", "scheduler_config"]  # Don't store these
        )

        # Save attributes #
        self.encoder_embeddings = encoder_embeddings
        self.decoder_embeddings = decoder_embeddings
        self.transformer = transformer
        self.flow = flow

        self.hard_names = hard_names
        self.reco_names = reco_names
        self.process_names = process_names

        # Safety checks #
        assert self.encoder_embeddings.embed_dim == self.decoder_embeddings.embed_dim, f'Different embeding dim between encoder {self.encoder_embeddings.embed_dim} and decoder {self.decoder_embeddings.embed_dim}'
        if not self.transformer.use_null_token:
            raise RuntimeError('Transformer must have `use_null_token = True`')

        # Get flow indices #
        self.flow_indices = [
            [decoder_features.index(feature) for feature in flow_features]
            for decoder_features,flow_features in zip(
                self.decoder_embeddings.features_per_type,
                self.flow.flow_features,
            )
        ]

    def add_null_token(self,data,mask):
        null_token = torch.ones(
            (
                data[0].shape[0],
                1,
                data[0].shape[2],
            ),
        ) * -1
        data = [
            torch.cat(
                (
                    null_token.to(data[0].device),
                    data[0],
                ),
                dim = 1, # along particle axis
            )
        ] + [data[:] for data in data[1:]]
        mask = [
            torch.cat(
                (
                    torch.full((mask[0].shape[0],1),fill_value=True).to(mask[0].device),
                    mask[0],
                ),
                dim = 1,
            )
        ] + [mask[:] for mask in mask[1:]]
        return data, mask



    def forward(self,batch):
        # Extract different components #
        hard_data  = batch['hard']['data']
        hard_mask_exist = batch['hard']['mask']
        hard_weights = batch['hard']['weights']
        reco_data = batch['reco']['data']
        reco_mask_exist = batch['reco']['mask']
        reco_weights = batch['reco']['weights']

        # Add null token #
        reco_data_null, reco_mask_exist_null = self.add_null_token(reco_data,reco_mask_exist)

        # Pass through embeddings #
        hard_data = self.encoder_embeddings(hard_data)
        reco_data_null = self.decoder_embeddings(reco_data_null)

        # Make condition through transformer #
        condition = self.transformer(
            x_enc = hard_data,
            m_enc = torch.cat(hard_mask_exist,dim=1),
            x_dec = reco_data_null,
            m_dec = torch.cat(reco_mask_exist_null,dim=1),
        )

        # Split conditions to match reco data split #
        # (also skips the last condition, not needed in autoregressive)
        slices = np.r_[0,np.array([data.shape[1] for data in reco_data]).cumsum()]
        conditions = [condition[:,ni:nf,:] for ni,nf in zip(slices[:-1],slices[1:])]

        # Select the reco features for the flow #
        reco_data_flow = [
            data[...,indices]
            for data,indices in zip(reco_data,self.flow_indices)
        ]

        # Pass to flow #
        log_probs = - self.flow(reco_data_flow,conditions)
        log_probs = [log_probs[:,ni:nf] for ni,nf in zip(slices[:-1],slices[1:])]

        # Return log_probs, mask and weights #
        return log_probs, reco_mask_exist, reco_weights


    def shared_eval(self, batch, batch_idx, prefix):
        # Get log-prob loss #
        log_probs, masks, weights = self(batch)
        assert len(log_probs) == len(masks), f'{len(log_probs)} elements for log probs, and {len(masks)} elements for maskss'
        assert len(log_probs) == len(weights), f'{len(log_probs)} elements for log probs, and {len(weights)} elements for weights'

        # Check for nans and infs before setting to 0 #
        for i in range(len(log_probs)):
            assert log_probs[i].shape == masks[i].shape,    f'Type {i} : log prob has shape {log_probs[i].shape}, and masks {masks[i].shape}'
            assert log_probs[i].shape == weights[i].shape, f'Type {i} : log prob has shape {log_probs[i].shape}, and weights {weights[i].shape}'

            if torch.isnan(log_probs[i]).sum()>0:
                pass
                #where_nan = torch.where(torch.isnan(log_probs[i]))[0]
                #masks_nan = masks[i][where_nan]>0
                #where_nan = [coord[masks_nan] for coord in where_nan]
                #if where_nan[0].nelement() > 0:
                #    print (f'nans at coordinates {where_nan}')
            if torch.isinf(log_probs[i]).sum()>0:
                pass
                #where_inf = torch.where(torch.isinf(log_probs[i]))[0]
                #print (where_inf)
                #masks_inf = masks[i][where_inf]>0
                #print (masks_inf)
                #where_inf = [coord[masks_inf] for coord in where_inf]
                #print (where_inf)
                #if where_inf[0].nelement() > 0:
                #    print (f'infs at coordinates {where_inf}')
            log_probs[i] = torch.nan_to_num(log_probs[i],nan=0.0,posinf=0.,neginf=0.)

        # Loss per object #
        if self.reco_names is not None:
            assert len(log_probs) == len(self.reco_names), f'{len(log_probs)} elements for log probs, and {len(self.reco_names)} names'
            names = self.reco_names
        else:
            names = [f'type_{i}' for i in range(len(log_probs))]

        # Loop over types #
        for i in range(len(log_probs)):
            # Loop over particles within type #
            for j in range(log_probs[i].shape[1]):
                # Skip if not particle present #
                if masks[i][:,j].sum() == 0:
                    continue
                # Average over the whole batch of present particles #
                log_name = f"{prefix}/loss_{names[i]}_{j}"
                log_value = (log_probs[i][:,j] * masks[i][:,j]).sum() / masks[i][:,j].sum()
                    # Only log the log_prob for existing objects
                    # averaged over number of existing objects
                self.log(log_name,log_value,prog_bar=False)
        # Concatenate #
        log_probs = torch.cat(log_probs,dim=1)
        masks     = torch.cat(masks,dim=1)
        weights   = torch.cat(weights,dim=1)

        # Log per process #
        if 'process' in batch.keys():
            for idx in torch.unique(batch['process']).sort()[0]:
                process_idx = torch.where(batch['process'] == idx)[0]
                process_name = self.process_names[idx] if self.process_names is not None else str(idx.item())
                log_name = f"{prefix}/loss_process_{process_name}"
                log_value = (
                    (log_probs[process_idx,:] * masks[process_idx,:]).sum(dim=-1) \
                    / masks[process_idx,:].sum(dim=-1)
                ).mean()
                self.log(log_name,log_value,prog_bar=False)

        # Get total loss, weighted and averaged over existing objects and events #
        log_prob_tot = (
            (
                log_probs * masks * weights         # log prob masked and weighted
            ).sum(dim=1) / masks.sum(dim=1)         # averaged over number of existing particles
        ).mean()                                    # averaged on all events
        self.log(f"{prefix}/loss_tot", log_prob_tot, prog_bar=True)
        # Return #
        return log_prob_tot

    def sample(self,hard_data,hard_mask_exist,reco_data,reco_mask_exist,N):
        # Autoregressive : use null token then generate iteratively #
        #reco_data = [torch.ones_like(data) for data in reco_data]
        samples = [
            torch.ones(
                data.shape[0] * N,
                data.shape[1],
                data.shape[2],
            ).to(data.device)
            for data in reco_data
        ]

        # Repeat data and masks #
        reco_data = [
            data.repeat((N,1,1))
            for data in reco_data
        ]
        hard_data = [
            data.repeat((N,1,1))
            for data in hard_data
        ]
        hard_mask_exist = [
            mask.repeat((N,1))
            for mask in hard_mask_exist
        ]
        reco_mask_exist = [
            mask.repeat((N,1))
            for mask in reco_mask_exist
        ]

        # Pass hard data through embedding (done once) #
        hard_data = self.encoder_embeddings(hard_data)
        hard_mask_exist = torch.cat(hard_mask_exist,dim=1)

        # Count particles #
        n_types = [data.shape[1] for data in reco_data]
        n_particles = sum(n_types)
        slices = np.r_[0,np.array(n_types).cumsum()]

        # Generation loop #
        idx_type = 0
        idx_part = 0
        for i in range(n_particles):
            # Add null token #
            reco_data_null, reco_mask_exist_null = self.add_null_token(samples,reco_mask_exist)
            # Pass through reco embeddings (done each loop) #
            reco_data_null_embed = self.decoder_embeddings(reco_data_null)
            # Get condition #
            condition = self.transformer(
                x_enc = hard_data,
                m_enc = hard_mask_exist,
                x_dec = reco_data_null_embed,
                m_dec = torch.cat(reco_mask_exist_null,dim=1),
            )
            # Slice per type #
            conditions = [condition[:,ni:nf,:] for ni,nf in zip(slices[:-1],slices[1:])]
            reco_data_null_embed = [reco_data_null_embed[:,ni:nf,:] for ni,nf in zip(slices[:-1],slices[1:])]
            # Find index of particle that has to be generated #
            if idx_part >= n_types[idx_type]:
                idx_type += 1
                idx_part = 0
            # Sample NF #
            x = self.flow.sample(reco_data_null_embed,conditions,idx_type,idx_part).squeeze(1)
            # Add features that are not generated #
            x = torch.cat( # concat between flow samples and missing features
                [
                    reco_data[idx_type][:,idx_part,j].unsqueeze(-1) if j not in self.flow_indices[idx_type] else x[:,j].unsqueeze(-1)
                    for j in torch.arange(reco_data[idx_type].shape[-1])
                ],
                dim=-1,
            ) * reco_mask_exist[idx_type][:,idx_part].unsqueeze(-1) # set to zero missing particles
            # Put generated particle back in reco_data #
            samples[idx_type][:,idx_part,:] = x
            idx_part += 1


        # return reco data #
        return [
            data.reshape(
                N,
                data.shape[0] // N,
                data.shape[1],
                data.shape[2],
            )
            for data in samples
        ]


