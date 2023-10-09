import torch.nn as nn
import torch
import numpy as np
from itertools import chain

class ConditioningTransformerLayer(nn.Module):
    def __init__(self, no_jets, no_lept, input_features, 
                 hidden_features, out_features,
                 nhead_encoder, no_layers_encoder,
                 nhead_decoder, no_layers_decoder,
                 no_decoders,
                 dim_feedforward_transformer=512,
                 aggregate=True, use_latent=False,
                 out_features_latent=None,
                 no_layers_decoder_latent=None,
                 dtype=torch.float32):
        super().__init__()

        self.nhead_encoder = nhead_encoder
        self.nhead_decoder = nhead_decoder
        self.no_layers_encoder = no_layers_encoder
        self.no_layers_decoder = no_layers_decoder
        self.dim_feedforward_transformer = dim_feedforward_transformer
        self.no_decoders = no_decoders
        self.use_latent = use_latent
        
        self.no_jets = no_jets
        self.no_lept = no_lept
        
        self.lin_input = nn.Linear(in_features=input_features,
                                 out_features=hidden_features - 1, dtype=dtype)
        self.lin_boost = nn.Linear(in_features=4,
                                   out_features=hidden_features - 1, dtype=dtype)

        self.gelu = nn.GELU()
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_features,
                                                   dim_feedforward= dim_feedforward_transformer,
                                                   nhead=nhead_encoder,
                                                   batch_first=True,
                                                   dtype=dtype)

        self.transformer_encoder = nn.TransformerEncoder(
                                        encoder_layer, num_layers=no_layers_encoder)

      
        self.aggregate = aggregate
        if self.aggregate:
            self.output_proj = nn.Linear(in_features=hidden_features, out_features=out_features-2,dtype=dtype)
        else:
            # do not aggregate but use the transformer encoder output to produce more decoded outputs\
            self.output_projs = nn.ModuleList([nn.Linear(in_features=hidden_features,
                                                         out_features=out_features[i],
                                                         dtype=dtype) #out_features is an array
                                               for i in range(no_decoders)])
            
            decoder_layer = nn.TransformerEncoderLayer(d_model=hidden_features,
                                                   dim_feedforward= dim_feedforward_transformer,
                                                   nhead=nhead_decoder,
                                                   batch_first=True,
                                                   dtype=dtype)

       
            self.transformer_decoders = nn.ModuleList([nn.TransformerEncoder(decoder_layer,
                                                                             num_layers=no_layers_decoder)
                                       for i in range(no_decoders)])

            if self.use_latent:
                # Additional latent vector that is not linked to a parton regression.
                # it is separated cause it can have different number of output features
                self.out_features_latent = out_features_latent
                self.no_layers_decoder_latent = no_layers_decoder_latent
                self.latent_decoder = nn.TransformerEncoder(decoder_layer, num_layers=no_layers_decoder_latent)
                self.latent_proj = nn.Linear(in_features=hidden_features,
                                             out_features=out_features_latent,dtype=dtype)
            
        
        self.register_buffer('ones', torch.ones(no_jets, 1, dtype=dtype))
        self.register_buffer('two', 2*torch.ones(no_lept, 1,dtype=dtype))
        self.register_buffer('three', 3*torch.ones(1, 1,dtype=dtype))
        self.register_buffer('four', 4*torch.ones(1, 1,dtype=dtype))
        

    def disable_latent_training(self):
        if self.use_latent:
            for param in chain(self.latent_decoder.parameters(),
                               self.latent_proj.parameters()):
                param.requires_grad = False

    def disable_regression_training(self):
        for p in self.parameters():
            p.requires_grad = False
        # Now activating only the latent space gradients
        if self.use_latent:
            for param in chain(self.latent_decoder.parameters(),
                               self.latent_proj.parameters()):
                param.requires_grad = True

    def enable_regression_training(self):
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, batch_recoParticles, batch_boost, mask_recoParticles, mask_boost):
        
        batch_size = batch_recoParticles.size(0)

        input_afterLin = self.gelu(self.lin_input(batch_recoParticles) * mask_recoParticles[:, :, None])
        boost_afterLin = self.gelu(self.lin_boost(batch_boost))
        
        labels = torch.cat((self.ones.expand(batch_size, *list(self.ones.shape)),
                            self.two.expand(batch_size, *list(self.two.shape)),
                            self.three.expand(batch_size, *list(self.three.shape))), dim=1)
        
        recoParticles_andLabel = torch.cat((input_afterLin, labels), dim=-1)
        boost_afterLin_andLabel = torch.cat((boost_afterLin, 
                                             self.four.expand(batch_size, *list(self.four.shape))), dim=-1)
        
        transformer_input = torch.concat(
            (boost_afterLin_andLabel, recoParticles_andLabel), dim=1)
        transformer_mask = torch.concat(
            (mask_boost, mask_recoParticles), dim=1)

        tmask  = transformer_mask == 0

        transformer_output = self.transformer_encoder(
            transformer_input, src_key_padding_mask=tmask)
        
        N_valid_objects = torch.sum(transformer_mask, dim=1).unsqueeze(1)  #[B, 1]
        
        if self.aggregate:
            # `computing ther average of not masked objects`
            transformer_output_sum = torch.sum(
                transformer_output * torch.unsqueeze(transformer_mask, -1), dim=1)  #[B, 64]

            conditional_out = transformer_output_sum / N_valid_objects

            conditional_out = self.output_proj(self.gelu(conditional_out))

            x1 = (batch_boost[:, :, 3] + batch_boost[:, :, 2]) / 13000.
            x2 = (batch_boost[:, :, 3] - batch_boost[:, :, 2]) / 13000.

            return torch.cat((x1, x2,conditional_out), dim=1)

        else:
            decoder_outputs = []
            for deco,proj in zip(self.transformer_decoders, self.output_projs):
                ou_deco = deco(transformer_output, src_key_padding_mask=tmask)
                # `computing ther average of not masked objects`
                transformer_output_sum = torch.sum(
                    ou_deco * torch.unsqueeze(transformer_mask, -1), dim=1)  #[B, 64]
                
                conditional_out = transformer_output_sum / N_valid_objects
                decoder_outputs.append(proj(self.gelu(conditional_out)))

            if self.use_latent:
                # Applying the latent decoder
                ou_deco_latent = self.latent_decoder(transformer_output, src_key_padding_mask=tmask)
                # Aggregate
                transformer_latent_sum = torch.sum(
                    ou_deco_latent * torch.unsqueeze(transformer_mask, -1), dim=1)  #[B, 64]
                # project
                latent_out = transformer_latent_sum / N_valid_objects

                decoder_outputs.append(self.latent_proj(self.gelu(latent_out)))

            return decoder_outputs
            
                
            
                
                
