import torch.nn as nn
import torch
import numpy as np


class ConditioningTransformerLayer(nn.Module):
    def __init__(self, no_jets, jets_features, no_lept, lepton_features, 
                 hidden_features, out_features,
                 nhead_encoder, no_layers_encoder,
                 nhead_decoder, no_layers_decoder,
                 no_decoders,
                 dim_feedforward_transformer=512,
                 aggregate=True, dtype=torch.float32):
        super().__init__()

        self.nhead_encoder = nhead_encoder
        self.nhead_decoder = nhead_decoder
        self.no_layers_encoder = no_layers_encoder
        self.no_layers_decoder = no_layers_decoder
        self.dim_feedforward_transformer = dim_feedforward_transformer
        self.no_decoders = no_decoders
        
        self.no_jets = no_jets
        self.no_lept = no_lept
        self.lin_jet = nn.Linear(in_features=jets_features,
                                 out_features=hidden_features - 1, dtype=dtype)
        self.lin_lept = nn.Linear(in_features=lepton_features,
                                  out_features=hidden_features - 1, dtype=dtype)
        self.lin_met = nn.Linear(in_features=3,
                                 out_features=hidden_features - 1, dtype=dtype)
        self.lin_boost = nn.Linear(in_features=4,
                                   out_features=hidden_features - 1, dtype=dtype)

        self.gelu = nn.GELU()
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_features,
                                                   dim_feedforward= dim_feedforward_transformer,
                                                   nhead=nhead_encoder,
                                                   batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=no_layers_encoder)

      
        self.aggregate = aggregate
        if self.aggregate:
            self.output_proj = nn.Linear(in_features=hidden_features, out_features=out_features-2)
        else:
            # do not aggregate but use the transformer encoder output to produce more decoded outputs\
            self.output_projs = nn.ModuleList([nn.Linear(in_features=hidden_features, out_features=out_features)
                                               for i in range(no_decoders)])
            
            decoder_layer = nn.TransformerEncoderLayer(d_model=hidden_features,
                                                   dim_feedforward= dim_feedforward_transformer,
                                                   nhead=nhead_decoder,
                                                   batch_first=True)

       
            self.transformer_decoders = nn.ModuleList([nn.TransformerEncoder(decoder_layer, num_layers=no_layers_decoder)
                                       for i in range(no_decoders)])
            
        

        self.register_buffer('ones', torch.ones(no_jets, 1))
        self.register_buffer('two', 2*torch.ones(no_lept, 1))
        self.register_buffer('three', 3*torch.ones(1, 1))
        self.register_buffer('four', 4*torch.ones(1, 1))
        

    def reset_parameters(self):
        self.lin_jet.reset_parameters()
        self.lin_lept.reset_parameters()
        self.lin_met.reset_parameters()
        self.lin_boost.reset_parameters()

    def forward(self, batch_jet, batch_lepton, batch_met, batch_boost, mask_jets, mask_lepton, mask_met, mask_boost):

        jets_afterLin = self.gelu(self.lin_jet(
            batch_jet) * mask_jets[:, :, None])
        lept_afterLin = self.gelu(self.lin_lept(batch_lepton))
        met_afterLin = self.gelu(self.lin_met(batch_met))
        boost_afterLin = self.gelu(self.lin_boost(batch_boost))

        batch_size = batch_jet.size(0)       

        jet_afterLin_andLabel = torch.cat((jets_afterLin, 
                                           self.ones.expand(batch_size, *list(self.ones.shape))), dim=-1)
        lept_afterLin_andLabel = torch.cat((lept_afterLin, 
                                            self.two.expand(batch_size, *list(self.two.shape))), dim=-1)
        met_afterLin_andLabel = torch.cat((met_afterLin, 
                                           self.three.expand(batch_size, *list(self.three.shape))), dim=-1)
        boost_afterLin_andLabel = torch.cat((boost_afterLin, 
                                             self.four.expand(batch_size, *list(self.four.shape))), dim=-1)

        transformer_input = torch.concat(
            (boost_afterLin_andLabel, lept_afterLin_andLabel, met_afterLin_andLabel, jet_afterLin_andLabel), dim=1)
        transformer_mask = torch.concat(
            (mask_boost, mask_lepton, mask_met, mask_jets), dim=1)

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
                    transformer_output * torch.unsqueeze(transformer_mask, -1), dim=1)  #[B, 64]
                
                conditional_out = transformer_output_sum / N_valid_objects
                
                decoder_outputs.append(proj(self.gelu(conditional_out)))

            return decoder_outputs
            
                
            
                
                
