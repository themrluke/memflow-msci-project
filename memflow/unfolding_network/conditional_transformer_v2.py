import torch.nn as nn
import torch
import numpy as np
from itertools import chain

class ConditioningTransformerLayer_v2(nn.Module):
    def __init__(self, no_recoVars, no_partonVars,
                 hidden_features,
                 nhead_encoder, no_layers_encoder,
                 no_layers_decoder,
                 transformer_activation=nn.GELU(),
                 dim_feedforward_transformer=512,
                 aggregate=True, use_latent=False,
                 out_features_latent=None,
                 no_layers_decoder_latent=None,
                 dtype=torch.float32,
                 device=torch.device('cpu')):
        super().__init__()

        self.nhead_encoder = nhead_encoder
        self.no_layers_encoder = no_layers_encoder
        self.no_layers_decoder = no_layers_decoder
        self.dim_feedforward_transformer = dim_feedforward_transformer
        self.use_latent = use_latent
        self.dtype = dtype
        self.device = device

        # WHAT I CAN DO TO DECODE:
        # Initialize some partons as null tokens => 4 null tokens (positional encoding)
        # sin cos embedding + same linear DNN

        self.linearDNN_reco = nn.Linear(in_features=no_recoVars, out_features=hidden_features, dtype=dtype)
        self.linearDNN_parton = nn.Linear(in_features=1, out_features=hidden_features, dtype=dtype)
        self.linearDNN_boost = nn.Linear(in_features=4, out_features=hidden_features, dtype=dtype)
        
        self.transformer_model = nn.Transformer(d_model=hidden_features,
                                                nhead=nhead_encoder,
                                                num_encoder_layers=no_layers_encoder,
                                                num_decoder_layers=no_layers_decoder,
                                                dim_feedforward=dim_feedforward_transformer,
                                                activation=transformer_activation,
                                                batch_first=True,
                                                dtype=dtype)

        self.gelu = nn.GELU()

        self.aggregate = aggregate
        if self.aggregate:
            self.output_proj = nn.Linear(in_features=hidden_features, out_features=out_features-2,dtype=dtype)
        else:
            # do not aggregate but use the transformer encoder output to produce more decoded outputs
            out_features = [3,1,2] 
            self.output_projs = nn.ModuleList([nn.Linear(in_features=hidden_features,
                                                         out_features=out_features[i],
                                                         dtype=dtype) #out_features is an array
                                               for i in range(3)]) # no decoders = max no of regressions

            if self.use_latent:
                # Additional latent vector that is not linked to a parton regression.
                # it is separated cause it can have different number of output features
                self.out_features_latent = out_features_latent
                self.no_layers_decoder_latent = no_layers_decoder_latent
                #self.latent_decoder = nn.TransformerEncoder(decoder_layer, num_layers=no_layers_decoder_latent)
                self.latent_proj = nn.Linear(in_features=hidden_features,
                                             out_features=out_features_latent,
                                             dtype=dtype)
            
        
        #self.register_buffer('ones', torch.ones(no_jets, 1, dtype=dtype))
        #self.register_buffer('two', 2*torch.ones(no_lept, 1,dtype=dtype))
        #self.register_buffer('three', 3*torch.ones(1, 1,dtype=dtype))
        #self.register_buffer('four', 4*torch.ones(1, 1,dtype=dtype))
        

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

    def forward(self, batch_recoParticles, batch_boost, mask_recoParticles, mask_boost, No_regressed_vars, sin_cos_reco, sin_cos_partons):
        
        batch_size = batch_recoParticles.size(0)

        # create null token and its mask
        null_tokens = torch.ones((batch_size, No_regressed_vars, 1), device=self.device, dtype=self.dtype) * -1

        input_afterLin = self.gelu((self.linearDNN_reco(batch_recoParticles) + sin_cos_reco[1:]) * mask_recoParticles[:, :, None])
        boost_afterLin = self.gelu(self.linearDNN_boost(batch_boost) + sin_cos_reco[:1])
        partons_afterLin = self.gelu(self.linearDNN_parton(null_tokens) + sin_cos_partons)

        transformer_input = torch.concat((boost_afterLin, input_afterLin), dim=1)
        transformer_mask = torch.concat((mask_boost, mask_recoParticles), dim=1)

        tmask  = transformer_mask == 0

        # partons_afterLin -> actually null tokens after lin 
        transformer_output = self.transformer_model(transformer_input, partons_afterLin) # no tgt mask => no autoreg structure

        intermediate_propagators = self.output_projs[0](transformer_output[:,:3])
        boost = self.output_projs[1](transformer_output[:,3])
        decay_angles = self.output_projs[2](transformer_output[:,4:])
        

        if False:
        
            if self.aggregate:
                # `computing ther average of not masked objects`
                transformer_output_sum = torch.sum(
                    transformer_output * torch.unsqueeze(transformer_mask, -1), dim=1)  #[B, 64]
    
                conditional_out = transformer_output_sum / N_valid_objects
    
                conditional_out = self.output_proj(self.gelu(conditional_out))
    
                x1 = (batch_boost[:, :, 3] + batch_boost[:, :, 2]) / 13000.
                x2 = (batch_boost[:, :, 3] - batch_boost[:, :, 2]) / 13000.
    
                return torch.cat((x1, x2, conditional_out), dim=1)
    
            else:
                decoder_outputs = []
                
                for proj in zip(self.output_projs):
                    # `computing ther average of not masked objects`
                    transformer_output_sum = torch.sum(
                        transformer_output * torch.unsqueeze(transformer_mask, -1), dim=1)  #[B, 64]
                    
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

            labels = torch.cat((self.ones.expand(batch_size, *list(self.ones.shape)),
                                self.two.expand(batch_size, *list(self.two.shape)),
                                self.three.expand(batch_size, *list(self.three.shape))), dim=1)
            
            recoParticles_andLabel = torch.cat((input_afterLin, labels), dim=-1)
            boost_afterLin_andLabel = torch.cat((boost_afterLin, 
                                                 self.four.expand(batch_size, *list(self.four.shape))), dim=-1)
        

        return intermediate_propagators, boost, decay_angles
            
                
            
                
                
