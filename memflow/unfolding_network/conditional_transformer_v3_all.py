import torch.nn as nn
import torch
import numpy as np
from itertools import chain

class ConditioningTransformerLayer_v3_all(nn.Module):
    def __init__(self, no_recoVars, no_partonVars,
                 hidden_features,
                 DNN_input,
                 nhead_encoder, no_layers_encoder,
                 no_layers_decoder,
                 transformer_activation=nn.GELU(),
                 dim_feedforward_transformer=512,
                 aggregate=True, use_latent=False,
                 DNN_layers=3,
                 DNN_nodes=1024,
                 out_features_latent=None,
                 no_layers_decoder_latent=None,
                 arctanh=False,
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
        self.arctanh = arctanh

        # WHAT I CAN DO TO DECODE:
        # Initialize some partons as null tokens => 4 null tokens (positional encoding)
        # sin cos embedding + same linear DNN

        self.linearDNN_reco = nn.Linear(in_features=no_recoVars, out_features=hidden_features, dtype=dtype)
        self.linearDNN_parton = nn.Linear(in_features=1, out_features=hidden_features, dtype=dtype)
        self.linearDNN_boost = nn.Linear(in_features=2, out_features=hidden_features, dtype=dtype)

        # TODO: tune dropout of this model
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

        layers = [nn.Linear(DNN_input, DNN_nodes, dtype=dtype), nn.GELU()] 
        layers_boost = [nn.Linear(DNN_input, DNN_nodes, dtype=dtype), nn.GELU()]
        for i in range(DNN_layers - 1):
            layers.append(nn.Linear(DNN_nodes, DNN_nodes, dtype=dtype))
            layers.append(nn.GELU())
            
            layers_boost.append(nn.Linear(DNN_nodes, DNN_nodes, dtype=dtype))
            layers_boost.append(nn.GELU())
        
        
        if self.aggregate:
            self.output_proj = nn.Linear(in_features=hidden_features, out_features=out_features-2,dtype=dtype)
        else:
            # do not aggregate but use the transformer encoder output to produce more decoded outputs
            self.out_features = [3, 2]

            # final layer activation = linenar
            # try GeLU and SeLU
            layers.append(nn.Linear(DNN_nodes, self.out_features[0], dtype=dtype))
            layers_boost.append(nn.Linear(DNN_nodes, self.out_features[1], dtype=dtype))

            model = nn.Sequential(*layers)
            model_boost = nn.Sequential(*layers_boost)
            
            self.output_projs = nn.ModuleList([model, model_boost]) # no decoders = max no of regressions

            if self.use_latent:
                # Additional latent vector that is not linked to a parton regression.
                # it is separated cause it can have different number of output features
                self.out_features_latent = out_features_latent
                self.no_layers_decoder_latent = no_layers_decoder_latent
                #self.latent_decoder = nn.TransformerEncoder(decoder_layer, num_layers=no_layers_decoder_latent)
                self.latent_proj = nn.Linear(in_features=hidden_features,
                                             out_features=out_features_latent,
                                             dtype=dtype)
        

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

    def forward(self, batch_recoParticles, reco_boost, mask_recoParticles, mask_boost, No_regressed_vars, sin_cos_reco, sin_cos_partons, sin_cos_embedding=False, attach_position=None, eps_arctanh=0.):
        
        batch_size = batch_recoParticles.size(0)

        # create null token and its mask
        null_tokens = torch.ones((batch_size, No_regressed_vars, 1), device=self.device, dtype=self.dtype) * -1

        if sin_cos_embedding:

            input_afterLin = self.gelu((self.linearDNN_reco(batch_recoParticles) + sin_cos_reco[:]) * mask_recoParticles[:, :, None])
            boost_afterLin = self.gelu(self.linearDNN_boost(reco_boost))
            partons_afterLin = self.gelu(self.linearDNN_parton(null_tokens) + sin_cos_partons)

        else:
            input_afterLin = self.gelu((self.linearDNN_reco(batch_recoParticles)) * mask_recoParticles[:, :, None])
            boost_afterLin = self.gelu(self.linearDNN_boost(reco_boost))  # reco_boost = [E, pz]
            partons_afterLin = self.gelu(self.linearDNN_parton(null_tokens))

        transformer_input = torch.concat((input_afterLin, boost_afterLin), dim=1)
        transformer_mask = torch.concat((mask_recoParticles, mask_boost), dim=1)

        tmask  = transformer_mask == 0

        # partons_afterLin -> actually null tokens after lin 
        transformer_output = self.transformer_model(transformer_input, partons_afterLin) # no tgt mask => no autoreg structure

        if attach_position != None:
            attach_position = attach_position.expand(transformer_output.shape[0], -1).unsqueeze(dim=2)
            transformer_output = torch.cat((transformer_output, attach_position), dim=2)

        if self.arctanh == True:
            decay_vars_arctanh_phi = self.output_projs[0](transformer_output[:,:-1]) # to be changed when considering the boost
            phi_angle = (np.pi + eps_arctanh)*torch.tanh(decay_vars_arctanh_phi[...,-1:])
            decay_vars = torch.cat((decay_vars_arctanh_phi[:,:,:2], phi_angle), dim=2)
    
        else:
            decay_vars = self.output_projs[0](transformer_output[:,:-1]) # to be changed when considering the boost
        
        boost = self.output_projs[1](transformer_output[:,-1:])

        return decay_vars, boost
            
                
            
                
                
