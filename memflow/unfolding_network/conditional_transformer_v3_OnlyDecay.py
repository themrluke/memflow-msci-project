import torch.nn as nn
import torch
import numpy as np
from itertools import chain

class ConditioningTransformerLayer_v3(nn.Module):
    def __init__(self, no_recoVars, no_partonVars,
                 hidden_features,
                 DNN_input,
                 nhead_encoder, no_layers_encoder,
                 no_layers_decoder,
                 transformer_activation=nn.GELU(),
                 dim_feedforward_transformer=512,
                 aggregate=True,
                 DNN_layers=3,
                 DNN_nodes=1024,
                 sin_cos=False,
                 arctanh=False,
                 angles_CM=False,
                 dtype=torch.float32,
                 device=torch.device('cpu')):
        super().__init__()

        self.nhead_encoder = nhead_encoder
        self.no_layers_encoder = no_layers_encoder
        self.no_layers_decoder = no_layers_decoder
        self.dim_feedforward_transformer = dim_feedforward_transformer
        self.dtype = dtype
        self.device = device
        self.arctanh = arctanh
        self.angles_CM = angles_CM

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
        layers_angles = [nn.Linear(DNN_input, DNN_nodes, dtype=dtype), nn.GELU()]
        
        for i in range(DNN_layers - 1):
            layers.append(nn.Linear(DNN_nodes, DNN_nodes, dtype=dtype))
            layers.append(nn.GELU())
            
            layers_boost.append(nn.Linear(DNN_nodes, DNN_nodes, dtype=dtype))
            layers_boost.append(nn.GELU())

            layers_angles.append(nn.Linear(DNN_nodes, DNN_nodes, dtype=dtype))
            layers_angles.append(nn.GELU())
        
        
        if self.aggregate:
            self.output_proj = nn.Linear(in_features=hidden_features, out_features=out_features-2,dtype=dtype)
        else:
            # do not aggregate but use the transformer encoder output to produce more decoded outputs
            #out_features = [3,1,2] 
            self.out_features = [3, 2]
            if sin_cos:
                self.out_features = [4, 2]
            elif angles_CM:
                self.out_features = [3, 2, 2]

            # final layer activation = linenar
            # try GeLU and SeLU
            layers.append(nn.Linear(DNN_nodes, self.out_features[0], dtype=dtype))
            layers_boost.append(nn.Linear(DNN_nodes, self.out_features[1], dtype=dtype))
            if angles_CM:
                layers_angles.append(nn.Linear(DNN_nodes, self.out_features[2], dtype=dtype))
                model_angles = nn.Sequential(*layers_angles)
    
            model = nn.Sequential(*layers)
            model_boost = nn.Sequential(*layers_boost)
            
            self.output_projs = nn.ModuleList([model, model_boost]) # no decoders = max no of regressions
            if angles_CM:
                self.output_projs = nn.ModuleList([model, model_boost, model_angles]) # no decoders = max no of regressions
        

    def disable_regression_training(self):
        for p in self.parameters():
            p.requires_grad = False

    def enable_regression_training(self):
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, batch_recoParticles, reco_boost, mask_recoParticles, mask_boost, No_regressed_vars, sin_cos_reco, sin_cos_partons, sin_cos_embedding=False, attach_position=None, eps_arctanh=0.):
        
        batch_size = batch_recoParticles.size(0)

        # create null token and its mask
        null_tokens = torch.ones((batch_size, No_regressed_vars, 1), device=self.device, dtype=self.dtype) * -1

        if sin_cos_embedding:

            input_afterLin = self.gelu((self.linearDNN_reco(batch_recoParticles) + sin_cos_reco[:]) * mask_recoParticles[:, :, None])
            #boost_afterLin = self.gelu(self.linearDNN_boost(batch_boost) + sin_cos_reco[-1:])
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

        boost = self.output_projs[-1](transformer_output[:,8:9])
        
        if No_regressed_vars > 9:
            free_latent_space = transformer_output[:,9:10]
        else:
            # empty vector
            free_latent_space = torch.empty((transformer_output.shape[0], 1, 0), device=self.device, dtype=self.dtype)

        if self.out_features[0] == 4:
            # SIN PHI
            decay_vars_cos_phi = self.output_projs[0](transformer_output[:,:-1]) # to be changed when considering the boost
            phi_angle = torch.atan2(decay_vars_cos_phi[:,:,-1:], decay_vars_cos_phi[:,:,-2:-1])
            decay_vars = torch.cat((decay_vars_cos_phi[:,:,:2], phi_angle), dim=2)

        elif self.angles_CM:
            propagators_momenta = self.output_projs[0](transformer_output[:,:3]) # higgs, thad, tlep
            decayVars_momenta = self.output_projs[1](transformer_output[:,3:8]) # higgs_b1, thad_b, thad_q, tlep_b, tlep_q
            if self.arctanh:
                propagators_phi = (np.pi + eps_arctanh)*torch.tanh(propagators_momenta[...,-1:])
                decayVars_phi = (np.pi + eps_arctanh)*torch.tanh(decayVars_momenta[...,-1:])
                propagators_momenta = torch.cat((propagators_momenta[...,:2], propagators_phi), dim=2)
                decayVars_momenta = torch.cat((decayVars_momenta[...,:1], decayVars_phi), dim=2)

            return propagators_momenta, decayVars_momenta, boost, free_latent_space

        elif self.arctanh:
            decay_vars_arctanh_phi = self.output_projs[0](transformer_output[:,:-1]) # to be changed when considering the boost
            phi_angle = (np.pi + eps_arctanh)*torch.tanh(decay_vars_arctanh_phi[...,-1:])
            decay_vars = torch.cat((decay_vars_arctanh_phi[:,:,:2], phi_angle), dim=2)
    
        else:
            decay_vars = self.output_projs[0](transformer_output[:,:-1]) # to be changed when considering the boost
        
        

        return decay_vars, boost, free_latent_space
            
                
            
                
                
