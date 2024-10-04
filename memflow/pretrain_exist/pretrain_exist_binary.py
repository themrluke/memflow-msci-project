import torch.nn as nn
import torch
import numpy as np
import zuko
#from zuko.flows import TransformModule, SimpleAffineTransform
from zuko.distributions import BoxUniform
from zuko.distributions import DiagNormal
from memflow.unfolding_flow.utils import Compute_ParticlesTensor as particle_tools
import memflow.phasespace.utils as ps_utils

class Classify_ExistJet(nn.Module):
    def __init__(self,
                 no_recoVars, no_partonVars,
                 no_recoObjects=18,

                 no_transformers=1,
                 transformer_input_features=64,
                 transformer_nhead=8,
                 transformer_num_encoder_layers=4,
                 transformer_num_decoder_layers=4,
                 transformer_dim_feedforward=128,
                 transformer_activation=nn.GELU(),

                 DNN_nodes=64,
                 DNN_layers=8,
                 no_max_objects=10,
                 dropout=False,
                 
                 device=torch.device('cpu'),
                 dtype=torch.float32,
                 eps=1e-4):

        super(Classify_ExistJet, self).__init__()
        
        self.device = device
        self.dtype = dtype
        self.eps = eps # used for small values like the mass of the gluon for numerical reasons
        
        self.linearDNN_reco = nn.Linear(in_features=no_recoVars, out_features=transformer_input_features, dtype=dtype)
        self.linearDNN_parton = nn.Linear(in_features=no_partonVars, out_features=transformer_input_features, dtype=dtype)
        self.linearDNN_boost = nn.Linear(in_features=4, out_features=transformer_input_features, dtype=dtype)
        self.gelu = nn.GELU()
        self.no_max_objects = no_max_objects
        
        self.transformer_model = nn.Transformer(d_model=transformer_input_features,
                                                nhead=transformer_nhead,
                                                num_encoder_layers=transformer_num_encoder_layers,
                                                num_decoder_layers=transformer_num_decoder_layers,
                                                dim_feedforward=transformer_dim_feedforward,
                                                activation=transformer_activation,
                                                batch_first=True,
                                                dtype=dtype)

        # mask to keep the decoder autoregressive (add 1 for the null token)
        self.tgt_mask = self.transformer_model.generate_square_subsequent_mask(no_recoObjects+1, device=device)

        layers = [nn.Linear(transformer_input_features + 1, DNN_nodes), nn.GELU()] # + 1 because we have added the encoded position in the output_decoder
        for i in range(DNN_layers - 1):
            if dropout:
                layers.extend([nn.Linear(DNN_nodes, DNN_nodes), nn.GELU(), nn.Dropout(0.2)])
            else:
                layers.extend([nn.Linear(DNN_nodes, DNN_nodes), nn.GELU()])
        layers.append(nn.Linear(DNN_nodes, 1))
        layers.append(nn.Sigmoid())
    
        self.model = nn.Sequential(*layers)

        #print(self.model)

        hot_encoded = [i for i in range(no_recoObjects + 1)]
        self.hot_encoded = torch.tensor(hot_encoded, device=self.device, dtype=self.dtype).unsqueeze(dim=1)

        if dtype == torch.float32:
            self.tgt_mask = self.tgt_mask.float()
        elif dtype == torch.float64:
            self.tgt_mask == self.tgt_mask.double()

    def disable_conditioner_regression_training(self):
        ''' Disable the conditioner regression training, but keep the
        latent space training'''
        self.cond_transformer.disable_regression_training()

    def enable_regression_training(self):
        self.cond_transformer.enable_regression_training()
        
    def forward(self, scaling_reco_lab, scaling_partons_lab, scaling_RegressedBoost_lab,
                mask_reco, mask_boost, makeExistContinuos=0.4, encode_position=True):

        # create null token and its mask
        null_token = torch.ones((scaling_reco_lab.shape[0], 1, 5), device=self.device, dtype=self.dtype) * -1
        null_token[:,0,0] = 0 # exist flag = 0 not -1
        # mask for the null token = True
        null_token_mask = torch.ones((mask_reco.shape[0], 1), device=self.device, dtype=torch.bool)

        # attach null token and update the mask for the scaling_reco_lab
        scaling_reco_lab_withNullToken = torch.cat((null_token, scaling_reco_lab), dim=1)
        mask_reco_withNullToken = torch.cat((null_token_mask, mask_reco), dim=1)
        
        scaledLogReco_afterLin = self.gelu(self.linearDNN_reco(scaling_reco_lab_withNullToken) * mask_reco_withNullToken[..., None])
        scaledLogParton_afterLin = self.gelu(self.linearDNN_parton(scaling_partons_lab))        

        # output decoder shape: [events, null_token+jets, 64]
        # first elem of output decoder -> null token
        output_decoder = self.transformer_model(scaledLogParton_afterLin, scaledLogReco_afterLin, tgt_mask=self.tgt_mask)

        if encode_position:
            hot_encoded = self.hot_encoded.expand(output_decoder.shape[0], -1, -1)
            output_decoder = torch.cat((output_decoder, hot_encoded), dim=2)

        no_objects_per_event = torch.sum(mask_reco[:,:self.no_max_objects], dim=1) # compute the number of objects per event

        prob_each_jet = self.model(output_decoder).squeeze(dim=2)
        prob_each_jet = prob_each_jet[:,:self.no_max_objects]
        prob_each_event = torch.sum(prob_each_jet*mask_reco[:,:self.no_max_objects], dim=1) # take avg of masked objects
        #prob_each_event = torch.div(prob_each_event, no_objects_per_event) # divide the total loss in the event at the no_objects_per_event
        prob_avg = prob_each_event.mean()
                                
        return prob_avg, prob_each_event, prob_each_jet