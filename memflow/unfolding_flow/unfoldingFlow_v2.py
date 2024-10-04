import torch.nn as nn
import torch
import numpy as np
import utils
import zuko
from zuko.flows import TransformModule, SimpleAffineTransform
from zuko.distributions import BoxUniform
from zuko.distributions import DiagNormal
from memflow.unfolding_flow.utils import Compute_ParticlesTensor as particle_tools
import memflow.phasespace.utils as ps_utils

class UnfoldingFlow_v2(nn.Module):
    def __init__(self,
                 no_recoVars, no_partonVars,
                 no_partons=9,

                 no_transformers=1,
                 transformer_input_features=64,
                 transformer_nhead=8,
                 transformer_num_encoder_layers=4,
                 transformer_num_decoder_layers=4,
                 transformer_dim_feedforward=128,
                 transformer_activation=nn.GELU(),
                 
                 flow_nfeatures=12,
                 flow_ntransforms=5,
                 flow_hiddenMLP_NoLayers=4,
                 flow_hiddenMLP_LayerDim=128,
                 flow_bins=16,
                 flow_autoregressive=True,
                 flow_base=BoxUniform,
                 flow_base_first_arg=-1,
                 flow_base_second_arg=1,
                 flow_bound=1.,
                 randPerm=False,

                 flow_lepton_ntransforms=3,
                 flow_lepton_hiddenMLP_NoLayers=4,
                 flow_lepton_hiddenMLP_LayerDim=128,
                 flow_lepton_bins=16,

                 encode_position=True,
                 
                 device=torch.device('cpu'),
                 dtype=torch.float32,
                 eps=1e-4):

        super(UnfoldingFlow_v2, self).__init__()
        
        self.device = device
        self.dtype = dtype
        self.eps = eps # used for small values like the mass of the gluon for numerical reasons
        self.encode_position = encode_position
        
        self.linearDNN_reco = nn.Linear(in_features=no_recoVars, out_features=transformer_input_features, dtype=dtype)
        self.linearDNN_parton = nn.Linear(in_features=no_partonVars, out_features=transformer_input_features, dtype=dtype)
        self.linearDNN_boost = nn.Linear(in_features=4, out_features=transformer_input_features, dtype=dtype)
        self.gelu = nn.GELU()

        # flow for neutrino
        self.flow_kinematics_partons = zuko.flows.NSF(features=flow_nfeatures,
                              context=transformer_input_features + 1, # additional position of 'jet'
                              transforms=flow_lepton_ntransforms, 
                              bins=flow_lepton_bins, 
                              hidden_features=[flow_lepton_hiddenMLP_LayerDim]*flow_lepton_hiddenMLP_NoLayers, 
                              randperm=randPerm,
                              base=eval(flow_base),
                              base_args=[torch.ones(flow_nfeatures)*flow_base_first_arg, torch.ones(flow_nfeatures)*flow_base_second_arg],
                              dtype=dtype,
                              univariate_kwargs={"bound": flow_bound }, # Keeping the flow in the [-B,B] box.
                              passes= 2 if not flow_autoregressive else flow_nfeatures)

        self.transformer_model = nn.Transformer(d_model=transformer_input_features,
                                                nhead=transformer_nhead,
                                                num_encoder_layers=transformer_num_encoder_layers,
                                                num_decoder_layers=transformer_num_decoder_layers,
                                                dim_feedforward=transformer_dim_feedforward,
                                                activation=transformer_activation,
                                                batch_first=True,
                                                dtype=dtype)

        # mask to keep the decoder autoregressive (add 1 for the null token)
        self.tgt_mask = self.transformer_model.generate_square_subsequent_mask(no_partons+1, device=device)

        hot_encoded = [i for i in range(no_partons + 1)]
        self.hot_encoded = torch.tensor(hot_encoded, device=self.device, dtype=self.dtype).unsqueeze(dim=1)
        
        if dtype == torch.float32:
            self.tgt_mask = self.tgt_mask.float()
            self.flow_kinematics_partons = self.flow_kinematics_partons.float()
        elif dtype == torch.float64:
            self.tgt_mask == self.tgt_mask.double()
            self.flow_kinematics_partons = self.flow_kinematics_partons.double()
        
    def forward(self, scaling_reco_lab, scaling_partons_lab, scaling_RegressedBoost_lab,
                mask_partons, mask_reco, mask_boost):

        # create null token and its mask
        null_token = torch.ones((scaling_partons_lab.shape[0], 1, 5), device=self.device, dtype=self.dtype) * -1
        # mask for the null token = True
        null_token_mask = torch.ones((mask_partons.shape[0], 1), device=self.device, dtype=torch.bool)

        # attach null token and update the mask for the scaling_reco_lab
        scaling_parton_lab_withNullToken = torch.cat((null_token, scaling_partons_lab), dim=1)
        mask_partons_withNullToken = torch.cat((null_token_mask, mask_partons), dim=1)
        
        scaledLogReco_afterLin = self.gelu(self.linearDNN_reco(scaling_reco_lab) * mask_reco[..., None])
        scaledLogParton_afterLin = self.gelu(self.linearDNN_parton(scaling_parton_lab_withNullToken) * mask_partons_withNullToken[...,None])      

        # output decoder shape: [events, null_token+jets, 64]
        # first elem of output decoder -> null token
        output_decoder = self.transformer_model(scaledLogReco_afterLin, scaledLogParton_afterLin, tgt_mask=self.tgt_mask)

        if self.encode_position:
            hot_encoded = self.hot_encoded.expand(output_decoder.shape[0], -1, -1)
            output_decoder = torch.cat((output_decoder, hot_encoded), dim=2)

        conditioning = output_decoder

        #print(conditioning.shape)
        #print(scaling_partons_lab.shape)

        # partons flow
        flow_prob_partons = self.flow_kinematics_partons(conditioning[:,:-1]).log_prob(scaling_partons_lab[...,:3])
        
        # compute flow prob per event
        flow_prob_batch = torch.sum(flow_prob_partons*mask_partons, dim=1) # take avg of masked objects
        no_objects_batch = torch.count_nonzero(mask_partons, dim=1)
        avg_flow_prob = (flow_prob_batch/no_objects_batch).mean()
                                
        return avg_flow_prob, flow_prob_batch, flow_prob_partons