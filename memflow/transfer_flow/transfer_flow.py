import torch.nn as nn
import torch
import numpy as np
import utils
from memflow.unfolding_network.conditional_transformer import ConditioningTransformerLayer
import zuko
from zuko.flows import TransformModule, SimpleAffineTransform
from zuko.distributions import BoxUniform
from zuko.distributions import DiagNormal
from memflow.unfolding_flow.utils import Compute_ParticlesTensor as particle_tools
import memflow.phasespace.utils as ps_utils

class TransferFlow(nn.Module):
    def __init__(self,
                 no_recoVars, no_partonVars,
                 
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
                 no_max_objects=10,
                 
                 device=torch.device('cpu'),
                 dtype=torch.float32,
                 eps=1e-4):

        super(TransferFlow, self).__init__()
        
        self.device = device
        self.dtype = dtype
        self.eps = eps # used for small values like the mass of the gluon for numerical reasons
        
        self.linearDNN_reco = nn.Linear(in_features=no_recoVars, out_features=transformer_input_features)
        self.linearDNN_parton = nn.Linear(in_features=no_partonVars, out_features=transformer_input_features)
        self.linearDNN_boost = nn.Linear(in_features=4, out_features=transformer_input_features)
        self.gelu = nn.GELU()
        self.no_max_objects = no_max_objects
        
        self.transformer_model = nn.Transformer(d_model=transformer_input_features,
                                                nhead=transformer_nhead,
                                                num_encoder_layers=transformer_num_encoder_layers,
                                                num_decoder_layers=transformer_num_decoder_layers,
                                                dim_feedforward=transformer_dim_feedforward,
                                                activation=transformer_activation,
                                                batch_first=True)
         
        
        self.flow = zuko.flows.NSF(features=flow_nfeatures,
                              context=transformer_input_features,
                              transforms=flow_ntransforms, 
                              bins=flow_bins, 
                              hidden_features=[flow_hiddenMLP_LayerDim]*flow_hiddenMLP_NoLayers, 
                              randperm=randPerm,
                              base=eval(flow_base),
                              base_args=[torch.ones(flow_nfeatures)*flow_base_first_arg, torch.ones(flow_nfeatures)*flow_base_second_arg],
                                   univariate_kwargs={"bound": flow_bound }, # Keeping the flow in the [-B,B] box.
                              passes= 2 if not flow_autoregressive else flow_nfeatures)
        

    def disable_conditioner_regression_training(self):
        ''' Disable the conditioner regression training, but keep the
        latent space training'''
        self.cond_transformer.disable_regression_training()

    def enable_regression_training(self):
        self.cond_transformer.enable_regression_training()
        
    def forward(self,  scaling_reco_lab, scaling_partons_lab, scaling_RegressedBoost_lab,
                mask_reco, mask_boost, flow_eval="normalizing", Nsamples=0):
        
        scaledLogReco_afterLin = self.gelu(self.linearDNN_reco(scaling_reco_lab) * mask_reco[..., None])
        scaledLogParton_afterLin = self.gelu(self.linearDNN_parton(scaling_partons_lab))
        #boost_afterLin = self.gelu(self.linearDNN_boost(scaling_RegressedBoost_lab))
        #scaledLogReco_withBoost_afterLin = torch.cat((scaledLogReco_afterLin, boost_afterLin), dim=1)
        
        
        tgt_mask = self.transformer_model.generate_square_subsequent_mask(scaledLogReco_afterLin.size(1))
        output_decoder = self.transformer_model(scaledLogParton_afterLin, scaledLogReco_afterLin,
                                          tgt_mask=tgt_mask)
        
        flow_prob = self.flow(output_decoder[:,:self.no_max_objects]).log_prob(scaling_reco_lab[:,:self.no_max_objects,:3])
        
        #tgt_mask = self.transformer_model.generate_square_subsequent_mask(scaledLogReco_withBoost_afterLin.size(1))
        #output_decoder = self.transformer_model(scaledLogParton_afterLin, scaledLogReco_withBoost_afterLin,
        #                                  tgt_mask=tgt_mask)
        
        #scaling_reco_lab_andBoost = torch.cat((scaling_reco_lab[:,:,:3], scaling_RegressedBoost_lab[:,:,1:]), dim=1)
        #mask_reco_andBoost = torch.cat((mask_reco, mask_boost), dim=1)
        #flow_prob = self.flow(output_decoder[:,:no_max_objects]).log_prob(scaling_reco_lab_andBoost[:,:no_max_objects])
        # i will want to add the boost in the target too
        
        flow_prob_batch = torch.sum(flow_prob*mask_reco[:,:self.no_max_objects], dim=1) # take avg of masked objects
        avg_flow_prob = flow_prob_batch.mean() # is this good?? 
                
        return avg_flow_prob, flow_prob_batch, flow_prob
    
        

