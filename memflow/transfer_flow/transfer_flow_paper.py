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

class TransferFlow_Paper(nn.Module):
    def __init__(self,
                 no_recoVars, no_partonVars,
                 no_recoObjects=18,
                 
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

        super(TransferFlow_Paper, self).__init__()
        
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

        # mask to keep the decoder autoregressive
        self.tgt_mask = self.transformer_model.generate_square_subsequent_mask(no_recoObjects, device=device)
        
        self.flow_pt = zuko.flows.NSF(features=flow_nfeatures,
                              context=transformer_input_features,
                              transforms=flow_ntransforms, 
                              bins=flow_bins, 
                              hidden_features=[flow_hiddenMLP_LayerDim]*flow_hiddenMLP_NoLayers, 
                              randperm=randPerm,
                              base=eval(flow_base),
                              base_args=[torch.ones(flow_nfeatures)*flow_base_first_arg, torch.ones(flow_nfeatures)*flow_base_second_arg],
                                   univariate_kwargs={"bound": flow_bound }, # Keeping the flow in the [-B,B] box.
                              passes= 2 if not flow_autoregressive else flow_nfeatures)

        self.flow_eta = zuko.flows.NSF(features=flow_nfeatures,
                              context=transformer_input_features + 1, # additional condition pt
                              transforms=flow_ntransforms, 
                              bins=flow_bins, 
                              hidden_features=[flow_hiddenMLP_LayerDim]*flow_hiddenMLP_NoLayers, 
                              randperm=randPerm,
                              base=eval(flow_base),
                              base_args=[torch.ones(flow_nfeatures)*flow_base_first_arg, torch.ones(flow_nfeatures)*flow_base_second_arg],
                                   univariate_kwargs={"bound": flow_bound }, # Keeping the flow in the [-B,B] box.
                              passes= 2 if not flow_autoregressive else flow_nfeatures)

        self.flow_phi = zuko.flows.NSF(features=flow_nfeatures,
                              context=transformer_input_features + 2, # additional condition pt/eta
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
        
    def forward(self,  scaling_reco_lab, scaling_partons_lab, scaling_RegressedBoost_lab, mask_reco, mask_boost):
        
        scaledLogReco_afterLin = self.gelu(self.linearDNN_reco(scaling_reco_lab) * mask_reco[..., None])
        scaledLogParton_afterLin = self.gelu(self.linearDNN_parton(scaling_partons_lab))
        
        output_decoder = self.transformer_model(scaledLogParton_afterLin, scaledLogReco_afterLin,
                                                tgt_mask=self.tgt_mask)
        no_objects_per_event = torch.sum(mask_reco[:,:self.no_max_objects], dim=1) # compute the number of objects per event
        
        conditioning_pt = output_decoder[:,:self.no_max_objects]
        scaled_reco_lab_pt = scaling_reco_lab[:,:self.no_max_objects,0].unsqueeze(dim=2)
        flow_prob_pt = self.flow_pt(conditioning_pt).log_prob(scaled_reco_lab_pt)
        flow_prob_pt_batch = torch.sum(flow_prob_pt*mask_reco[:,:self.no_max_objects], dim=1) # take avg of masked objects
        flow_prob_pt_batch = torch.div(flow_prob_pt_batch, no_objects_per_event) # divide the total loss in the event at the no_objects_per_event
        avg_flow_prob_pt = flow_prob_pt_batch.mean()
        
        conditioning_eta = torch.cat((output_decoder[:,:self.no_max_objects], scaled_reco_lab_pt), dim=2) # add pt in conditioning
        scaled_reco_lab_eta = scaling_reco_lab[:,:self.no_max_objects,1].unsqueeze(dim=2)
        flow_prob_eta = self.flow_eta(conditioning_eta).log_prob(scaled_reco_lab_eta)
        flow_prob_eta_batch = torch.sum(flow_prob_eta*mask_reco[:,:self.no_max_objects], dim=1) # take avg of masked objects
        flow_prob_eta_batch = torch.div(flow_prob_eta_batch, no_objects_per_event) # divide the total loss in the event at the no_objects_per_event
        avg_flow_prob_eta = flow_prob_eta_batch.mean()
        
        conditioning_phi = torch.cat((output_decoder[:,:self.no_max_objects], scaled_reco_lab_pt, scaled_reco_lab_eta), dim=2)
        scaled_reco_lab_phi = scaling_reco_lab[:,:self.no_max_objects,2].unsqueeze(dim=2)
        flow_prob_phi = self.flow_phi(conditioning_phi).log_prob(scaled_reco_lab_phi)
        flow_prob_phi_batch = torch.sum(flow_prob_phi*mask_reco[:,:self.no_max_objects], dim=1) # take avg of masked objects
        flow_prob_phi_batch = torch.div(flow_prob_phi_batch, no_objects_per_event) # divide the total loss in the event at the no_objects_per_event
        avg_flow_prob_phi = flow_prob_phi_batch.mean()
                                
        return avg_flow_prob_pt, flow_prob_pt_batch, flow_prob_pt, \
                avg_flow_prob_eta, flow_prob_eta_batch, flow_prob_eta, \
                avg_flow_prob_phi, flow_prob_phi_batch, flow_prob_phi