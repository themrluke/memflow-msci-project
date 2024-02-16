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
                 no_max_objects=10,
                 
                 device=torch.device('cpu'),
                 dtype=torch.float32,
                 eps=1e-4):

        super(TransferFlow_Paper, self).__init__()
        
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

        self.transformer_list = nn.ModuleList([self.transformer_model for i in range(no_transformers)])

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
                              dtype=dtype,
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
                              dtype=dtype,
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
                              dtype=dtype,
                              univariate_kwargs={"bound": flow_bound }, # Keeping the flow in the [-B,B] box.
                              passes= 2 if not flow_autoregressive else flow_nfeatures)

        self.flow_exist = zuko.flows.NSF(features=flow_nfeatures,
                              context=transformer_input_features,
                              transforms=flow_ntransforms, 
                              bins=flow_bins, 
                              hidden_features=[flow_hiddenMLP_LayerDim]*flow_hiddenMLP_NoLayers, 
                              randperm=randPerm,
                              base=eval(flow_base),
                              base_args=[torch.ones(flow_nfeatures)*flow_base_first_arg, torch.ones(flow_nfeatures)*flow_base_second_arg],
                              dtype=dtype,
                              univariate_kwargs={"bound": flow_bound }, # Keeping the flow in the [-B,B] box.
                              passes= 2 if not flow_autoregressive else flow_nfeatures)
        
        if dtype == torch.float32:
            self.tgt_mask = self.tgt_mask.float()
            self.flow_pt = self.flow_pt.float()
            self.flow_phi = self.flow_phi.float()
            self.flow_eta = self.flow_eta.float()
            self.flow_exist = self.flow_exist.float()
        elif dtype == torch.float64:
            self.tgt_mask == self.tgt_mask.double()
            self.flow_pt = self.flow_pt.double()
            self.flow_phi = self.flow_phi.double()
            self.flow_eta = self.flow_eta.double()
            self.flow_exist = self.flow_exist.double()

    def disable_conditioner_regression_training(self):
        ''' Disable the conditioner regression training, but keep the
        latent space training'''
        self.cond_transformer.disable_regression_training()

    def enable_regression_training(self):
        self.cond_transformer.enable_regression_training()
        
    def forward(self,  scaling_reco_lab, scaling_partons_lab, scaling_RegressedBoost_lab, mask_reco, mask_boost):
        
        scaledLogReco_afterLin = self.gelu(self.linearDNN_reco(scaling_reco_lab) * mask_reco[..., None])
        scaledLogParton_afterLin = self.gelu(self.linearDNN_parton(scaling_partons_lab))        
                    
        output_decoder = scaledLogReco_afterLin

        for transfermer in self.transformer_list:
            output_decoder = transfermer(scaledLogParton_afterLin, output_decoder, tgt_mask=self.tgt_mask)

        # remove the null token
        no_objects_per_event = torch.sum(mask_reco[:,1:self.no_max_objects], dim=1) # compute the number of objects per event

        # I should take into consideration the first column (related to the null token), but not the last column
        conditioning_pt = output_decoder[:,:self.no_max_objects-1]

        # very important: pt on the 2nd position + remove the null token
        scaled_reco_lab_pt = scaling_reco_lab[:,1:self.no_max_objects,1].unsqueeze(dim=2)
        flow_prob_pt = self.flow_pt(conditioning_pt).log_prob(scaled_reco_lab_pt)
        flow_prob_pt_batch = torch.sum(flow_prob_pt*mask_reco[:,1:self.no_max_objects], dim=1) # take avg of masked objects
        flow_prob_pt_batch = torch.div(flow_prob_pt_batch, no_objects_per_event) # divide the total loss in the event at the no_objects_per_event
        avg_flow_prob_pt = flow_prob_pt_batch.mean()

        # very important: eta on the 3rd position + remove the null token
        conditioning_eta = torch.cat((output_decoder[:,:self.no_max_objects-1], scaled_reco_lab_pt), dim=2) # add pt in conditioning
        scaled_reco_lab_eta = scaling_reco_lab[:,1:self.no_max_objects,2].unsqueeze(dim=2)
        flow_prob_eta = self.flow_eta(conditioning_eta).log_prob(scaled_reco_lab_eta)
        flow_prob_eta_batch = torch.sum(flow_prob_eta*mask_reco[:,1:self.no_max_objects], dim=1) # take avg of masked objects
        flow_prob_eta_batch = torch.div(flow_prob_eta_batch, no_objects_per_event) # divide the total loss in the event at the no_objects_per_event
        avg_flow_prob_eta = flow_prob_eta_batch.mean()

        # very important: phi on the 4th position
        conditioning_phi = torch.cat((output_decoder[:,:self.no_max_objects-1], scaled_reco_lab_pt, scaled_reco_lab_eta), dim=2)
        scaled_reco_lab_phi = scaling_reco_lab[:,1:self.no_max_objects,3].unsqueeze(dim=2)
        flow_prob_phi = self.flow_phi(conditioning_phi).log_prob(scaled_reco_lab_phi)
        flow_prob_phi_batch = torch.sum(flow_prob_phi*mask_reco[:,1:self.no_max_objects], dim=1) # take avg of masked objects
        flow_prob_phi_batch = torch.div(flow_prob_phi_batch, no_objects_per_event) # divide the total loss in the event at the no_objects_per_event
        avg_flow_prob_phi = flow_prob_phi_batch.mean()

        random_matrix = torch.rand(scaling_reco_lab[:,1:self.no_max_objects,0].shape, device=self.device, dtype=self.dtype)

        # make the exist flag continuos
        # if exist == 0 => continuos_exist=(0.0,0.4)
        # if exist == 1 => continuos_exist=(0.6,1.0)
        continuos_exist = torch.where(scaling_reco_lab[:,1:self.no_max_objects,0] == 0,
                                      random_matrix*0.4,
                                      1-random_matrix*0.4)
        continuos_exist = continuos_exist.unsqueeze(dim=2)

        # very important: exist on the 1st position
        # exist flag depends only on the 'previous' jets and partons
        conditioning_exist = output_decoder[:,:self.no_max_objects-1]
        flow_prob_exist = self.flow_exist(conditioning_exist).log_prob(continuos_exist)
        flow_prob_exist_batch = torch.sum(flow_prob_exist*mask_reco[:,1:self.no_max_objects], dim=1) # take avg of masked objects
        flow_prob_exist_batch = torch.div(flow_prob_exist_batch, no_objects_per_event) # divide the total loss in the event at the no_objects_per_event
        avg_flow_prob_exist = flow_prob_exist_batch.mean()
                                
        return avg_flow_prob_pt, flow_prob_pt_batch, flow_prob_pt, \
                avg_flow_prob_eta, flow_prob_eta_batch, flow_prob_eta, \
                avg_flow_prob_phi, flow_prob_phi_batch, flow_prob_phi, \
                avg_flow_prob_exist, flow_prob_exist_batch, flow_prob_exist