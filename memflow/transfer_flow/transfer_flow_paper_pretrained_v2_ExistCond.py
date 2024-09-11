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

from memflow.pretrain_exist.pretrain_exist_binary import Classify_ExistJet

class TransferFlow_Paper_pretrained_v2_ExistCond(nn.Module):
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

                 DNN_nodes=64, DNN_layers=8,
                 pretrained_classifier='',
                 load_classifier=False,
                 freeze_classifier=True,
                 encode_position=True,
                 
                 device=torch.device('cpu'),
                 dtype=torch.float32,
                 eps=1e-4):

        super(TransferFlow_Paper_pretrained_v2_ExistCond, self).__init__()
        
        self.device = device
        self.dtype = dtype
        self.eps = eps # used for small values like the mass of the gluon for numerical reasons
        self.encode_position = encode_position
        
        self.linearDNN_reco = nn.Linear(in_features=no_recoVars, out_features=transformer_input_features, dtype=dtype)
        self.linearDNN_parton = nn.Linear(in_features=no_partonVars, out_features=transformer_input_features, dtype=dtype)
        self.linearDNN_boost = nn.Linear(in_features=4, out_features=transformer_input_features, dtype=dtype)
        self.gelu = nn.GELU()
        self.no_max_objects = no_max_objects
        
        self.flow_pt = zuko.flows.NSF(features=flow_nfeatures,
                              context=transformer_input_features+2, # additional position of 'jet' + output classifier
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
                              context=transformer_input_features + 3, # additional position of 'jet' + condition pt + output classifier
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
                              context=transformer_input_features + 4, # additional position of 'jet' + condition pt/eta + output classifier
                              transforms=flow_ntransforms, 
                              bins=flow_bins, 
                              hidden_features=[flow_hiddenMLP_LayerDim]*flow_hiddenMLP_NoLayers, 
                              randperm=randPerm,
                              base=eval(flow_base),
                              base_args=[torch.ones(flow_nfeatures)*flow_base_first_arg, torch.ones(flow_nfeatures)*flow_base_second_arg],
                              dtype=dtype,
                              univariate_kwargs={"bound": flow_bound }, # Keeping the flow in the [-B,B] box.
                              passes= 2 if not flow_autoregressive else flow_nfeatures)

        self.classifier_exist = Classify_ExistJet(no_recoVars, no_partonVars,
                                                     no_recoObjects, no_transformers,
                                                     transformer_input_features, transformer_nhead,
                                                     transformer_num_encoder_layers,
                                                     transformer_num_decoder_layers,
                                                     transformer_dim_feedforward,
                                                     transformer_activation,
                                                     DNN_nodes, DNN_layers,
                                                     no_max_objects, device, dtype, eps)

        if load_classifier:
            state_dict = torch.load(pretrained_classifier, map_location="cpu")
            self.classifier_exist.load_state_dict(state_dict['model_state_dict'])

        # mask to keep the decoder autoregressive (add 1 for the null token)
        self.tgt_mask = self.classifier_exist.transformer_model.generate_square_subsequent_mask(no_recoObjects+1, device=device)
        
        if dtype == torch.float32:
            self.tgt_mask = self.tgt_mask.float()
            self.flow_pt = self.flow_pt.float()
            self.flow_phi = self.flow_phi.float()
            self.flow_eta = self.flow_eta.float()
        elif dtype == torch.float64:
            self.tgt_mask == self.tgt_mask.double()
            self.flow_pt = self.flow_pt.double()
            self.flow_phi = self.flow_phi.double()
            self.flow_eta = self.flow_eta.double()

    def disable_conditioner_regression_training(self):
        ''' Disable the conditioner regression training, but keep the
        latent space training'''
        self.cond_transformer.disable_regression_training()

    def enable_regression_training(self):
        self.cond_transformer.enable_regression_training()
        
    def forward(self, scaling_reco_lab, scaling_partons_lab, scaling_RegressedBoost_lab,
                mask_reco, mask_boost):

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
        # transformer from classifier_exist used for classification and c_i
        output_decoder = self.classifier_exist.transformer_model(scaledLogParton_afterLin, scaledLogReco_afterLin, tgt_mask=self.classifier_exist.tgt_mask)

        # check if we keep the same behaviour in the classifier (to have the same output_decoder)
        if self.encode_position:
            hot_encoded = self.classifier_exist.hot_encoded.expand(output_decoder.shape[0], -1, -1)
            output_decoder = torch.cat((output_decoder, hot_encoded), dim=2)

        no_objects_per_event = torch.sum(mask_reco[:,:self.no_max_objects], dim=1) # compute the number of objects per event

        # exist classifier 
        prob_each_jet = self.classifier_exist.model(output_decoder).squeeze(dim=2)
        prob_each_jet = prob_each_jet[:,:self.no_max_objects]
        prob_each_event = torch.sum(prob_each_jet*mask_reco[:,:self.no_max_objects], dim=1) # take avg of masked objects
        prob_each_event = torch.div(prob_each_event, no_objects_per_event) # divide the total loss in the event at the no_objects_per_event
        prob_avg = prob_each_event.mean()

        prob_each_jet = prob_each_jet.unsqueeze(dim=2) # unsqueeze
        conditioning_pt = output_decoder[:,:self.no_max_objects]
        conditioning_pt = torch.cat((conditioning_pt, prob_each_jet), dim=2)
        # very important: pt on the 2nd position
        scaled_reco_lab_pt = scaling_reco_lab[:,:self.no_max_objects,1].unsqueeze(dim=2)
        flow_prob_pt = self.flow_pt(conditioning_pt).log_prob(scaled_reco_lab_pt)
        flow_prob_pt_batch = torch.sum(flow_prob_pt*mask_reco[:,:self.no_max_objects], dim=1) # take avg of masked objects
        flow_prob_pt_batch = torch.div(flow_prob_pt_batch, no_objects_per_event) # divide the total loss in the event at the no_objects_per_event
        avg_flow_prob_pt = flow_prob_pt_batch.mean()

        # very important: eta on the 3rd position
        conditioning_eta = torch.cat((output_decoder[:,:self.no_max_objects], scaled_reco_lab_pt, prob_each_jet), dim=2) # add pt in conditioning
        scaled_reco_lab_eta = scaling_reco_lab[:,:self.no_max_objects,2].unsqueeze(dim=2)
        flow_prob_eta = self.flow_eta(conditioning_eta).log_prob(scaled_reco_lab_eta)
        # eta = 0 for all MET => remove it from the training of flow_eta 
        flow_prob_eta[:,7] = 0 # set it to 0 because MET have always eta=0
        flow_prob_eta_batch = torch.sum(flow_prob_eta*mask_reco[:,:self.no_max_objects], dim=1) # take avg of masked objects
        # divide at 'no_objects_per_event-1' because MET is removed from the eta training
        flow_prob_eta_batch = torch.div(flow_prob_eta_batch, no_objects_per_event-1) # divide the total loss in the event at the no_objects_per_event
        avg_flow_prob_eta = flow_prob_eta_batch.mean()

        # very important: phi on the 4th position
        conditioning_phi = torch.cat((output_decoder[:,:self.no_max_objects], scaled_reco_lab_pt, scaled_reco_lab_eta, prob_each_jet), dim=2)
        scaled_reco_lab_phi = scaling_reco_lab[:,:self.no_max_objects,3].unsqueeze(dim=2)
        flow_prob_phi = self.flow_phi(conditioning_phi).log_prob(scaled_reco_lab_phi)
        flow_prob_phi_batch = torch.sum(flow_prob_phi*mask_reco[:,:self.no_max_objects], dim=1) # take avg of masked objects
        flow_prob_phi_batch = torch.div(flow_prob_phi_batch, no_objects_per_event) # divide the total loss in the event at the no_objects_per_event
        avg_flow_prob_phi = flow_prob_phi_batch.mean()
 
        prob_each_jet = prob_each_jet.squeeze(dim=2) # squeeze back
                                
        return avg_flow_prob_pt, flow_prob_pt_batch, flow_prob_pt, \
                avg_flow_prob_eta, flow_prob_eta_batch, flow_prob_eta, \
                avg_flow_prob_phi, flow_prob_phi_batch, flow_prob_phi, \
                prob_avg, prob_each_event, prob_each_jet