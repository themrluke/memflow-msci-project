import torch.nn as nn
import torch
import numpy as np
import utils
import zuko
#from zuko.flows import TransformModule, SimpleAffineTransform
from zuko.distributions import BoxUniform
from zuko.distributions import DiagNormal
from memflow.unfolding_flow.utils import Compute_ParticlesTensor as particle_tools
import memflow.phasespace.utils as ps_utils

from memflow.pretrain_exist.pretrain_exist_binary_newInputDNN import Classify_ExistJet_newInputDNN
from memflow.transfer_flow.periodicNSF_gaussian import NCSF_gaussian

class TransferFlow_Paper_AllPartons_Nobtag_autoreg_latentSpace_gaussian_classifier(nn.Module):
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

                 flow_lepton_hiddenMLP_NoLayers=4,
                 flow_lepton_hiddenMLP_LayerDim=128,
                 flow_lepton_bins=16,

                 DNN_nodes=64, DNN_layers=8,
                 pretrained_classifier='',
                 load_classifier=False,
                 freeze_classifier=True,
                 encode_position=True,
                 
                 device=torch.device('cpu'),
                 dtype=torch.float32,
                 eps=1e-4):

        super(TransferFlow_Paper_AllPartons_Nobtag_autoreg_latentSpace_gaussian_classifier, self).__init__()
        
        self.device = device
        self.dtype = dtype
        self.eps = eps # used for small values like the mass of the gluon for numerical reasons
        self.encode_position = encode_position
        
        self.linearDNN_reco = nn.Linear(in_features=no_recoVars, out_features=transformer_input_features, dtype=dtype)
        self.linearDNN_parton = nn.Linear(in_features=no_partonVars, out_features=transformer_input_features, dtype=dtype)
        self.linearDNN_boost = nn.Linear(in_features=4, out_features=transformer_input_features, dtype=dtype)
        self.gelu = nn.GELU()
        self.no_max_objects = no_max_objects
        self.transformer_input_features = transformer_input_features

        # by default between -5 and 5 (bound = 5)
        self.flow_kinematics_lepton_pt = zuko.flows.NSF(features=1,
                              context=transformer_input_features + 3, # additional position of 'jet'/phi/eta
                              transforms=flow_ntransforms, 
                              bins=flow_lepton_bins, 
                              hidden_features=[flow_lepton_hiddenMLP_LayerDim]*flow_lepton_hiddenMLP_NoLayers, 
                              randperm=randPerm,
                              passes= 2 if not flow_autoregressive else 1)

        # flow eta
        self.flow_kinematics_lepton_eta = zuko.flows.NSF(features=1, # 2 because MET doesn't have eta
                              context=transformer_input_features + 3, # additional position of 'jet'/phi
                              transforms=flow_ntransforms, 
                              bins=flow_lepton_bins, 
                              hidden_features=[flow_lepton_hiddenMLP_LayerDim]*flow_lepton_hiddenMLP_NoLayers, 
                              randperm=randPerm,
                              passes= 2 if not flow_autoregressive else 1)

        # flow for phi (periodic)
        self.flow_kinematics_lepton_phi = NCSF_gaussian(features=1,
                              context=transformer_input_features + 3, # additional position of 'jet'
                              transforms=flow_ntransforms, 
                              bins=flow_lepton_bins, 
                              hidden_features=[flow_lepton_hiddenMLP_LayerDim]*flow_lepton_hiddenMLP_NoLayers, 
                              randperm=randPerm,
                              passes= 2 if not flow_autoregressive else 1)

        # by default between -5 and 5 (bound = 5)
        self.flow_kinematics_pt = zuko.flows.NSF(features=1,
                              context=transformer_input_features + 3, # additional position of 'jet'/phi/eta
                              transforms=flow_ntransforms, 
                              bins=flow_bins, 
                              hidden_features=[flow_hiddenMLP_LayerDim]*flow_hiddenMLP_NoLayers, 
                              randperm=randPerm,
                              passes= 2 if not flow_autoregressive else 1)

        # flow eta
        self.flow_kinematics_eta = zuko.flows.NSF(features=1, # 2 because MET doesn't have eta
                              context=transformer_input_features + 3, # additional position of 'jet'/phi
                              transforms=flow_ntransforms, 
                              bins=flow_bins, 
                              hidden_features=[flow_hiddenMLP_LayerDim]*flow_hiddenMLP_NoLayers, 
                              randperm=randPerm,
                              passes= 2 if not flow_autoregressive else 1)

        # flow for phi (periodic)
        self.flow_kinematics_phi = NCSF_gaussian(features=1,
                              context=transformer_input_features + 3, # additional position of 'jet'
                              transforms=flow_ntransforms, 
                              bins=flow_bins, 
                              hidden_features=[flow_hiddenMLP_LayerDim]*flow_hiddenMLP_NoLayers, 
                              randperm=randPerm,
                              passes= 2 if not flow_autoregressive else 1)

        self.classifier_exist = Classify_ExistJet_newInputDNN(no_recoVars, no_partonVars,
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
            self.flow_kinematics_pt = self.flow_kinematics_pt.float()
            self.flow_kinematics_eta = self.flow_kinematics_eta.float()
            self.flow_kinematics_phi = self.flow_kinematics_phi.float()
            self.flow_kinematics_lepton_pt = self.flow_kinematics_lepton_pt.float()
            self.flow_kinematics_lepton_eta = self.flow_kinematics_lepton_eta.float()
            self.flow_kinematics_lepton_phi = self.flow_kinematics_lepton_phi.float()
        elif dtype == torch.float64:
            self.tgt_mask == self.tgt_mask.double()
            self.flow_kinematics_pt = self.flow_kinematics_pt.double()
            self.flow_kinematics_eta = self.flow_kinematics_eta.double()
            self.flow_kinematics_phi = self.flow_kinematics_phi.double()
            self.flow_kinematics_lepton_pt = self.flow_kinematics_lepton_pt.double()
            self.flow_kinematics_lepton_eta = self.flow_kinematics_lepton_eta.double()
            self.flow_kinematics_lepton_phi = self.flow_kinematics_lepton_phi.double()

    def disable_conditioner_regression_training(self):
        ''' Disable the conditioner regression training, but keep the
        latent space training'''
        self.cond_transformer.disable_regression_training()

    def enable_regression_training(self):
        self.cond_transformer.enable_regression_training()
        
    def forward(self, scaling_reco_lab, scaling_partons_lab, scaling_RegressedBoost_lab,
                mask_partons, mask_reco, mask_boost):

        # create null token and its mask
        null_token = torch.ones((scaling_reco_lab.shape[0], 1, 5), device=self.device, dtype=self.dtype) * -1
        null_token[:,0,0] = 0 # exist flag = 0 not -1
        # mask for the null token = True
        null_token_mask = torch.ones((mask_reco.shape[0], 1), device=self.device, dtype=torch.bool)

        # attach null token and update the mask for the scaling_reco_lab
        scaling_reco_lab_withNullToken = torch.cat((null_token, scaling_reco_lab), dim=1)
        mask_reco_withNullToken = torch.cat((null_token_mask, mask_reco), dim=1)
        
        scaledLogReco_afterLin = self.gelu(self.linearDNN_reco(scaling_reco_lab_withNullToken) * mask_reco_withNullToken[..., None])
        scaledLogParton_afterLin = self.gelu(self.linearDNN_parton(scaling_partons_lab) * mask_partons[...,None])    

        # output decoder shape: [events, null_token+jets, 64]
        # first elem of output decoder -> null token
        # transformer from classifier_exist used for classification and c_i
        output_decoder = self.classifier_exist.transformer_model(scaledLogParton_afterLin, scaledLogReco_afterLin, tgt_mask=self.classifier_exist.tgt_mask)
        # problem: I think here it's fine -> I condition also on the non existence elems
        # but this should be ok

        # check if we keep the same behaviour in the classifier (to have the same output_decoder)
        if self.encode_position:
            hot_encoded = self.classifier_exist.hot_encoded.expand(output_decoder.shape[0], -1, -1)
            
            # first 9 partons (decay products) + rest of null tokens
            no_null_tokens = scaling_reco_lab.shape[1] - 9 + 1
            null_token_partons = torch.ones((scaling_partons_lab.shape[0], no_null_tokens, self.transformer_input_features), device=self.device, dtype=self.dtype) * -1

            # attach null tokens to scaledLogParton_afterLin
            inputDNN_scaledLogParton = torch.cat((scaledLogParton_afterLin[:,:9], null_token_partons), dim=1)

            # new input for the DNN exist
            inputDNN_exist = torch.cat((output_decoder, inputDNN_scaledLogParton, hot_encoded), dim=2)
            output_decoder = torch.cat((output_decoder, hot_encoded), dim=2)

        # mask the logprob of jets with exist==0
        # start from 2 (remove lepton/MET)
        maskExist = scaling_reco_lab[:,:self.no_max_objects,0] == 1

        scaled_reco_lab_pt = scaling_reco_lab[:,:self.no_max_objects,1].unsqueeze(dim=2)
        scaled_reco_lab_eta = scaling_reco_lab[:,:self.no_max_objects,2].unsqueeze(dim=2)
        scaled_reco_lab_phi = scaling_reco_lab[:,:self.no_max_objects,3].unsqueeze(dim=2)

         # LEPTONS
         # FLOW PT
        conditioning_pt_lepton = torch.cat((output_decoder[:,:2], scaled_reco_lab_phi[:,:2], scaled_reco_lab_eta[:,:2]), dim=2) # add phi/eta in conditioning
        # very important: pt on the 2nd position
        flow_prob_lepton_pt = self.flow_kinematics_lepton_pt(conditioning_pt_lepton).log_prob(scaled_reco_lab_pt[:,:2]) #+ torch.exp(scaled_reco_lab_pt[:,:2]).squeeze(dim=2)

        # FLOW ETA
        pt_lepton_latentSpace = self.flow_kinematics_lepton_pt(conditioning_pt_lepton).transform(scaled_reco_lab_pt[:,:2])
        conditioning_eta_lepton = torch.cat((output_decoder[:,:2], scaled_reco_lab_phi[:,:2], pt_lepton_latentSpace), dim=2)
        flow_prob_lepton_eta = self.flow_kinematics_lepton_eta(conditioning_eta_lepton).log_prob(scaled_reco_lab_eta[:,:2])
        flow_prob_lepton_eta[:,1] = 0 # set log_prob(eta) for MET = 0 -> doesn't affect the computations

        # FLOW PHI
        eta_lepton_latentSpace = self.flow_kinematics_lepton_eta(conditioning_eta_lepton).transform(scaled_reco_lab_eta[:,:2])
        eta_lepton_latentSpace[:,1] = 0
        conditioning_phi_lepton = torch.cat((output_decoder[:,:2], pt_lepton_latentSpace, eta_lepton_latentSpace), dim=2)
        flow_prob_lepton_phi = self.flow_kinematics_lepton_phi(conditioning_phi_lepton).log_prob(scaled_reco_lab_phi[:,:2])


         # JETS
         # FLOW PT
        conditioning_pt = torch.cat((output_decoder[:,2:self.no_max_objects], scaled_reco_lab_phi[:,2:], scaled_reco_lab_eta[:,2:]), dim=2) # add phi/eta in conditioning
        # very important: pt on the 2nd position
        flow_prob_pt = self.flow_kinematics_pt(conditioning_pt).log_prob(scaled_reco_lab_pt[:,2:]) #+ torch.exp(scaled_reco_lab_pt[:,2:]).squeeze(dim=2)
        flow_prob_pt = torch.cat((flow_prob_lepton_pt, flow_prob_pt), dim=1)
        flow_prob_pt_batch = torch.sum(flow_prob_pt*maskExist, dim=1) # take avg of masked objects
        avg_flow_prob_pt = flow_prob_pt_batch.mean()

        # FLOW ETA
        pt_latentSpace = self.flow_kinematics_pt(conditioning_pt).transform(scaled_reco_lab_pt[:,2:])
        conditioning_eta = torch.cat((output_decoder[:,2:self.no_max_objects], scaled_reco_lab_phi[:,2:], pt_latentSpace), dim=2)
        flow_prob_eta = self.flow_kinematics_eta(conditioning_eta).log_prob(scaled_reco_lab_eta[:,2:])
        flow_prob_eta = torch.cat((flow_prob_lepton_eta, flow_prob_eta), dim=1)
        flow_prob_eta_batch = torch.sum(flow_prob_eta*maskExist, dim=1) # take avg of masked objects
        avg_flow_prob_eta = flow_prob_eta_batch.mean()


        # FLOW PHI
        eta_latentSpace = self.flow_kinematics_eta(conditioning_eta).transform(scaled_reco_lab_eta[:,2:])
        conditioning_phi = torch.cat((output_decoder[:,2:self.no_max_objects], pt_latentSpace, eta_latentSpace), dim=2)
        flow_prob_phi = self.flow_kinematics_phi(conditioning_phi).log_prob(scaled_reco_lab_phi[:,2:])
        flow_prob_phi = torch.cat((flow_prob_lepton_phi, flow_prob_phi), dim=1)
        flow_prob_phi_batch = torch.sum(flow_prob_phi*maskExist, dim=1) # take avg of masked objects
        #flow_prob_phi_batch = torch.div(flow_prob_phi_batch, no_objects_per_event) # divide the total loss in the event at the no_objects_per_event
        avg_flow_prob_phi = flow_prob_phi_batch.mean()
        

        # compute flow prob per event
        flow_prob_batch = flow_prob_phi_batch + flow_prob_eta_batch + flow_prob_pt_batch # take avg of masked objects
        avg_flow_prob = flow_prob_batch.mean()

        # exist classifier
        # here we used the old mask: mask_reco
        prob_each_jet = self.classifier_exist.model(inputDNN_exist).squeeze(dim=2)
        prob_each_jet = prob_each_jet[:,:self.no_max_objects]
        #prob_each_event = torch.sum(prob_each_jet*mask_reco[:,:self.no_max_objects], dim=1) # take avg of masked objects
        prob_each_event = torch.sum(prob_each_jet, dim=1) # take avg of masked objects TO BE REMOVED
        prob_avg = prob_each_event.mean() # TO BE REMOVED
                                
        return avg_flow_prob, flow_prob_batch, \
                avg_flow_prob_pt, flow_prob_pt_batch, flow_prob_pt, \
                avg_flow_prob_eta, flow_prob_eta_batch, flow_prob_eta, \
                avg_flow_prob_phi, flow_prob_phi_batch, flow_prob_phi, \
                prob_avg, prob_each_event, prob_each_jet