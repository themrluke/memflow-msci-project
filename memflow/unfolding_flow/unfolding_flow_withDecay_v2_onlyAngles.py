import torch.nn as nn
import torch
import numpy as np
import utils
from memflow.unfolding_network.conditional_transformer_v3_OnlyDecay import ConditioningTransformerLayer_v3

import zuko
from zuko.distributions import BoxUniform
from zuko.distributions import DiagNormal
from memflow.unfolding_flow.utils import Compute_ParticlesTensor as particle_tools
import memflow.phasespace.utils as ps_utils
from memflow.transfer_flow.periodicNSF_gaussian import NCSF_gaussian

from memflow.unfolding_flow.utils import Compute_ParticlesTensor
from memflow.phasespace.phasespace import PhaseSpace

class UnfoldingFlow_withDecay_v2(nn.Module):
    def __init__(self,
                 regression_hidden_features=16,
                 regression_DNN_input=64,
                 regression_dim_feedforward=16,
                 regression_nhead_encoder=4,
                 regression_noLayers_encoder=3,
                 regression_noLayers_decoder=3,
                 regression_DNN_layers=2,
                 regression_DNN_nodes=16,
                 regression_aggregate=False,
                 regression_atanh=True,
                 regression_angles_CM=True,
                 
                 flow_nfeatures=12,
                 flow_ncond=34, 
                 flow_ntransforms=5,
                 flow_hiddenMLP_NoLayers=16,
                 flow_hiddenMLP_LayerDim=128,
                 flow_bins=16,
                 flow_autoregressive=True, 
                 flow_base=BoxUniform,
                 flow_base_first_arg=-1,
                 flow_base_second_arg=1,
                 flow_bound=1.,
                 randPerm=False,
                 
                 flow_context_angles=10,
                 flow_ntransforms_angles=5,
                 flow_nbins_angles=5,
                 flow_hiddenMLP_LayerDim_angles=32,
                 flow_hiddenMLP_NoLayers_angles=4,
                 flow_base_anglesCM=BoxUniform,
                 flow_base_first_arg_anglesCM=-1,
                 flow_base_second_arg_anglesCM=1,
                 randPerm_angles=True,
                 
                 device=torch.device('cpu'),
                 dtype=torch.float32,
                 pretrained_model='',
                 load_conditioning_model=False):

        super(UnfoldingFlow_withDecay_v2, self).__init__()

        self.device = device
        self.dtype = dtype
        
        self.cond_transformer = ConditioningTransformerLayer_v3(no_recoVars=4, # exist + 3-mom
                                                            no_partonVars=3, # 
                                                            hidden_features=regression_hidden_features,
                                                            DNN_input=regression_DNN_input,
                                                            dim_feedforward_transformer=regression_dim_feedforward,
                                                            nhead_encoder=regression_nhead_encoder,
                                                            no_layers_encoder=regression_noLayers_encoder,
                                                            no_layers_decoder=regression_noLayers_decoder,
                                                            transformer_activation=nn.GELU(),
                                                            DNN_layers=regression_DNN_layers,
                                                            DNN_nodes=regression_DNN_nodes,
                                                            aggregate=regression_aggregate,                                                
                                                            arctanh=regression_atanh,
                                                            angles_CM=regression_angles_CM,
                                                            dtype=dtype,
                                                            device=self.device) 

        if load_conditioning_model:
            print('Read weights')
            state_dict = torch.load(pretrained_model, map_location="cpu")
            if 'latent_proj.weight' or 'latent_proj.bias' in state_dict['model_state_dict']:
                state_dict['model_state_dict'].pop('latent_proj.weight', None)
                state_dict['model_state_dict'].pop('latent_proj.bias', None)
            self.cond_transformer.load_state_dict(state_dict['model_state_dict'])   

        # flow for higgs angles: theta and phi
        # even if theta doesn't have any periodicity
        self.flow_higgs_thetaPhi_CM_unscaled = NCSF_gaussian(features=2,
                              context=2 + flow_context_angles,   # condition on regressed phi + sampled eta
                              transforms=flow_ntransforms_angles, 
                              bins=flow_nbins_angles, 
                              hidden_features=[flow_hiddenMLP_LayerDim_angles]*flow_hiddenMLP_NoLayers_angles, 
                              randperm=randPerm_angles,
                              passes= 2 if not flow_autoregressive else 1)
        
        self.flow_thad_b_thetaPhi_CM_unscaled = NCSF_gaussian(features=2,
                              context=2 + flow_context_angles,      # condition on regressed phi + sampled eta
                              transforms=flow_ntransforms_angles, 
                              bins=flow_nbins_angles, 
                              hidden_features=[flow_hiddenMLP_LayerDim_angles]*flow_hiddenMLP_NoLayers_angles, 
                              randperm=randPerm_angles,
                              passes= 2 if not flow_autoregressive else 1)

        self.flow_tlep_b_thetaPhi_CM_unscaled = NCSF_gaussian(features=2,      
                              context=2 + flow_context_angles,         # condition on regressed phi + sampled eta
                              transforms=flow_ntransforms_angles,
                              bins=flow_nbins_angles,
                              hidden_features=[flow_hiddenMLP_LayerDim_angles]*flow_hiddenMLP_NoLayers_angles,
                              randperm=randPerm_angles,
                              passes= 2 if not flow_autoregressive else 1)

        self.flow_thad_W_thetaPhi_CM_unscaled = NCSF_gaussian(features=2,
                              context=2 + flow_context_angles,      # condition on regressed phi + sampled eta
                              transforms=flow_ntransforms_angles, 
                              bins=flow_nbins_angles, 
                              hidden_features=[flow_hiddenMLP_LayerDim_angles]*flow_hiddenMLP_NoLayers_angles, 
                              randperm=randPerm_angles,
                              passes= 2 if not flow_autoregressive else 1)

        self.flow_tlep_W_thetaPhi_CM_unscaled = NCSF_gaussian(features=2,      
                              context=2 + flow_context_angles,         # condition on regressed phi + sampled eta
                              transforms=flow_ntransforms_angles,
                              bins=flow_nbins_angles,
                              hidden_features=[flow_hiddenMLP_LayerDim_angles]*flow_hiddenMLP_NoLayers_angles,
                              randperm=randPerm_angles,
                              passes= 2 if not flow_autoregressive else 1)


        if dtype == torch.float32:
            self.flow_higgs_thetaPhi_CM_unscaled = self.flow_higgs_thetaPhi_CM_unscaled.float()
            self.flow_thad_b_thetaPhi_CM_unscaled = self.flow_thad_b_thetaPhi_CM_unscaled.float()
            self.flow_tlep_b_thetaPhi_CM_unscaled = self.flow_tlep_b_thetaPhi_CM_unscaled.float()
            self.flow_thad_W_thetaPhi_CM_unscaled = self.flow_thad_W_thetaPhi_CM_unscaled.float()
            self.flow_tlep_W_thetaPhi_CM_unscaled = self.flow_tlep_W_thetaPhi_CM_unscaled.float()
        elif dtype == torch.float64:
            self.flow_higgs_thetaPhi_CM_unscaled = self.flow_higgs_thetaPhi_CM_unscaled.double()
            self.flow_thad_b_thetaPhi_CM_unscaled = self.flow_thad_b_thetaPhi_CM_unscaled.double()
            self.flow_tlep_b_thetaPhi_CM_unscaled = self.flow_tlep_b_thetaPhi_CM_unscaled.double()
            self.flow_thad_W_thetaPhi_CM_unscaled = self.flow_thad_W_thetaPhi_CM_unscaled.double()
            self.flow_tlep_W_thetaPhi_CM_unscaled = self.flow_tlep_W_thetaPhi_CM_unscaled.double()
        

    def disable_conditioner_regression_training(self):
        ''' Disable the conditioner regression training, but keep the
        latent space training'''
        self.cond_transformer.disable_regression_training()

    def enable_regression_training(self):
        self.cond_transformer.enable_regression_training()
        
    def forward(self,  logScaled_reco_Spanet, data_boost_reco,
                mask_recoParticles, mask_boost_reco,
                higgs_etaPhi_unscaled_CM_target,
                thad_etaPhi_unscaled_CM_target,
                tlep_etaPhi_unscaled_CM_target,
                log_mean_parton, log_std_parton,
                log_mean_boost_parton, log_std_boost_parton,
                log_mean_parton_Hthad, log_std_parton_Hthad,
                order=[0,1,2,3], disableGradConditioning =False,
                flow_eval="normalizing", Nsamples=0, No_regressed_vars=9,
                sin_cos_embedding=False, sin_cos_reco=None, sin_cos_partons=None,
                attach_position_regression=None, rambo=None):


        if disableGradConditioning:  # do no train cond transformer at all with sampling epoch
            with torch.no_grad():
                cond_X = self.cond_transformer(logScaled_reco_Spanet, data_boost_reco[:,:,[0,3]],
                                               mask_recoParticles, mask_boost_reco,
                                               No_regressed_vars = No_regressed_vars, sin_cos_reco = sin_cos_reco,
                                               sin_cos_partons=sin_cos_partons, sin_cos_embedding=True,
                                               attach_position=attach_position_regression, eps_arctanh=0.)
        else:
            cond_X = self.cond_transformer(logScaled_reco_Spanet, data_boost_reco[:,:,[0,3]],
                                           mask_recoParticles, mask_boost_reco,
                                           No_regressed_vars = No_regressed_vars, sin_cos_reco = sin_cos_reco,
                                           sin_cos_partons=sin_cos_partons, sin_cos_embedding=True,
                                           attach_position=attach_position_regression, eps_arctanh=0.)


        Hthadtlep_lab_ptetaphi_scaled = cond_X[0]
        decayVars_etaPhi_CM_unscaled = cond_X[1]
        boost_regressed_Epz_scaled = cond_X[2]
        free_latent_space = cond_X[3]

        # log_mean_partons have only [E,pz] components
        regressed_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab = Compute_ParticlesTensor.get_decayPartons_fromlab_propagators_angles(Hthadtlep_lab_ptetaphi_scaled,
                                                                  higgs_angles=decayVars_etaPhi_CM_unscaled[:,0],
                                                                  thad_b_angles=decayVars_etaPhi_CM_unscaled[:,1],
                                                                  thad_W_angles=decayVars_etaPhi_CM_unscaled[:,2],
                                                                  tlep_b_angles=decayVars_etaPhi_CM_unscaled[:,3],
                                                                  tlep_W_angles=decayVars_etaPhi_CM_unscaled[:,4],
                                                                  boost=boost_regressed_Epz_scaled,  # here must be scaled pt, scaled eta, phi
                                                                  log_mean_parton_lab=log_mean_parton, log_std_parton_lab=log_std_parton,
                                                                  log_mean_boost=log_mean_boost_parton, log_std_boost=log_std_boost_parton,
                                                                  log_mean_parton_Hthadtlep=log_mean_parton_Hthad, log_std_parton_Hthadtlep=log_std_parton_Hthad,
                                                                  device=self.device,
                                                                  higgs_mass=125.25,
                                                                  thad_mass=172.5,
                                                                  tlep_mass=172.5,
                                                                  W_had_mass=80.4,
                                                                  W_lep_mass=80.4,
                                                                  b_mass=0.0,
                                                                  ptetaphi=True, eps=1e-4,
                                                                  pt_cut=None, unscale_phi=False, debug=False,
                                                                  final_scaling=True)


        #mask_higgs_pt = regressed_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab[:,1,1] < regressed_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab[:,2,1]
        #mask_thad_pt = regressed_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab[:,5,1] < regressed_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab[:,6,1]

        #regressed_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab[:,[1,2]] = torch.where(mask_higgs_pt[:,None,None],
                                                                                        #regressed_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab[:,[2,1]],
                                                                                        #regressed_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab[:,[1,2]])

        #regressed_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab[:,[5,6]] = torch.where(mask_thad_pt[:,None,None],
                                                                                        #regressed_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab[:,[6,5]],
                                                                                        #regressed_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab[:,[5,6]])

        #mask_higgs_pt = regressed_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab[:,1,1] < regressed_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab[:,2,1]
        #mask_thad_pt = regressed_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab[:,5,1] < regressed_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab[:,6,1]
        #if (mask_higgs_pt.any() or mask_thad_pt.any()):
        #    raise Exception(f"wrong higgs pt: {torch.count_nonzero(mask_higgs_pt)} ------ and wrong thad pt: {torch.count_nonzero(mask_thad_pt)}")
        
        #flow_higgs_context_thetaPhi_unscaled_CM = decayVars_etaPhi_CM_unscaled[:,0:1].clone() # Higgs b1: eta and phi
        #flow_thad_b_context_thetaPhi_unscaled_CM = decayVars_etaPhi_CM_unscaled[:,1:2].clone() # thad b: eta and phi
        #flow_thad_W_context_thetaPhi_unscaled_CM = decayVars_etaPhi_CM_unscaled[:,2:3].clone() # thad q1: eta and phi
        #flow_tlep_b_context_thetaPhi_unscaled_CM = decayVars_etaPhi_CM_unscaled[:,3:4].clone() # tlep b: eta and phi
        #flow_tlep_W_context_thetaPhi_unscaled_CM = decayVars_etaPhi_CM_unscaled[:,4:5].clone() # tlep el: eta and phi

        flow_higgs_context_thetaPhi_unscaled_CM = decayVars_etaPhi_CM_unscaled[:,0:1].clone() # Higgs b1: eta and phi
        flow_thad_b_context_thetaPhi_unscaled_CM = decayVars_etaPhi_CM_unscaled[:,1:2].clone() # thad b: eta and phi
        flow_thad_W_context_thetaPhi_unscaled_CM = decayVars_etaPhi_CM_unscaled[:,2:3].clone() # thad q1: eta and phi
        flow_tlep_b_context_thetaPhi_unscaled_CM = decayVars_etaPhi_CM_unscaled[:,3:4].clone() # tlep b: eta and phi
        flow_tlep_W_context_thetaPhi_unscaled_CM = decayVars_etaPhi_CM_unscaled[:,4:5].clone() # tlep el: eta and phi

        #for i in range(2):
        #    print(i)
        #    print(f'min: {torch.min(flow_higgs_context_thetaPhi_unscaled_CM[...,i])} and max: {torch.max(flow_higgs_context_thetaPhi_unscaled_CM[...,i])}')
        #    print(f'min: {torch.min(flow_thad_b_context_thetaPhi_unscaled_CM[...,i])} and max: {torch.max(flow_thad_b_context_thetaPhi_unscaled_CM[...,i])}')
        #    print(f'min: {torch.min(flow_thad_W_context_thetaPhi_unscaled_CM[...,i])} and max: {torch.max(flow_thad_W_context_thetaPhi_unscaled_CM[...,i])}')
        #    print(f'min: {torch.min(flow_tlep_b_context_thetaPhi_unscaled_CM[...,i])} and max: {torch.max(flow_tlep_b_context_thetaPhi_unscaled_CM[...,i])}')
        #    print(f'min: {torch.min(flow_tlep_W_context_thetaPhi_unscaled_CM[...,i])} and max: {torch.max(flow_tlep_W_context_thetaPhi_unscaled_CM[...,i])}')
        #    print()

        # change from eta to theta to have an angle
        flow_higgs_context_thetaPhi_unscaled_CM[...,0] = 2*torch.atan(torch.exp(-1*flow_higgs_context_thetaPhi_unscaled_CM[...,0]))
        flow_thad_b_context_thetaPhi_unscaled_CM[...,0] = 2*torch.atan(torch.exp(-1*flow_thad_b_context_thetaPhi_unscaled_CM[...,0]))
        flow_thad_W_context_thetaPhi_unscaled_CM[...,0] = 2*torch.atan(torch.exp(-1*flow_thad_W_context_thetaPhi_unscaled_CM[...,0]))
        flow_tlep_b_context_thetaPhi_unscaled_CM[...,0] = 2*torch.atan(torch.exp(-1*flow_tlep_b_context_thetaPhi_unscaled_CM[...,0]))
        flow_tlep_W_context_thetaPhi_unscaled_CM[...,0] = 2*torch.atan(torch.exp(-1*flow_tlep_W_context_thetaPhi_unscaled_CM[...,0]))

        #print(f'theta: min: {torch.min(flow_higgs_context_thetaPhi_unscaled_CM[...,0])} and max: {torch.max(flow_higgs_context_thetaPhi_unscaled_CM[...,0])}')
        #print(f'theta: min: {torch.min(flow_thad_b_context_thetaPhi_unscaled_CM[...,0])} and max: {torch.max(flow_thad_b_context_thetaPhi_unscaled_CM[...,0])}')
        #print(f'theta: min: {torch.min(flow_thad_W_context_thetaPhi_unscaled_CM[...,0])} and max: {torch.max(flow_thad_W_context_thetaPhi_unscaled_CM[...,0])}')
        #print(f'theta: min: {torch.min(flow_tlep_b_context_thetaPhi_unscaled_CM[...,0])} and max: {torch.max(flow_tlep_b_context_thetaPhi_unscaled_CM[...,0])}')
        #print(f'theta: min: {torch.min(flow_tlep_W_context_thetaPhi_unscaled_CM[...,0])} and max: {torch.max(flow_tlep_W_context_thetaPhi_unscaled_CM[...,0])}')
        #print()

        # context vector for the flows for the angles
        condition_higgs_thetaPhi_unscaled_CM = torch.cat((flow_higgs_context_thetaPhi_unscaled_CM, free_latent_space), dim=2)
        condition_thad_b_thetaPhi_unscaled_CM = torch.cat((flow_thad_b_context_thetaPhi_unscaled_CM, free_latent_space), dim=2)
        condition_thad_W_thetaPhi_unscaled_CM = torch.cat((flow_thad_W_context_thetaPhi_unscaled_CM, free_latent_space), dim=2)
        condition_tlep_b_thetaPhi_unscaled_CM = torch.cat((flow_tlep_b_context_thetaPhi_unscaled_CM, free_latent_space), dim=2)
        condition_tlep_W_thetaPhi_unscaled_CM = torch.cat((flow_tlep_W_context_thetaPhi_unscaled_CM, free_latent_space), dim=2)
        
        # And now we can use the flow model
        if flow_eval == "normalizing":

            # move to theta
            higgs_thetaPhi_unscaled_CM_target = higgs_etaPhi_unscaled_CM_target.clone()
            thad_thetaPhi_unscaled_CM_target = thad_etaPhi_unscaled_CM_target.clone()
            tlep_thetaPhi_unscaled_CM_target = tlep_etaPhi_unscaled_CM_target.clone()
            
            #for i in range(2):
            #    print(i)
            #    print(f'min: {torch.min(higgs_thetaPhi_unscaled_CM_target[...,i])} and max: {torch.max(higgs_thetaPhi_unscaled_CM_target[...,i])}')
            #    print(f'min: {torch.min(thad_thetaPhi_unscaled_CM_target[...,0,i])} and max: {torch.max(thad_thetaPhi_unscaled_CM_target[...,0,i])}')
            #    print(f'min: {torch.min(thad_thetaPhi_unscaled_CM_target[...,1,i])} and max: {torch.max(thad_thetaPhi_unscaled_CM_target[...,1,i])}')
            #    print(f'min: {torch.min(tlep_thetaPhi_unscaled_CM_target[...,0,i])} and max: {torch.max(tlep_thetaPhi_unscaled_CM_target[...,0,i])}')
            #    print(f'min: {torch.min(tlep_thetaPhi_unscaled_CM_target[...,1,i])} and max: {torch.max(tlep_thetaPhi_unscaled_CM_target[...,1,i])}')
            #    print()
    
            # change from unscaled eta to theta to have an angle
            higgs_thetaPhi_unscaled_CM_target[...,0] = 2*torch.atan(torch.exp(-1*higgs_thetaPhi_unscaled_CM_target[...,0]))
            thad_thetaPhi_unscaled_CM_target[...,0] = 2*torch.atan(torch.exp(-1*thad_thetaPhi_unscaled_CM_target[...,0]))
            tlep_thetaPhi_unscaled_CM_target[...,0] = 2*torch.atan(torch.exp(-1*tlep_thetaPhi_unscaled_CM_target[...,0]))
    
            #print(f'theta: min: {torch.min(higgs_thetaPhi_unscaled_CM_target[...,0])} and max: {torch.max(higgs_thetaPhi_unscaled_CM_target[...,0])}')
            #print(f'theta: min: {torch.min(thad_thetaPhi_unscaled_CM_target[...,0,0])} and max: {torch.max(thad_thetaPhi_unscaled_CM_target[...,0,0])}')
            #print(f'theta: min: {torch.min(thad_thetaPhi_unscaled_CM_target[...,1,0])} and max: {torch.max(thad_thetaPhi_unscaled_CM_target[...,1,0])}')
            #print(f'theta: min: {torch.min(tlep_thetaPhi_unscaled_CM_target[...,0,0])} and max: {torch.max(tlep_thetaPhi_unscaled_CM_target[...,0,0])}')
            #print(f'theta: min: {torch.min(tlep_thetaPhi_unscaled_CM_target[...,1,0])} and max: {torch.max(tlep_thetaPhi_unscaled_CM_target[...,1,0])}')
            #print()
            
            flow_prob_higgs = self.flow_higgs_thetaPhi_CM_unscaled(condition_higgs_thetaPhi_unscaled_CM).log_prob(higgs_thetaPhi_unscaled_CM_target)
            flow_prob_thad_b = self.flow_thad_b_thetaPhi_CM_unscaled(condition_thad_b_thetaPhi_unscaled_CM).log_prob(thad_thetaPhi_unscaled_CM_target[:,0:1])
            flow_prob_thad_W = self.flow_thad_W_thetaPhi_CM_unscaled(condition_thad_W_thetaPhi_unscaled_CM).log_prob(thad_thetaPhi_unscaled_CM_target[:,1:2])
            flow_prob_tlep_b = self.flow_tlep_b_thetaPhi_CM_unscaled(condition_tlep_b_thetaPhi_unscaled_CM).log_prob(tlep_thetaPhi_unscaled_CM_target[:,0:1])
            flow_prob_tlep_W = self.flow_tlep_W_thetaPhi_CM_unscaled(condition_tlep_W_thetaPhi_unscaled_CM).log_prob(tlep_thetaPhi_unscaled_CM_target[:,1:2])

            return regressed_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab, boost_regressed_Epz_scaled, \
                    flow_prob_higgs, flow_prob_thad_b, flow_prob_thad_W, flow_prob_tlep_b, flow_prob_tlep_W
        
        elif flow_eval == "sampling":

            higgs_thetaPhi_unscaled_CM_sampled = self.flow_higgs_thetaPhi_CM_unscaled(condition_higgs_thetaPhi_unscaled_CM).rsample((Nsamples,))
            thad_b_thetaPhi_unscaled_CM_sampled = self.flow_thad_b_thetaPhi_CM_unscaled(condition_thad_b_thetaPhi_unscaled_CM).rsample((Nsamples,))
            thad_W_thetaPhi_unscaled_CM_sampled = self.flow_thad_W_thetaPhi_CM_unscaled(condition_thad_W_thetaPhi_unscaled_CM).rsample((Nsamples,))
            tlep_b_thetaPhi_unscaled_CM_sampled = self.flow_tlep_b_thetaPhi_CM_unscaled(condition_tlep_b_thetaPhi_unscaled_CM).rsample((Nsamples,))
            tlep_W_thetaPhi_unscaled_CM_sampled = self.flow_tlep_W_thetaPhi_CM_unscaled(condition_tlep_W_thetaPhi_unscaled_CM).rsample((Nsamples,))

            higgs_thetaPhi_unscaled_CM_sampled = torch.flatten(higgs_thetaPhi_unscaled_CM_sampled, start_dim=0, end_dim=1)
            thad_b_thetaPhi_unscaled_CM_sampled = torch.flatten(thad_b_thetaPhi_unscaled_CM_sampled, start_dim=0, end_dim=1)
            thad_W_thetaPhi_unscaled_CM_sampled = torch.flatten(thad_W_thetaPhi_unscaled_CM_sampled, start_dim=0, end_dim=1)
            tlep_b_thetaPhi_unscaled_CM_sampled = torch.flatten(tlep_b_thetaPhi_unscaled_CM_sampled, start_dim=0, end_dim=1)
            tlep_W_thetaPhi_unscaled_CM_sampled = torch.flatten(tlep_W_thetaPhi_unscaled_CM_sampled, start_dim=0, end_dim=1)

            higgs_etaPhi_unscaled_CM_sampled = higgs_thetaPhi_unscaled_CM_sampled
            thad_b_etaPhi_unscaled_CM_sampled = thad_b_thetaPhi_unscaled_CM_sampled
            thad_W_etaPhi_unscaled_CM_sampled = thad_W_thetaPhi_unscaled_CM_sampled
            tlep_b_etaPhi_unscaled_CM_sampled = tlep_b_thetaPhi_unscaled_CM_sampled
            tlep_W_etaPhi_unscaled_CM_sampled = tlep_W_thetaPhi_unscaled_CM_sampled

            # move from theta back to eta (no need to scale eta cuz decay products are not scaled)
            higgs_etaPhi_unscaled_CM_sampled[...,0] = -1*torch.log(torch.tan(higgs_thetaPhi_unscaled_CM_sampled[...,0]/2.))
            thad_b_etaPhi_unscaled_CM_sampled[...,0] = -1*torch.log(torch.tan(thad_b_thetaPhi_unscaled_CM_sampled[...,0]/2.))
            thad_W_etaPhi_unscaled_CM_sampled[...,0] = -1*torch.log(torch.tan(thad_W_thetaPhi_unscaled_CM_sampled[...,0]/2.))
            tlep_b_etaPhi_unscaled_CM_sampled[...,0] = -1*torch.log(torch.tan(tlep_b_thetaPhi_unscaled_CM_sampled[...,0]/2.))
            tlep_W_etaPhi_unscaled_CM_sampled[...,0] = -1*torch.log(torch.tan(tlep_W_thetaPhi_unscaled_CM_sampled[...,0]/2.))

            return regressed_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab, boost_regressed_Epz_scaled, \
                    higgs_etaPhi_unscaled_CM_sampled, thad_b_etaPhi_unscaled_CM_sampled, thad_W_etaPhi_unscaled_CM_sampled, \
                    tlep_b_etaPhi_unscaled_CM_sampled, tlep_W_etaPhi_unscaled_CM_sampled

        elif flow_eval == "both":
            # move to theta
            higgs_thetaPhi_unscaled_CM_target = higgs_etaPhi_unscaled_CM_target.clone()
            thad_thetaPhi_unscaled_CM_target = thad_etaPhi_unscaled_CM_target.clone()
            tlep_thetaPhi_unscaled_CM_target = tlep_etaPhi_unscaled_CM_target.clone()
            
            # change from unscaled eta to theta to have an angle
            higgs_thetaPhi_unscaled_CM_target[...,0] = 2*torch.atan(torch.exp(-1*higgs_thetaPhi_unscaled_CM_target[...,0]))
            thad_thetaPhi_unscaled_CM_target[...,0] = 2*torch.atan(torch.exp(-1*thad_thetaPhi_unscaled_CM_target[...,0]))
            tlep_thetaPhi_unscaled_CM_target[...,0] = 2*torch.atan(torch.exp(-1*tlep_thetaPhi_unscaled_CM_target[...,0]))

            flow_prob_higgs = self.flow_higgs_thetaPhi_CM_unscaled(condition_higgs_thetaPhi_unscaled_CM).log_prob(higgs_thetaPhi_unscaled_CM_target)
            flow_prob_thad_b = self.flow_thad_b_thetaPhi_CM_unscaled(condition_thad_b_thetaPhi_unscaled_CM).log_prob(thad_thetaPhi_unscaled_CM_target[:,0:1])
            flow_prob_thad_W = self.flow_thad_W_thetaPhi_CM_unscaled(condition_thad_W_thetaPhi_unscaled_CM).log_prob(thad_thetaPhi_unscaled_CM_target[:,1:2])
            flow_prob_tlep_b = self.flow_tlep_b_thetaPhi_CM_unscaled(condition_tlep_b_thetaPhi_unscaled_CM).log_prob(tlep_thetaPhi_unscaled_CM_target[:,0:1])
            flow_prob_tlep_W = self.flow_tlep_W_thetaPhi_CM_unscaled(condition_tlep_W_thetaPhi_unscaled_CM).log_prob(tlep_thetaPhi_unscaled_CM_target[:,1:2])

            # sampling
            higgs_thetaPhi_unscaled_CM_sampled = self.flow_higgs_thetaPhi_CM_unscaled(condition_higgs_thetaPhi_unscaled_CM).rsample((Nsamples,))
            thad_b_thetaPhi_unscaled_CM_sampled = self.flow_thad_b_thetaPhi_CM_unscaled(condition_thad_b_thetaPhi_unscaled_CM).rsample((Nsamples,))
            thad_W_thetaPhi_unscaled_CM_sampled = self.flow_thad_W_thetaPhi_CM_unscaled(condition_thad_W_thetaPhi_unscaled_CM).rsample((Nsamples,))
            tlep_b_thetaPhi_unscaled_CM_sampled = self.flow_tlep_b_thetaPhi_CM_unscaled(condition_tlep_b_thetaPhi_unscaled_CM).rsample((Nsamples,))
            tlep_W_thetaPhi_unscaled_CM_sampled = self.flow_tlep_W_thetaPhi_CM_unscaled(condition_tlep_W_thetaPhi_unscaled_CM).rsample((Nsamples,))

            higgs_thetaPhi_unscaled_CM_sampled = torch.flatten(higgs_thetaPhi_unscaled_CM_sampled, start_dim=0, end_dim=1)
            thad_b_thetaPhi_unscaled_CM_sampled = torch.flatten(thad_b_thetaPhi_unscaled_CM_sampled, start_dim=0, end_dim=1)
            thad_W_thetaPhi_unscaled_CM_sampled = torch.flatten(thad_W_thetaPhi_unscaled_CM_sampled, start_dim=0, end_dim=1)
            tlep_b_thetaPhi_unscaled_CM_sampled = torch.flatten(tlep_b_thetaPhi_unscaled_CM_sampled, start_dim=0, end_dim=1)
            tlep_W_thetaPhi_unscaled_CM_sampled = torch.flatten(tlep_W_thetaPhi_unscaled_CM_sampled, start_dim=0, end_dim=1)

            higgs_etaPhi_unscaled_CM_sampled = higgs_thetaPhi_unscaled_CM_sampled
            thad_b_etaPhi_unscaled_CM_sampled = thad_b_thetaPhi_unscaled_CM_sampled
            thad_W_etaPhi_unscaled_CM_sampled = thad_W_thetaPhi_unscaled_CM_sampled
            tlep_b_etaPhi_unscaled_CM_sampled = tlep_b_thetaPhi_unscaled_CM_sampled
            tlep_W_etaPhi_unscaled_CM_sampled = tlep_W_thetaPhi_unscaled_CM_sampled

            #print('check tan(theta/2) < 0')
            #print((torch.tan(higgs_thetaPhi_unscaled_CM_sampled[...,0]/2.) < 0).any())
            #print((torch.tan(thad_b_thetaPhi_unscaled_CM_sampled[...,0]/2.) < 0).any())
            #print((torch.tan(thad_W_thetaPhi_unscaled_CM_sampled[...,0]/2.) < 0).any())
            #print((torch.tan(tlep_b_thetaPhi_unscaled_CM_sampled[...,0]/2.) < 0).any())
            #print((torch.tan(tlep_W_thetaPhi_unscaled_CM_sampled[...,0]/2.) < 0).any())

            higgs_thetaPhi_unscaled_CM_sampled[...,0] = torch.clamp(higgs_thetaPhi_unscaled_CM_sampled[...,0].clone(), min=1e-2, max=np.pi-1e-2)
            thad_b_thetaPhi_unscaled_CM_sampled[...,0] = torch.clamp(thad_b_thetaPhi_unscaled_CM_sampled[...,0].clone(), min=1e-2, max=np.pi-1e-2)
            thad_W_thetaPhi_unscaled_CM_sampled[...,0] = torch.clamp(thad_W_thetaPhi_unscaled_CM_sampled[...,0].clone(), min=1e-2, max=np.pi-1e-2)
            tlep_b_thetaPhi_unscaled_CM_sampled[...,0] = torch.clamp(tlep_b_thetaPhi_unscaled_CM_sampled[...,0].clone(), min=1e-2, max=np.pi-1e-2)
            tlep_W_thetaPhi_unscaled_CM_sampled[...,0] = torch.clamp(tlep_W_thetaPhi_unscaled_CM_sampled[...,0].clone(), min=1e-2, max=np.pi-1e-2)

            # move from theta back to eta (no need to scale eta cuz decay products are not scaled)
            higgs_etaPhi_unscaled_CM_sampled[...,0] = -1*torch.log(torch.tan(higgs_thetaPhi_unscaled_CM_sampled[...,0]/2.))
            thad_b_etaPhi_unscaled_CM_sampled[...,0] = -1*torch.log(torch.tan(thad_b_thetaPhi_unscaled_CM_sampled[...,0]/2.))
            thad_W_etaPhi_unscaled_CM_sampled[...,0] = -1*torch.log(torch.tan(thad_W_thetaPhi_unscaled_CM_sampled[...,0]/2.))
            tlep_b_etaPhi_unscaled_CM_sampled[...,0] = -1*torch.log(torch.tan(tlep_b_thetaPhi_unscaled_CM_sampled[...,0]/2.))
            tlep_W_etaPhi_unscaled_CM_sampled[...,0] = -1*torch.log(torch.tan(tlep_W_thetaPhi_unscaled_CM_sampled[...,0]/2.))

            #print('check nan eta')
            #print(torch.isnan(higgs_etaPhi_unscaled_CM_sampled).any())
            #print(torch.isnan(thad_b_etaPhi_unscaled_CM_sampled).any())
            #print(torch.isnan(thad_W_etaPhi_unscaled_CM_sampled).any())
            #print(torch.isnan(tlep_b_etaPhi_unscaled_CM_sampled).any())
            #print(torch.isnan(tlep_W_etaPhi_unscaled_CM_sampled).any())


            return regressed_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab, boost_regressed_Epz_scaled, \
                    flow_prob_higgs, flow_prob_thad_b, flow_prob_thad_W, flow_prob_tlep_b, flow_prob_tlep_W, \
                    higgs_etaPhi_unscaled_CM_sampled, thad_b_etaPhi_unscaled_CM_sampled, thad_W_etaPhi_unscaled_CM_sampled, \
                    tlep_b_etaPhi_unscaled_CM_sampled, tlep_W_etaPhi_unscaled_CM_sampled


        else:
            raise Exception(f"Invalid flow_eval mode {flow_eval}")
