import torch.nn as nn
import torch
import numpy as np
import utils
from memflow.unfolding_network.conditional_transformer_v3_OnlyDecay import ConditioningTransformerLayer_v3

import zuko
from memflow.unfolding_flow.utils import Compute_ParticlesTensor as particle_tools
import memflow.phasespace.utils as ps_utils
from memflow.transfer_flow.periodicNSF_gaussian import NCSF_gaussian

from .custom_spline_flow_ps import Custom_spline_flow_ps
from .custom_spline_flow_eta import Custom_spline_flow_eta

from memflow.unfolding_flow.utils import Compute_ParticlesTensor
from memflow.phasespace.phasespace import PhaseSpace

class UnfoldingFlow_withDecay_v2_eta(nn.Module):
    def __init__(self,
                 scaling_partons_CM_ps,

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
                 flow_base=None,
                 flow_base_first_arg=-1,
                 flow_base_second_arg=1,
                 flow_bound=1.,
                 randPerm=False,
                 
                 flow_context_angles=10,
                 flow_ntransforms_angles=5,
                 flow_nbins_angles=5,
                 flow_hiddenMLP_LayerDim_angles=32,
                 flow_hiddenMLP_NoLayers_angles=4,
                 flow_base_anglesCM=None,
                 flow_base_first_arg_anglesCM=-1,
                 flow_base_second_arg_anglesCM=1,
                 randPerm_angles=True,

                 DNN_condition=False,
                 DNN_layers=2,
                 DNN_dim=256,
                 
                 device=torch.device('cpu'),
                 dtype=torch.float32,
                 pretrained_model='',
                 load_conditioning_model=False):

        super(UnfoldingFlow_withDecay_v2_eta, self).__init__()

        self.device = device
        self.dtype = dtype
        self.scaling_partons_CM_ps = scaling_partons_CM_ps
        
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

        self.DNN_condition = DNN_condition
        
        if DNN_condition:
            layers = [nn.Linear(regression_DNN_input, DNN_dim, dtype=dtype), nn.GELU()] 
        
            for i in range(DNN_layers - 1):
                layers.append(nn.Linear(DNN_dim, DNN_dim, dtype=dtype))
                layers.append(nn.GELU())

            layers.append(nn.Linear(DNN_dim, flow_context_angles, dtype=dtype))

            self.DNN_context = nn.Sequential(*layers)
            

        self.flow_logit_scaled_ps = Custom_spline_flow_ps(features=flow_nfeatures,
                              context=flow_ncond, 
                              transforms=flow_ntransforms, 
                              bins=flow_bins,
                              hidden_features=[flow_hiddenMLP_LayerDim]*flow_hiddenMLP_NoLayers, 
                              randperm=randPerm,
                              passes= 2 if not flow_autoregressive else flow_nfeatures)

        # eta flows
        self.flow_higgs_CM_unscaled_eta = Custom_spline_flow_eta(features=1,
                              context=1 + flow_context_angles,   # condition on regressed phi + sampled eta
                              transforms=1, 
                              bins=flow_nbins_angles, 
                              hidden_features=[flow_hiddenMLP_LayerDim_angles]*flow_hiddenMLP_NoLayers_angles, 
                              randperm=randPerm_angles,
                              passes= 2 if not flow_autoregressive else 1)
        
        self.flow_thad_b_CM_unscaled_eta = Custom_spline_flow_eta(features=1,
                              context=1 + flow_context_angles,      # condition on regressed phi + sampled eta
                              transforms=1, 
                              bins=flow_nbins_angles, 
                              hidden_features=[flow_hiddenMLP_LayerDim_angles]*flow_hiddenMLP_NoLayers_angles, 
                              randperm=randPerm_angles,
                              passes= 2 if not flow_autoregressive else 1)

        self.flow_tlep_b_CM_unscaled_eta = Custom_spline_flow_eta(features=1,      
                              context=1 + flow_context_angles,         # condition on regressed phi + sampled eta
                              transforms=1,
                              bins=flow_nbins_angles,
                              hidden_features=[flow_hiddenMLP_LayerDim_angles]*flow_hiddenMLP_NoLayers_angles,
                              randperm=randPerm_angles,
                              passes= 2 if not flow_autoregressive else 1)

        self.flow_thad_W_CM_unscaled_eta = Custom_spline_flow_eta(features=1,
                              context=1 + flow_context_angles,      # condition on regressed phi + sampled eta
                              transforms=1, 
                              bins=flow_nbins_angles, 
                              hidden_features=[flow_hiddenMLP_LayerDim_angles]*flow_hiddenMLP_NoLayers_angles, 
                              randperm=randPerm_angles,
                              passes= 2 if not flow_autoregressive else 1)

        self.flow_tlep_W_CM_unscaled_eta = Custom_spline_flow_eta(features=1,      
                              context=1 + flow_context_angles,         # condition on regressed phi + sampled eta
                              transforms=1,
                              bins=flow_nbins_angles,
                              hidden_features=[flow_hiddenMLP_LayerDim_angles]*flow_hiddenMLP_NoLayers_angles,
                              randperm=randPerm_angles,
                              passes= 2 if not flow_autoregressive else 1)

        # phi flows
        self.flow_higgs_CM_unscaled_phi = NCSF_gaussian(features=1,
                              context=1 + flow_context_angles,   # condition on regressed phi + sampled eta
                              transforms=1, 
                              bins=flow_nbins_angles, 
                              hidden_features=[flow_hiddenMLP_LayerDim_angles]*flow_hiddenMLP_NoLayers_angles, 
                              randperm=randPerm_angles,
                              passes= 2 if not flow_autoregressive else 1)
        
        self.flow_thad_b_CM_unscaled_phi = NCSF_gaussian(features=1,
                              context=1 + flow_context_angles,      # condition on regressed phi + sampled eta
                              transforms=1, 
                              bins=flow_nbins_angles, 
                              hidden_features=[flow_hiddenMLP_LayerDim_angles]*flow_hiddenMLP_NoLayers_angles, 
                              randperm=randPerm_angles,
                              passes= 2 if not flow_autoregressive else 1)

        self.flow_tlep_b_CM_unscaled_phi = NCSF_gaussian(features=1,      
                              context=1 + flow_context_angles,         # condition on regressed phi + sampled eta
                              transforms=1,
                              bins=flow_nbins_angles,
                              hidden_features=[flow_hiddenMLP_LayerDim_angles]*flow_hiddenMLP_NoLayers_angles,
                              randperm=randPerm_angles,
                              passes= 2 if not flow_autoregressive else 1)

        self.flow_thad_W_CM_unscaled_phi = NCSF_gaussian(features=1,
                              context=1 + flow_context_angles,      # condition on regressed phi + sampled eta
                              transforms=1, 
                              bins=flow_nbins_angles, 
                              hidden_features=[flow_hiddenMLP_LayerDim_angles]*flow_hiddenMLP_NoLayers_angles, 
                              randperm=randPerm_angles,
                              passes= 2 if not flow_autoregressive else 1)

        self.flow_tlep_W_CM_unscaled_phi = NCSF_gaussian(features=1,      
                              context=1 + flow_context_angles,         # condition on regressed phi + sampled eta
                              transforms=1,
                              bins=flow_nbins_angles,
                              hidden_features=[flow_hiddenMLP_LayerDim_angles]*flow_hiddenMLP_NoLayers_angles,
                              randperm=randPerm_angles,
                              passes= 2 if not flow_autoregressive else 1)


        if dtype == torch.float32:
            self.flow_logit_scaled_ps = self.flow_logit_scaled_ps.float()
            
            self.flow_higgs_CM_unscaled_eta = self.flow_higgs_CM_unscaled_eta.float()
            self.flow_thad_b_CM_unscaled_eta = self.flow_thad_b_CM_unscaled_eta.float()
            self.flow_tlep_b_CM_unscaled_eta = self.flow_tlep_b_CM_unscaled_eta.float()
            self.flow_thad_W_CM_unscaled_eta = self.flow_thad_W_CM_unscaled_eta.float()
            self.flow_tlep_W_CM_unscaled_eta = self.flow_tlep_W_CM_unscaled_eta.float()

            self.flow_higgs_CM_unscaled_phi = self.flow_higgs_CM_unscaled_phi.float()
            self.flow_thad_b_CM_unscaled_phi = self.flow_thad_b_CM_unscaled_phi.float()
            self.flow_tlep_b_CM_unscaled_phi = self.flow_tlep_b_CM_unscaled_phi.float()
            self.flow_thad_W_CM_unscaled_phi = self.flow_thad_W_CM_unscaled_phi.float()
            self.flow_tlep_W_CM_unscaled_phi = self.flow_tlep_W_CM_unscaled_phi.float()
        elif dtype == torch.float64:
            self.flow_logit_scaled_ps = self.flow_logit_scaled_ps.double()
            
            self.flow_higgs_CM_unscaled_eta = self.flow_higgs_CM_unscaled_eta.double()
            self.flow_thad_b_CM_unscaled_eta = self.flow_thad_b_CM_unscaled_eta.double()
            self.flow_tlep_b_CM_unscaled_eta = self.flow_tlep_b_CM_unscaled_eta.double()
            self.flow_thad_W_CM_unscaled_eta = self.flow_thad_W_CM_unscaled_eta.double()
            self.flow_tlep_W_CM_unscaled_eta = self.flow_tlep_W_CM_unscaled_eta.double()

            self.flow_higgs_CM_unscaled_phi = self.flow_higgs_CM_unscaled_phi.double()
            self.flow_thad_b_CM_unscaled_phi = self.flow_thad_b_CM_unscaled_phi.double()
            self.flow_tlep_b_CM_unscaled_phi = self.flow_tlep_b_CM_unscaled_phi.double()
            self.flow_thad_W_CM_unscaled_phi = self.flow_thad_W_CM_unscaled_phi.double()
            self.flow_tlep_W_CM_unscaled_phi = self.flow_tlep_W_CM_unscaled_phi.double()

    def disable_conditioner_regression_training(self):
        ''' Disable the conditioner regression training, but keep the
        latent space training'''
        self.cond_transformer.disable_regression_training()

    def enable_regression_training(self):
        self.cond_transformer.enable_regression_training()
        
    def forward(self,  logScaled_reco_Spanet, data_boost_reco,
                mask_recoParticles, mask_boost_reco,
                logit_ps_scaled_target, 
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
        regressed_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_cartesian_unscaled_lab = Compute_ParticlesTensor.get_decayPartons_fromlab_propagators_angles(Hthadtlep_lab_ptetaphi_scaled,
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
                                                                  ptetaphi=False, eps=1e-4,
                                                                  pt_cut=None, unscale_phi=False, debug=False,
                                                                  final_scaling=False)
    
        

        # Note that we are not constraining x1x2 to be <1 . It is numerically constrained in get_PS
        propagators_regressed_cartesian_unscaled_lab = regressed_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_cartesian_unscaled_lab[:,[0,3,7,11]]

        boost_parton_unscaled_lab = torch.sum(propagators_regressed_cartesian_unscaled_lab, dim=1)

        # Move to CM
        # Rescaling boost and gluon to return the scaled vectors for  regression losses
        # In principle there should be no problem here, but let's just keep it
        mask_wrong_boostE = torch.sqrt(boost_parton_unscaled_lab[..., 0]**2 - boost_parton_unscaled_lab[..., 3]**2) < particle_tools.M_MIN_TOT
        # print("N. events with wrong regressed boost", mask_wrong_boostE.sum())
        boost_parton_unscaled_lab[mask_wrong_boostE][...,0] = torch.sqrt(boost_parton_unscaled_lab[mask_wrong_boostE][...,3]**2 +  particle_tools.M_MIN_TOT**2 + 1e-3)

        boost_parton_lab  = ps_utils.boostVector_t(boost_parton_unscaled_lab).unsqueeze(dim=1)      

        HthadtlepISR_regressed_cartesian_unscaled_cm = ps_utils.boost_tt(propagators_regressed_cartesian_unscaled_lab , -boost_parton_lab) #[ B, 4particles,4]

        # Compute PS
        regressed_ps, detjinv_rambo_regr, mask_problematic =  particle_tools.get_PS(HthadtlepISR_regressed_cartesian_unscaled_cm, boost_parton_unscaled_lab)

        if ((regressed_ps < 0)|(regressed_ps > 1)).any():
            print("WRONG REGRESSED PS")
            breakpoint()

        # Now logit and scale PS
        #print(f'ps: min : {torch.min(regressed_ps)} and max: {torch.max(regressed_ps)}')
        
        regressed_logit_ps = torch.logit(regressed_ps, eps=5e-5) 
        regressed_logit_ps_scaled = (regressed_logit_ps - self.scaling_partons_CM_ps[0] ) / self.scaling_partons_CM_ps[1]

        condition_logit_ps_scaled = torch.cat((regressed_logit_ps_scaled[:,None,:], free_latent_space), dim=2)

        regressed_H_b1_b2_thad_b_q1_q2_tlep_b_el_nu_ISR_Eptetaphi_unscaled_lab = Compute_ParticlesTensor.get_ptetaphi_comp_batch(regressed_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_cartesian_unscaled_lab)

        #for i in range(4):
        #    print(i)
        #    print(f'min: {torch.min(regressed_H_b1_b2_thad_b_q1_q2_tlep_b_el_nu_ISR_Eptetaphi_unscaled_lab[...,i])} and max: {torch.max(regressed_H_b1_b2_thad_b_q1_q2_tlep_b_el_nu_ISR_Eptetaphi_unscaled_lab[...,i])}')
        #    print()
        
        flow_higgs_context_etaPhi_unscaled_CM = decayVars_etaPhi_CM_unscaled[:,0:1] # Higgs b1: eta and phi
        flow_thad_b_context_etaPhi_unscaled_CM = decayVars_etaPhi_CM_unscaled[:,1:2] # thad b: eta and phi
        flow_thad_W_context_etaPhi_unscaled_CM = decayVars_etaPhi_CM_unscaled[:,2:3] # thad q1: eta and phi
        flow_tlep_b_context_etaPhi_unscaled_CM = decayVars_etaPhi_CM_unscaled[:,3:4] # tlep b: eta and phi
        flow_tlep_W_context_etaPhi_unscaled_CM = decayVars_etaPhi_CM_unscaled[:,4:5] # tlep el: eta and phi

        #for i in range(2):
        #    print(i)
        #    print(f'min: {torch.min(flow_higgs_context_thetaPhi_unscaled_CM[...,i])} and max: {torch.max(flow_higgs_context_thetaPhi_unscaled_CM[...,i])}')
        #    print(f'min: {torch.min(flow_thad_b_context_thetaPhi_unscaled_CM[...,i])} and max: {torch.max(flow_thad_b_context_thetaPhi_unscaled_CM[...,i])}')
        #    print(f'min: {torch.min(flow_thad_W_context_thetaPhi_unscaled_CM[...,i])} and max: {torch.max(flow_thad_W_context_thetaPhi_unscaled_CM[...,i])}')
        #    print(f'min: {torch.min(flow_tlep_b_context_thetaPhi_unscaled_CM[...,i])} and max: {torch.max(flow_tlep_b_context_thetaPhi_unscaled_CM[...,i])}')
        #    print(f'min: {torch.min(flow_tlep_W_context_thetaPhi_unscaled_CM[...,i])} and max: {torch.max(flow_tlep_W_context_thetaPhi_unscaled_CM[...,i])}')
        #    print()

        if self.DNN_condition and free_latent_space.shape[2] > 0:
            free_latent_space = self.DNN_context(free_latent_space)

        # context vector for the flows for the angles
        condition_higgs_etaPhi_unscaled_CM = torch.cat((flow_higgs_context_etaPhi_unscaled_CM, free_latent_space), dim=2)
        condition_thad_b_etaPhi_unscaled_CM = torch.cat((flow_thad_b_context_etaPhi_unscaled_CM, free_latent_space), dim=2)
        condition_thad_W_etaPhi_unscaled_CM = torch.cat((flow_thad_W_context_etaPhi_unscaled_CM, free_latent_space), dim=2)
        condition_tlep_b_etaPhi_unscaled_CM = torch.cat((flow_tlep_b_context_etaPhi_unscaled_CM, free_latent_space), dim=2)
        condition_tlep_W_etaPhi_unscaled_CM = torch.cat((flow_tlep_W_context_etaPhi_unscaled_CM, free_latent_space), dim=2)
        
        # And now we can use the flow model
        if flow_eval == "normalizing":

            flow_prob_logit_scaled_ps = self.flow_logit_scaled_ps(condition_logit_ps_scaled[:,0]).log_prob(logit_ps_scaled_target)

            list_allElems_ExceptPhi = [i for i in range(condition_higgs_etaPhi_unscaled_CM.shape[2]) if i != 1]
            list_allElems_ExceptEta = [i for i in range(condition_higgs_etaPhi_unscaled_CM.shape[2]) if i != 0]
            
            # eta flows
            flow_prob_higgs_eta = self.flow_higgs_CM_unscaled_eta(condition_higgs_etaPhi_unscaled_CM[...,list_allElems_ExceptPhi]).log_prob(higgs_etaPhi_unscaled_CM_target[...,0:1])
            flow_prob_thad_b_eta = self.flow_thad_b_CM_unscaled_eta(condition_thad_b_etaPhi_unscaled_CM[...,list_allElems_ExceptPhi]).log_prob(thad_etaPhi_unscaled_CM_target[:,0:1,0:1])
            flow_prob_thad_W_eta = self.flow_thad_W_CM_unscaled_eta(condition_thad_W_etaPhi_unscaled_CM[...,list_allElems_ExceptPhi]).log_prob(thad_etaPhi_unscaled_CM_target[:,1:2,0:1])
            flow_prob_tlep_b_eta = self.flow_tlep_b_CM_unscaled_eta(condition_tlep_b_etaPhi_unscaled_CM[...,list_allElems_ExceptPhi]).log_prob(tlep_etaPhi_unscaled_CM_target[:,0:1,0:1])
            flow_prob_tlep_W_eta = self.flow_tlep_W_CM_unscaled_eta(condition_tlep_W_etaPhi_unscaled_CM[...,list_allElems_ExceptPhi]).log_prob(tlep_etaPhi_unscaled_CM_target[:,1:2,0:1])

            # phi flows
            flow_prob_higgs_phi = self.flow_higgs_CM_unscaled_phi(condition_higgs_etaPhi_unscaled_CM[...,list_allElems_ExceptEta]).log_prob(higgs_etaPhi_unscaled_CM_target[...,1:2])
            flow_prob_thad_b_phi = self.flow_thad_b_CM_unscaled_phi(condition_thad_b_etaPhi_unscaled_CM[...,list_allElems_ExceptEta]).log_prob(thad_etaPhi_unscaled_CM_target[:,0:1,1:2])
            flow_prob_thad_W_phi = self.flow_thad_W_CM_unscaled_phi(condition_thad_W_etaPhi_unscaled_CM[...,list_allElems_ExceptEta]).log_prob(thad_etaPhi_unscaled_CM_target[:,1:2,1:2])
            flow_prob_tlep_b_phi = self.flow_tlep_b_CM_unscaled_phi(condition_tlep_b_etaPhi_unscaled_CM[...,list_allElems_ExceptEta]).log_prob(tlep_etaPhi_unscaled_CM_target[:,0:1,1:2])
            flow_prob_tlep_W_phi = self.flow_tlep_W_CM_unscaled_phi(condition_tlep_W_etaPhi_unscaled_CM[...,list_allElems_ExceptEta]).log_prob(tlep_etaPhi_unscaled_CM_target[:,1:2,1:2])

            flow_prob_higgs = flow_prob_higgs_eta + flow_prob_higgs_phi
            flow_prob_thad_b = flow_prob_thad_b_eta + flow_prob_thad_b_phi
            flow_prob_thad_W = flow_prob_thad_W_eta + flow_prob_thad_W_phi
            flow_prob_tlep_b = flow_prob_tlep_b_eta + flow_prob_tlep_b_phi
            flow_prob_tlep_W = flow_prob_tlep_W_eta + flow_prob_tlep_W_phi

            return regressed_H_b1_b2_thad_b_q1_q2_tlep_b_el_nu_ISR_Eptetaphi_unscaled_lab, boost_regressed_Epz_scaled, \
                    flow_prob_logit_scaled_ps, flow_prob_higgs, flow_prob_thad_b, flow_prob_thad_W, flow_prob_tlep_b, \
                    flow_prob_tlep_W
        
        elif flow_eval == "sampling":
            
            logit_scaled_ps_samples = self.flow_logit_scaled_ps(condition_logit_ps_scaled).rsample((Nsamples,))

            list_allElems_ExceptPhi = [i for i in range(condition_higgs_etaPhi_unscaled_CM.shape[2]) if i != 1]
            list_allElems_ExceptEta = [i for i in range(condition_higgs_etaPhi_unscaled_CM.shape[2]) if i != 0]

            higgs_eta_unscaled_CM_sampled = self.flow_higgs_CM_unscaled_eta(condition_higgs_etaPhi_unscaled_CM[...,list_allElems_ExceptPhi]).rsample((Nsamples,))
            thad_b_eta_unscaled_CM_sampled = self.flow_thad_b_CM_unscaled_eta(condition_thad_b_etaPhi_unscaled_CM[...,list_allElems_ExceptPhi]).rsample((Nsamples,))
            thad_W_eta_unscaled_CM_sampled = self.flow_thad_W_CM_unscaled_eta(condition_thad_W_etaPhi_unscaled_CM[...,list_allElems_ExceptPhi]).rsample((Nsamples,))
            tlep_b_eta_unscaled_CM_sampled = self.flow_tlep_b_CM_unscaled_eta(condition_tlep_b_etaPhi_unscaled_CM[...,list_allElems_ExceptPhi]).rsample((Nsamples,))
            tlep_W_eta_unscaled_CM_sampled = self.flow_tlep_W_CM_unscaled_eta(condition_tlep_W_etaPhi_unscaled_CM[...,list_allElems_ExceptPhi]).rsample((Nsamples,))

            higgs_phi_unscaled_CM_sampled = self.flow_higgs_CM_unscaled_phi(condition_higgs_etaPhi_unscaled_CM[...,list_allElems_ExceptEta]).rsample((Nsamples,))
            thad_b_phi_unscaled_CM_sampled = self.flow_thad_b_CM_unscaled_phi(condition_thad_b_etaPhi_unscaled_CM[...,list_allElems_ExceptEta]).rsample((Nsamples,))
            thad_W_phi_unscaled_CM_sampled = self.flow_thad_W_CM_unscaled_phi(condition_thad_W_etaPhi_unscaled_CM[...,list_allElems_ExceptEta]).rsample((Nsamples,))
            tlep_b_phi_unscaled_CM_sampled = self.flow_tlep_b_CM_unscaled_phi(condition_tlep_b_etaPhi_unscaled_CM[...,list_allElems_ExceptEta]).rsample((Nsamples,))
            tlep_W_phi_unscaled_CM_sampled = self.flow_tlep_W_CM_unscaled_phi(condition_tlep_W_etaPhi_unscaled_CM[...,list_allElems_ExceptEta]).rsample((Nsamples,))

            higgs_etaPhi_unscaled_CM_sampled = torch.cat((higgs_eta_unscaled_CM_sampled, higgs_phi_unscaled_CM_sampled), dim=3)
            thad_b_etaPhi_unscaled_CM_sampled = torch.cat((thad_b_eta_unscaled_CM_sampled, thad_b_phi_unscaled_CM_sampled), dim=3)
            thad_W_etaPhi_unscaled_CM_sampled = torch.cat((thad_W_eta_unscaled_CM_sampled, thad_W_phi_unscaled_CM_sampled), dim=3)
            tlep_b_etaPhi_unscaled_CM_sampled = torch.cat((tlep_b_eta_unscaled_CM_sampled, tlep_b_phi_unscaled_CM_sampled), dim=3)
            tlep_W_etaPhi_unscaled_CM_sampled = torch.cat((tlep_W_eta_unscaled_CM_sampled, tlep_W_phi_unscaled_CM_sampled), dim=3)

            logit_scaled_ps_samples = torch.flatten(logit_scaled_ps_samples, start_dim=0, end_dim=1)
            higgs_etaPhi_unscaled_CM_sampled = torch.flatten(higgs_etaPhi_unscaled_CM_sampled, start_dim=0, end_dim=1)
            thad_b_etaPhi_unscaled_CM_sampled = torch.flatten(thad_b_etaPhi_unscaled_CM_sampled, start_dim=0, end_dim=1)
            thad_W_etaPhi_unscaled_CM_sampled = torch.flatten(thad_W_etaPhi_unscaled_CM_sampled, start_dim=0, end_dim=1)
            tlep_b_etaPhi_unscaled_CM_sampled = torch.flatten(tlep_b_etaPhi_unscaled_CM_sampled, start_dim=0, end_dim=1)
            tlep_W_etaPhi_unscaled_CM_sampled = torch.flatten(tlep_W_etaPhi_unscaled_CM_sampled, start_dim=0, end_dim=1)

            # ps is already in the logit space --> move to hypercube first
            logit_unscaled_ps_samples = logit_scaled_ps_samples * self.scaling_partons_CM_ps[1] + self.scaling_partons_CM_ps[0]
            ps_samples = torch.sigmoid(logit_unscaled_ps_samples[:,0]) # NOW IT's TRUE RAMBO + now dims = [B, 10]
            ps_samples = torch.clamp(ps_samples, min=0.0001, max=0.9999)
            
            samples_mask = ps_samples.isnan().sum(1) == 0 # by event
            # H thad tlep ISR
            propagators_unscaled_cartesian_CM_sampled, _, x1sample, x2sample = rambo.get_momenta_from_ps(ps_samples[samples_mask], requires_grad=True)

            zeros_pxpy = torch.zeros((x1sample.shape), device=self.device, dtype=self.dtype)
            boost_sampled_Epz_unscaled = torch.stack((rambo.collider_energy*(x1sample+x2sample)/2, zeros_pxpy, zeros_pxpy, rambo.collider_energy*(x1sample-x2sample)/2), dim=1)

            boost_vectors_B  = ps_utils.boostVector_t(boost_sampled_Epz_unscaled).unsqueeze(dim=1)  

            # + here not '-'
            propagators_unscaled_cartesian_lab_sampled = ps_utils.boost_tt(propagators_unscaled_cartesian_CM_sampled[:,-4:-1] , boost_vectors_B) #

            # get pt/eta/phi components and scale it and take the log for H/thad/tlep
            H_thad_tlep_sampled_unscaled_Eptetaphi_lab = Compute_ParticlesTensor.get_ptetaphi_comp_batch(propagators_unscaled_cartesian_lab_sampled) # witoput ISR
            H_thad_tlep_sampled_scaled_ptetaphi_lab = H_thad_tlep_sampled_unscaled_Eptetaphi_lab[...,1:].clone() 
            H_thad_tlep_sampled_scaled_ptetaphi_lab[...,0] = torch.log(H_thad_tlep_sampled_unscaled_Eptetaphi_lab[...,1] + 1)
            H_thad_tlep_sampled_scaled_ptetaphi_lab[...,0:2] = (H_thad_tlep_sampled_scaled_ptetaphi_lab[...,0:2] - log_mean_parton_Hthad[:2])/log_std_parton_Hthad[:2]

            # get E/pz components of boost + take the log of E
            boost_sampled_scaled = boost_sampled_Epz_unscaled[...,[0,3]].clone()
            boost_sampled_scaled[...,0] = torch.log(boost_sampled_Epz_unscaled[...,0] + 1)
            boost_sampled_scaled = (boost_sampled_scaled - log_mean_boost_parton)/log_std_boost_parton

            # log_mean_partons have only [E,pz] components
            # I think eta of decay products doesn't need to be scaled
            # but the propagators need scaling
            sampled_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab = Compute_ParticlesTensor.get_decayPartons_fromlab_propagators_angles(H_thad_tlep_sampled_scaled_ptetaphi_lab, # without ISR
                                                                  higgs_angles=higgs_etaPhi_unscaled_CM_sampled[:,0],
                                                                  thad_b_angles=thad_b_etaPhi_unscaled_CM_sampled[:,0],
                                                                  thad_W_angles=thad_W_etaPhi_unscaled_CM_sampled[:,0],
                                                                  tlep_b_angles=tlep_b_etaPhi_unscaled_CM_sampled[:,0],
                                                                  tlep_W_angles=tlep_W_etaPhi_unscaled_CM_sampled[:,0],
                                                                  boost=boost_sampled_scaled[:,None,:],
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
            
            return regressed_H_b1_b2_thad_b_q1_q2_tlep_b_el_nu_ISR_Eptetaphi_unscaled_lab, boost_regressed_Epz_scaled, \
                    sampled_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab, boost_sampled_scaled

        elif flow_eval == "both":

            flow_prob_logit_scaled_ps = self.flow_logit_scaled_ps(condition_logit_ps_scaled[:,0]).log_prob(logit_ps_scaled_target)

            # takes all elems except eta or phi (depending on the flow)
            list_allElems_ExceptPhi = [i for i in range(condition_higgs_etaPhi_unscaled_CM.shape[2]) if i != 1]
            list_allElems_ExceptEta = [i for i in range(condition_higgs_etaPhi_unscaled_CM.shape[2]) if i != 0]
            
            # eta flows
            flow_prob_higgs_eta = self.flow_higgs_CM_unscaled_eta(condition_higgs_etaPhi_unscaled_CM[...,list_allElems_ExceptPhi]).log_prob(higgs_etaPhi_unscaled_CM_target[...,0:1])
            flow_prob_thad_b_eta = self.flow_thad_b_CM_unscaled_eta(condition_thad_b_etaPhi_unscaled_CM[...,list_allElems_ExceptPhi]).log_prob(thad_etaPhi_unscaled_CM_target[:,0:1,0:1])
            flow_prob_thad_W_eta = self.flow_thad_W_CM_unscaled_eta(condition_thad_W_etaPhi_unscaled_CM[...,list_allElems_ExceptPhi]).log_prob(thad_etaPhi_unscaled_CM_target[:,1:2,0:1])
            flow_prob_tlep_b_eta = self.flow_tlep_b_CM_unscaled_eta(condition_tlep_b_etaPhi_unscaled_CM[...,list_allElems_ExceptPhi]).log_prob(tlep_etaPhi_unscaled_CM_target[:,0:1,0:1])
            flow_prob_tlep_W_eta = self.flow_tlep_W_CM_unscaled_eta(condition_tlep_W_etaPhi_unscaled_CM[...,list_allElems_ExceptPhi]).log_prob(tlep_etaPhi_unscaled_CM_target[:,1:2,0:1])

            # phi flows
            flow_prob_higgs_phi = self.flow_higgs_CM_unscaled_phi(condition_higgs_etaPhi_unscaled_CM[...,list_allElems_ExceptEta]).log_prob(higgs_etaPhi_unscaled_CM_target[...,1:2])
            flow_prob_thad_b_phi = self.flow_thad_b_CM_unscaled_phi(condition_thad_b_etaPhi_unscaled_CM[...,list_allElems_ExceptEta]).log_prob(thad_etaPhi_unscaled_CM_target[:,0:1,1:2])
            flow_prob_thad_W_phi = self.flow_thad_W_CM_unscaled_phi(condition_thad_W_etaPhi_unscaled_CM[...,list_allElems_ExceptEta]).log_prob(thad_etaPhi_unscaled_CM_target[:,1:2,1:2])
            flow_prob_tlep_b_phi = self.flow_tlep_b_CM_unscaled_phi(condition_tlep_b_etaPhi_unscaled_CM[...,list_allElems_ExceptEta]).log_prob(tlep_etaPhi_unscaled_CM_target[:,0:1,1:2])
            flow_prob_tlep_W_phi = self.flow_tlep_W_CM_unscaled_phi(condition_tlep_W_etaPhi_unscaled_CM[...,list_allElems_ExceptEta]).log_prob(tlep_etaPhi_unscaled_CM_target[:,1:2,1:2])

            flow_prob_higgs = flow_prob_higgs_eta + flow_prob_higgs_phi
            flow_prob_thad_b = flow_prob_thad_b_eta + flow_prob_thad_b_phi
            flow_prob_thad_W = flow_prob_thad_W_eta + flow_prob_thad_W_phi
            flow_prob_tlep_b = flow_prob_tlep_b_eta + flow_prob_tlep_b_phi
            flow_prob_tlep_W = flow_prob_tlep_W_eta + flow_prob_tlep_W_phi

            logit_scaled_ps_samples = self.flow_logit_scaled_ps(condition_logit_ps_scaled).rsample((Nsamples,))

            higgs_eta_unscaled_CM_sampled = self.flow_higgs_CM_unscaled_eta(condition_higgs_etaPhi_unscaled_CM[...,list_allElems_ExceptPhi]).rsample((Nsamples,))
            thad_b_eta_unscaled_CM_sampled = self.flow_thad_b_CM_unscaled_eta(condition_thad_b_etaPhi_unscaled_CM[...,list_allElems_ExceptPhi]).rsample((Nsamples,))
            thad_W_eta_unscaled_CM_sampled = self.flow_thad_W_CM_unscaled_eta(condition_thad_W_etaPhi_unscaled_CM[...,list_allElems_ExceptPhi]).rsample((Nsamples,))
            tlep_b_eta_unscaled_CM_sampled = self.flow_tlep_b_CM_unscaled_eta(condition_tlep_b_etaPhi_unscaled_CM[...,list_allElems_ExceptPhi]).rsample((Nsamples,))
            tlep_W_eta_unscaled_CM_sampled = self.flow_tlep_W_CM_unscaled_eta(condition_tlep_W_etaPhi_unscaled_CM[...,list_allElems_ExceptPhi]).rsample((Nsamples,))

            higgs_phi_unscaled_CM_sampled = self.flow_higgs_CM_unscaled_phi(condition_higgs_etaPhi_unscaled_CM[...,list_allElems_ExceptEta]).rsample((Nsamples,))
            thad_b_phi_unscaled_CM_sampled = self.flow_thad_b_CM_unscaled_phi(condition_thad_b_etaPhi_unscaled_CM[...,list_allElems_ExceptEta]).rsample((Nsamples,))
            thad_W_phi_unscaled_CM_sampled = self.flow_thad_W_CM_unscaled_phi(condition_thad_W_etaPhi_unscaled_CM[...,list_allElems_ExceptEta]).rsample((Nsamples,))
            tlep_b_phi_unscaled_CM_sampled = self.flow_tlep_b_CM_unscaled_phi(condition_tlep_b_etaPhi_unscaled_CM[...,list_allElems_ExceptEta]).rsample((Nsamples,))
            tlep_W_phi_unscaled_CM_sampled = self.flow_tlep_W_CM_unscaled_phi(condition_tlep_W_etaPhi_unscaled_CM[...,list_allElems_ExceptEta]).rsample((Nsamples,))
            
            higgs_etaPhi_unscaled_CM_sampled = torch.cat((higgs_eta_unscaled_CM_sampled, higgs_phi_unscaled_CM_sampled), dim=3)
            thad_b_etaPhi_unscaled_CM_sampled = torch.cat((thad_b_eta_unscaled_CM_sampled, thad_b_phi_unscaled_CM_sampled), dim=3)
            thad_W_etaPhi_unscaled_CM_sampled = torch.cat((thad_W_eta_unscaled_CM_sampled, thad_W_phi_unscaled_CM_sampled), dim=3)
            tlep_b_etaPhi_unscaled_CM_sampled = torch.cat((tlep_b_eta_unscaled_CM_sampled, tlep_b_phi_unscaled_CM_sampled), dim=3)
            tlep_W_etaPhi_unscaled_CM_sampled = torch.cat((tlep_W_eta_unscaled_CM_sampled, tlep_W_phi_unscaled_CM_sampled), dim=3)

            logit_scaled_ps_samples = torch.flatten(logit_scaled_ps_samples, start_dim=0, end_dim=1)
            higgs_etaPhi_unscaled_CM_sampled = torch.flatten(higgs_etaPhi_unscaled_CM_sampled, start_dim=0, end_dim=1)
            thad_b_etaPhi_unscaled_CM_sampled = torch.flatten(thad_b_etaPhi_unscaled_CM_sampled, start_dim=0, end_dim=1)
            thad_W_etaPhi_unscaled_CM_sampled = torch.flatten(thad_W_etaPhi_unscaled_CM_sampled, start_dim=0, end_dim=1)
            tlep_b_etaPhi_unscaled_CM_sampled = torch.flatten(tlep_b_etaPhi_unscaled_CM_sampled, start_dim=0, end_dim=1)
            tlep_W_etaPhi_unscaled_CM_sampled = torch.flatten(tlep_W_etaPhi_unscaled_CM_sampled, start_dim=0, end_dim=1)

            # ps is already in the logit space --> move to hypercube first
            logit_unscaled_ps_samples = logit_scaled_ps_samples * self.scaling_partons_CM_ps[1] + self.scaling_partons_CM_ps[0]
            ps_samples = torch.sigmoid(logit_unscaled_ps_samples[:,0]) # NOW IT's TRUE RAMBO + now dims = [B, 10]
            ps_samples = torch.clamp(ps_samples, min=0.001, max=0.999)

            #print('ps_samples')
            #print(torch.isnan(logit_unscaled_ps_samples).any())
            #print(torch.isnan(ps_samples).any())
            
            samples_mask = ps_samples.isnan().sum(1) == 0 # by event

            #print(torch.min(ps_samples))
            #print(torch.max(ps_samples))
            #print(ps_samples.shape)
            # H thad tlep ISR
            propagators_unscaled_cartesian_CM_sampled, _, x1sample, x2sample = rambo.get_momenta_from_ps(ps_samples[samples_mask], requires_grad=True)

            if torch.isnan(propagators_unscaled_cartesian_CM_sampled).any():
                print('NAN values')
                print(propagators_unscaled_cartesian_CM_sampled.shape)
                mask_events = torch.any(torch.isnan(propagators_unscaled_cartesian_CM_sampled), dim=(1,2))
                print(mask_events)
                print(ps_samples[mask_events])
                exit(0)

            zeros_pxpy = torch.zeros((x1sample.shape), device=self.device, dtype=self.dtype)
            boost_sampled_Epz_unscaled = torch.stack((rambo.collider_energy*(x1sample+x2sample)/2, zeros_pxpy, zeros_pxpy, rambo.collider_energy*(x1sample-x2sample)/2), dim=1)

            #print('propagators_unscaled_cartesian_CM_sampled')
            #print(torch.isnan(propagators_unscaled_cartesian_CM_sampled).any())
            #print(torch.isnan(boost_sampled_Epz_unscaled).any())

            boost_vectors_B  = ps_utils.boostVector_t(boost_sampled_Epz_unscaled).unsqueeze(dim=1)  

            # + here not '-'
            propagators_unscaled_cartesian_lab_sampled = ps_utils.boost_tt(propagators_unscaled_cartesian_CM_sampled[:,-4:-1] , boost_vectors_B) #

            # get pt/eta/phi components and scale it and take the log for H/thad/tlep
            H_thad_tlep_sampled_unscaled_Eptetaphi_lab = Compute_ParticlesTensor.get_ptetaphi_comp_batch(propagators_unscaled_cartesian_lab_sampled) # witoput ISR
            H_thad_tlep_sampled_scaled_ptetaphi_lab = H_thad_tlep_sampled_unscaled_Eptetaphi_lab[...,1:].clone() 
            H_thad_tlep_sampled_scaled_ptetaphi_lab[...,0] = torch.log(H_thad_tlep_sampled_unscaled_Eptetaphi_lab[...,1] + 1)
            H_thad_tlep_sampled_scaled_ptetaphi_lab[...,0:2] = (H_thad_tlep_sampled_scaled_ptetaphi_lab[...,0:2] - log_mean_parton_Hthad[:2])/log_std_parton_Hthad[:2]

            # get E/pz components of boost + take the log of E
            boost_sampled_scaled = boost_sampled_Epz_unscaled[...,[0,3]].clone()
            boost_sampled_scaled[...,0] = torch.log(boost_sampled_Epz_unscaled[...,0] + 1)
            boost_sampled_scaled = (boost_sampled_scaled - log_mean_boost_parton)/log_std_boost_parton

            #print('H_thad_tlep_sampled_scaled_ptetaphi_lab')
            #print(torch.isnan(H_thad_tlep_sampled_scaled_ptetaphi_lab).any())
            #print(torch.isnan(boost_sampled_scaled).any())

            # log_mean_partons have only [E,pz] components
            # I think eta of decay products doesn't need to be scaled
            # but the propagators need scaling
            sampled_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab = Compute_ParticlesTensor.get_decayPartons_fromlab_propagators_angles(H_thad_tlep_sampled_scaled_ptetaphi_lab, # without ISR
                                                                  higgs_angles=higgs_etaPhi_unscaled_CM_sampled[:,0],
                                                                  thad_b_angles=thad_b_etaPhi_unscaled_CM_sampled[:,0],
                                                                  thad_W_angles=thad_W_etaPhi_unscaled_CM_sampled[:,0],
                                                                  tlep_b_angles=tlep_b_etaPhi_unscaled_CM_sampled[:,0],
                                                                  tlep_W_angles=tlep_W_etaPhi_unscaled_CM_sampled[:,0],
                                                                  boost=boost_sampled_scaled[:,None,:],
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

            #print('sampled_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab')
            #print(torch.isnan(sampled_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab).any())

            return regressed_H_b1_b2_thad_b_q1_q2_tlep_b_el_nu_ISR_Eptetaphi_unscaled_lab, boost_regressed_Epz_scaled, \
                    flow_prob_logit_scaled_ps, flow_prob_higgs, flow_prob_thad_b, flow_prob_thad_W, flow_prob_tlep_b, \
                    flow_prob_tlep_W, \
                    sampled_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab, boost_sampled_scaled

        else:
            raise Exception(f"Invalid flow_eval mode {flow_eval}")
