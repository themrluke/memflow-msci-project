import torch.nn as nn
import torch
import numpy as np
import utils
from memflow.unfolding_network.conditional_transformer_v3_OnlyDecay import ConditioningTransformerLayer_v3

import zuko
from zuko.flows import SimpleAffineTransform
from zuko.distributions import BoxUniform
from zuko.distributions import DiagNormal
from memflow.unfolding_flow.utils import Compute_ParticlesTensor as particle_tools
import memflow.phasespace.utils as ps_utils
from memflow.transfer_flow.periodicNSF_gaussian import NCSF_gaussian

class UnfoldingFlow_withDecay(nn.Module):
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

        super(UnfoldingFlow_withDecay, self).__init__()

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
            state_dict = torch.load(pretrained_model, map_location="cpu")
            self.cond_transformer.load_state_dict(state_dict['model_state_dict'])   

        self.flow = zuko.flows.NSF(features=flow_nfeatures,
                              context=flow_ncond, 
                              transforms=flow_ntransforms, 
                              bins=flow_bins, 
                              hidden_features=[flow_hiddenMLP_LayerDim]*flow_hiddenMLP_NoLayers, 
                              randperm=randPerm,
                              base=eval(flow_base),
                              base_args=[torch.ones(flow_nfeatures)*flow_base_first_arg, torch.ones(flow_nfeatures)*flow_base_second_arg],
                              univariate_kwargs={"bound": flow_bound }, # Keeping the flow in the [-B,B] box.
                              passes= 2 if not flow_autoregressive else flow_nfeatures)

        # flow for higgs angles --> transform pseudorapidity in angle
        self.flow_higgs_theta = zuko.flows.NSF(features=1,
                                          context=1 + flow_context_angles, 
                                          transforms=1,
                                          bins=flow_nbins_angles, 
                                          hidden_features=[flow_hiddenMLP_LayerDim_angles]*flow_hiddenMLP_NoLayers_angles, 
                                          randperm=randPerm_angles,
                                          base=eval(flow_base_anglesCM),
                                          base_args=[torch.ones(1)*flow_base_first_arg_anglesCM, torch.ones(1)*flow_base_second_arg_anglesCM],
                                          univariate_kwargs={"bound": 3.15 }, # Keeping the flow in the [-B,B] box.
                                          passes= 2 if not flow_autoregressive else 1)

        self.flow_thad_theta = zuko.flows.NSF(features=2,
                                          context=2 + flow_context_angles, 
                                          transforms=flow_ntransforms_angles, 
                                          bins=flow_nbins_angles, 
                                          hidden_features=[flow_hiddenMLP_LayerDim_angles]*flow_hiddenMLP_NoLayers_angles, 
                                          randperm=randPerm_angles,
                                          base=eval(flow_base_anglesCM),
                                          base_args=[torch.ones(2)*flow_base_first_arg_anglesCM, torch.ones(2)*flow_base_second_arg_anglesCM],
                                          univariate_kwargs={"bound": 3.15 }, # Keeping the flow in the [-B,B] box.
                                          passes= 2 if not flow_autoregressive else 1)

        self.flow_tlep_theta = zuko.flows.NSF(features=2,
                                          context=2 + flow_context_angles, 
                                          transforms=flow_ntransforms_angles, 
                                          bins=flow_nbins_angles, 
                                          hidden_features=[flow_hiddenMLP_LayerDim_angles]*flow_hiddenMLP_NoLayers_angles, 
                                          randperm=randPerm_angles,
                                          base=eval(flow_base_anglesCM),
                                          base_args=[torch.ones(2)*flow_base_first_arg_anglesCM, torch.ones(2)*flow_base_second_arg_anglesCM],
                                          univariate_kwargs={"bound": 3.15 }, # Keeping the flow in the [-B,B] box.
                                          passes= 2 if not flow_autoregressive else 1)


        # flow for thad angles --> transform pseudorapidity in angle
        self.flow_higgs_phi = NCSF_gaussian(features=1,
                              context=2 + flow_context_angles,   # condition on regressed phi + sampled eta
                              transforms=1, 
                              bins=flow_nbins_angles, 
                              hidden_features=[flow_hiddenMLP_LayerDim_angles]*flow_hiddenMLP_NoLayers_angles, 
                              randperm=randPerm_angles,
                              passes= 2 if not flow_autoregressive else 1)
        
        self.flow_thad_phi = NCSF_gaussian(features=2,
                              context=4 + flow_context_angles,      # condition on regressed phi + sampled eta
                              transforms=flow_ntransforms_angles, 
                              bins=flow_nbins_angles, 
                              hidden_features=[flow_hiddenMLP_LayerDim_angles]*flow_hiddenMLP_NoLayers_angles, 
                              randperm=randPerm_angles,
                              passes= 2 if not flow_autoregressive else 1)

        self.flow_tlep_phi = NCSF_gaussian(features=2,      
                              context=4 + flow_context_angles,         # condition on regressed phi + sampled eta
                              transforms=flow_ntransforms_angles,
                              bins=flow_nbins_angles,
                              hidden_features=[flow_hiddenMLP_LayerDim_angles]*flow_hiddenMLP_NoLayers_angles,
                              randperm=randPerm_angles,
                              passes= 2 if not flow_autoregressive else 1)
        
        


    def disable_conditioner_regression_training(self):
        ''' Disable the conditioner regression training, but keep the
        latent space training'''
        self.cond_transformer.disable_regression_training()

    def enable_regression_training(self):
        self.cond_transformer.enable_regression_training()
        
    def forward(self,  logScaled_reco_Spanet, data_boost_reco,
                mask_recoParticles, mask_boost_reco,
                ps_target, higgs_angles_target,
                thad_angles_target, tlep_angles_target,
                log_mean_parton, log_std_parton,
                log_mean_boost_parton, log_std_boost_parton,
                log_mean_parton_Hthad, log_std_parton_Hthad,
                order=[0,1,2,3], disableGradConditioning =False,
                flow_eval="normalizing", Nsamples=0, No_regressed_vars=9,
                sin_cos_embedding=False, sin_cos_reco=None, sin_cos_partons=None,
                attach_position_regression=None):


        if disableGradConditioning:  # do no train cond transformer at all with sampling epoch
            with torch.no_grad():
                cond_X = self.cond_transformer(logScaled_reco_Spanet, data_boost_reco[:,:,[0,3]],
                                               mask_recoParticles, mask_boost_reco,
                                               No_regressed_vars = No_regressed_vars, sin_cos_reco = pos_logScaledReco,
                                               sin_cos_partons=pos_partons, sin_cos_embedding=True,
                                               attach_position=attach_position_regression, eps_arctanh=0.)
        else:
            cond_X = self.cond_transformer(logScaled_reco_Spanet, data_boost_reco[:,:,[0,3]],
                                           mask_recoParticles, mask_boost_reco,
                                           No_regressed_vars = No_regressed_vars, sin_cos_reco = pos_logScaledReco,
                                           sin_cos_partons=pos_partons, sin_cos_embedding=True,
                                           attach_position=attach_position_regression, eps_arctanh=0.)


        propagators_momenta = cond_X[0]
        decayVars_momenta = cond_X[1]
        boost_regressed = cond_X[2]

        higgs_mass = torch.tensor([125.25], device=self.device, dtype=self.dtype)
        top_mass = torch.tensor([172.5], device=self.device, dtype=self.dtype)
        W_mass = torch.tensor([80.4], device=self.device, dtype=self.dtype)
        b_mass = torch.tensor([0.0], device=self.device, dtype=self.dtype)

        higgs_mass_tensor =  higgs_mass.repeat(logScaled_partons.shape[0])
        top_mass_tensor = top_mass.repeat(logScaled_partons.shape[0])
        W_mass_tensor = W_mass.repeat(logScaled_partons.shape[0])
        b_mass_tensor = b_mass.repeat(logScaled_partons.shape[0])

        # log_mean_partons have only [E,pz] components
        H_b1_b2_thad_b_q1_q2_tlep_b_el_nu_ISR = Compute_ParticlesTensor.get_decayPartons_fromlab_propagators_angles(propagators_momenta,
                                                      higgs_angles=decayVars_momenta[:,0],
                                                      thad_b_angles=decayVars_momenta[:,1],
                                                      thad_W_angles=decayVars_momenta[:,2],
                                                      tlep_b_angles=decayVars_momenta[:,3],
                                                      tlep_W_angles=decayVars_momenta[:,4],
                                                      boost=boost_regressed,  # here must be scaled pt, scaled eta, phi
                                                      log_mean_parton_lab=log_mean_parton, log_std_parton_lab=log_std_parton,
                                                      log_mean_boost=log_mean_boost_parton, log_std_boost=log_std_boost_parton,
                                                      log_mean_parton_Hthadtlep=log_mean_parton_Hthad, log_std_parton_Hthadtlep=log_std_parton_Hthad,
                                                      device=self.device,
                                                      higgs_mass=higgs_mass_tensor,
                                                      thad_mass=top_mass_tensor,
                                                      tlep_mass=top_mass_tensor,
                                                      W_had_mass=W_mass_tensor,
                                                      W_lep_mass=W_mass_tensor,
                                                      b_mass=b_mass_tensor,
                                                      cartesian=False, eps=1e-4,
                                                      pt_cut=None, unscale_phi=False, debug=False,
                                                      final_scaling=False)
        # unscale boost to move to CM frame
        boost_regressed = boost_regressed*log_std_boost + log_mean_boost # boost and mean contains only E and pz
        
        boost_regr_copy = boost_regressed.clone()
        boost_regressed[:,:,0] = torch.exp(torch.abs(boost_regr_copy[:,:,0])) - 1
        
        # Move to CM
        # Rescaling boost and gluon to return the scaled vectors for  regression losses
        # In principle there should be no problem here, but let's just keep it
        mask_wrong_boostE = torch.sqrt(boost_regressed[:, 0]**2 - boost_regressed[:, 3]**2) < particle_tools.M_MIN_TOT
        # print("N. events with wrong regressed boost", mask_wrong_boostE.sum())
        boost_regressed[mask_wrong_boostE, 0] = torch.sqrt(boost_regressed[mask_wrong_boostE, 3]**2 +  particle_tools.M_MIN_TOT**2 + 1e-3)

        # Note that we are not constraining x1x2 to be <1 . It is numerically constrained in get_PS
        boost_vectors_B  = ps_utils.boostVector_t(boost_regressed).unsqueeze(1) # [B, 1, 3]
        data_propagators_regressed_lab = H_b1_b2_thad_b_q1_q2_tlep_b_el_nu_ISR[:,[0,3,7,11]]
        
        data_regressed_cm = ps_utils.boost_tt(data_propagators_regressed_lab , -boost_vectors_B) #[ B, 4particles,4]

        # Compute PS
        ps, detjinv_rambo_regr, mask_problematic =  particle_tools.get_PS(data_regressed_cm, boost_regressed)

        if ((ps<0)|(ps>1)).any():
            print("WRONG REGRESSED PS")
            breakpoint()

        # Now logit and scale PS
        logit_ps = torch.logit(ps, eps=5e-5) 
        logit_ps_scaled = (logit_ps - self.scaling_partons_CM_ps[0] ) / self.scaling_partons_CM_ps[1]

        flow_cond_vector = logit_ps_scaled

        # now we can concatenate the latent
        #latent = cond_X[4]
        #flow_cond_vector = torch.cat((logit_ps_scaled, latent), dim=1)

        flow_higgs_context = H_b1_b2_thad_b_q1_q2_tlep_b_el_nu_ISR[:,1,[2,3]] # Higgs b1: eta and phi
        flow_thad_context = H_b1_b2_thad_b_q1_q2_tlep_b_el_nu_ISR[:,[4,5],[2,3]] # thad b & q1: eta and phi
        flow_tlep_context = H_b1_b2_thad_b_q1_q2_tlep_b_el_nu_ISR[:,[-4,-3],[2,3]] # tlep b & el: eta and phi

        # change from eta to theta to have periodicity
        flow_higgs_context[...,0] = 2*torch.atan(torch.exp(-1*flow_higgs_context[...,0]))
        flow_thad_context[...,0] = 2*torch.atan(torch.exp(-1*flow_thad_context[...,0]))
        flow_tlep_context[...,0] = 2*torch.atan(torch.exp(-1*flow_tlep_context[...,0]))

        # unscale eta for target decay angles
        higgs_angles_target[...,0] = higgs_angles_target[...,0]*log_std_parton[1] + log_mean_parton[1]
        thad_angles_target[...,0] = thad_angles_target[...,0]*log_std_parton[1] + log_mean_parton[1]
        tlep_angles_target[...,0] = tlep_angles_target[...,0]*log_std_parton[1] + log_mean_parton[1]

        # change from unscaled eta to theta to have an angle
        higgs_angles_target[...,0] = 2*torch.atan(torch.exp(-1*higgs_angles_target[...,0]))
        thad_angles_target[...,0] = 2*torch.atan(torch.exp(-1*thad_angles_target[...,0]))
        tlep_angles_target[...,0] = 2*torch.atan(torch.exp(-1*tlep_angles_target[...,0]))

        
        # And now we can use the flow model
        if flow_eval == "normalizing":
            flow_prob = self.flow(flow_cond_vector).log_prob(ps_target)

            condition_higgs_theta = torch.cat((flow_higgs_context[...,0], new_token), dim=2)
            condition_thad_theta = torch.cat((flow_thad_context[...,0], new_token), dim=2)
            condition_tlep_theta = torch.cat((flow_tlep_context[...,0], new_token), dim=2)

            higgs_theta_sampled = self.flow_higgs_theta(condition_higgs_theta).sample((1,))
            thad_theta_sampled = self.flow_thad_theta(condition_thad_theta).sample((1,))
            tlep_theta_sampled = self.flow_tlep_theta(condition_tlep_theta).sample((1,))

            condition_higgs_phi = torch.cat((flow_higgs_context[...,1], new_token, higgs_theta_sampled), dim=2)
            condition_thad_phi = torch.cat((flow_thad_context[...,1], new_token, thad_theta_sampled), dim=2)
            condition_tlep_phi = torch.cat((flow_tlep_context[...,1], new_token, tlep_theta_sampled), dim=2)
            
            flow_prob_higgs_theta = self.flow_higgs_theta(condition_higgs_theta).log_prob(higgs_angles_target[...,0])
            flow_prob_thad_theta = self.flow_thad_theta(condition_thad_theta).log_prob(thad_angles_target[...,0])
            flow_prob_tlep_theta = self.flow_tlep_theta(condition_tlep_theta).log_prob(tlep_angles_target[...,0])

            flow_prob_higgs_phi = self.flow_higgs_phi(condition_higgs_phi).log_prob(higgs_angles_target[...,1])
            flow_prob_thad_phi = self.flow_thad_phi(condition_thad_phi).log_prob(thad_angles_target[...,1])
            flow_prob_tlep_phi = self.flow_tlep_phi(condition_tlep_phi).log_prob(tlep_angles_target[...,1])

            return H_b1_b2_thad_b_q1_q2_tlep_b_el_nu_ISR, flow_prob, flow_prob_higgs_theta, flow_prob_thad_theta, flow_prob_tlep_theta, \
                    flow_prob_higgs_phi, flow_prob_thad_phi, flow_prob_tlep_phi
        
        elif flow_eval == "sampling":
            ps_samples = self.flow(flow_cond_vector).rsample((Nsamples,))

            condition_higgs_theta = torch.cat((flow_higgs_context[...,0], new_token), dim=2)
            condition_thad_theta = torch.cat((flow_thad_context[...,0], new_token), dim=2)
            condition_tlep_theta = torch.cat((flow_tlep_context[...,0], new_token), dim=2)

            higgs_theta_sampled = self.flow_higgs_theta(condition_higgs_theta).rsample((Nsamples,))
            thad_theta_sampled = self.flow_thad_theta(condition_thad_theta).rsample((Nsamples,))
            tlep_theta_sampled = self.flow_tlep_theta(condition_tlep_theta).rsample((Nsamples,))

            condition_higgs_phi = torch.cat((flow_higgs_context[...,1], new_token, higgs_theta_sampled), dim=2)
            condition_thad_phi = torch.cat((flow_thad_context[...,1], new_token, thad_theta_sampled), dim=2)
            condition_tlep_phi = torch.cat((flow_tlep_context[...,1], new_token, tlep_theta_sampled), dim=2)

            samples_higgs_phi = self.flow_higgs_phi(condition_higgs_phi).rsample((Nsamples,))
            samples_thad_phi = self.flow_thad_phi(condition_thad_phi).rsample((Nsamples,))
            samples_tlep_phi = self.flow_tlep_phi(condition_tlep_phi).rsample((Nsamples,))

            # move from theta back to eta
            samples_higgs_eta = -1*torch.log(torch.tan(higgs_theta_sampled/2.))
            samples_thad_eta = -1*torch.log(torch.tan(thad_theta_sampled/2.))
            samples_tlep_eta = -1*torch.log(torch.tan(tlep_theta_sampled/2.))

            # scale eta to send "scaled eta" to the function which returns full parton event
            samples_higgs_eta = (samples_higgs_eta - log_mean_parton[1])/log_std_parton[1]
            samples_thad_eta = (samples_thad_eta - log_mean_parton[1])/log_std_parton[1]
            samples_higgs_eta = (samples_higgs_eta - log_mean_parton[1])/log_std_parton[1]

            # higgs angles, thad angles, tlep angles
            sampled_higgs_angles = torch.cat((samples_higgs_eta, samples_higgs_phi), dim=2)
            sampled_thad_angles = torch.cat((samples_thad_eta, samples_thad_phi), dim=2)
            sampled_tlep_angles = torch.cat((samples_tlep_eta, samples_tlep_phi), dim=2)

            # TODO: ps is already in the logit space --> move to hypercube first
            ps_samples_unscaled = ps_samples_unscaled*self.scaling_partons_CM_ps[1] + self.scaling_partons_CM_ps[0]
            ps_samples_unscaled = torch.sigmoid(ps_samples_unscaled) # NOW IT's TRUE RAMBO
            
            samples_mask = ps_samples_unscaled.isnan().sum(1) == 0 # by event
            # H thad tlep ISR
            propagators_sampled, _, x1sample, x2sample = rambo.get_momenta_from_ps(ps_samples_unscaled[samples_mask], requires_grad=True)

            boost_sampled = x1sample + x2sample

            # here the boost is not scaled

            # log_mean_partons have only [E,pz] components
            sampled_H_b1_b2_thad_b_q1_q2_tlep_b_el_nu_ISR = Compute_ParticlesTensor.get_decayPartons_fromlab_propagators_angles(propagators_sampled[:,:3],
                                                                  higgs_angles=sampled_higgs_angles,
                                                                  thad_b_angles=sampled_thad_angles[:,0],
                                                                  thad_W_angles=sampled_thad_angles[:,1],
                                                                  tlep_b_angles=sampled_tlep_angles[:,0],
                                                                  tlep_W_angles=sampled_tlep_angles[:,1],
                                                                  boost=boost_sampled,
                                                                  log_mean_parton_lab=log_mean_parton, log_std_parton_lab=log_std_parton,
                                                                  log_mean_boost=log_mean_boost_parton, log_std_boost=log_std_boost_parton,
                                                                  log_mean_parton_Hthadtlep=log_mean_parton_Hthad, log_std_parton_Hthadtlep=log_std_parton_Hthad,
                                                                  device=self.device,
                                                                  higgs_mass=higgs_mass_tensor,
                                                                  thad_mass=top_mass_tensor,
                                                                  tlep_mass=top_mass_tensor,
                                                                  W_had_mass=W_mass_tensor,
                                                                  W_lep_mass=W_mass_tensor,
                                                                  b_mass=b_mass_tensor,
                                                                  cartesian=False, eps=1e-4,
                                                                  pt_cut=None, unscale_phi=False, debug=False,
                                                                  final_scaling=False)
            
            return H_b1_b2_thad_b_q1_q2_tlep_b_el_nu_ISR, sampled_H_b1_b2_thad_b_q1_q2_tlep_b_el_nu_ISR

        elif flow_eval == "both":
            samples = self.flow(flow_cond_vector).rsample((Nsamples,))

            condition_higgs_theta = torch.cat((flow_higgs_context[...,0], new_token), dim=2)
            condition_thad_theta = torch.cat((flow_thad_context[...,0], new_token), dim=2)
            condition_tlep_theta = torch.cat((flow_tlep_context[...,0], new_token), dim=2)

            higgs_theta_sampled = self.flow_higgs_theta(condition_higgs_theta).rsample((Nsamples,))
            thad_theta_sampled = self.flow_thad_theta(condition_thad_theta).rsample((Nsamples,))
            tlep_theta_sampled = self.flow_tlep_theta(condition_tlep_theta).rsample((Nsamples,))

            condition_higgs_phi = torch.cat((flow_higgs_context[...,1], new_token, higgs_theta_sampled), dim=2)
            condition_thad_phi = torch.cat((flow_thad_context[...,1], new_token, thad_theta_sampled), dim=2)
            condition_tlep_phi = torch.cat((flow_tlep_context[...,1], new_token, tlep_theta_sampled), dim=2)

            samples_higgs_eta = -1*torch.log(torch.tan(higgs_theta_sampled/2.))
            samples_thad_eta = -1*torch.log(torch.tan(thad_theta_sampled/2.))
            samples_tlep_eta = -1*torch.log(torch.tan(tlep_theta_sampled/2.))

            samples_higgs_phi = self.flow_higgs_phi(condition_higgs_phi).rsample((Nsamples,))
            samples_thad_phi = self.flow_thad_phi(condition_thad_phi).rsample((Nsamples,))
            samples_tlep_phi = self.flow_tlep_phi(condition_tlep_phi).rsample((Nsamples,))

            flow_prob_higgs_theta = self.flow_higgs_theta(condition_higgs_theta).log_prob(higgs_angles_target[...,0])
            flow_prob_thad_theta = self.flow_thad_theta(condition_thad_theta).log_prob(thad_angles_target[...,0])
            flow_prob_tlep_theta = self.flow_tlep_theta(condition_tlep_theta).log_prob(tlep_angles_target[...,0])

            flow_prob_higgs_phi = self.flow_higgs_phi(condition_higgs_phi).log_prob(higgs_angles_target[...,1])
            flow_prob_thad_phi = self.flow_thad_phi(condition_thad_phi).log_prob(thad_angles_target[...,1])
            flow_prob_tlep_phi = self.flow_tlep_phi(condition_tlep_phi).log_prob(tlep_angles_target[...,1])

            return H_b1_b2_thad_b_q1_q2_tlep_b_el_nu_ISR, flow_prob, flow_prob_higgs_theta, flow_prob_thad_theta, flow_prob_tlep_theta, \
                    flow_prob_higgs_phi, flow_prob_thad_phi, flow_prob_tlep_phi, \
                    samples, samples_higgs_eta, samples_thad_eta, samples_tlep_eta, \
                    samples_higgs_phi, samples_thad_phi, samples_tlep_phi


        else:
            raise Exception(f"Invalid flow_eval mode {flow_eval}")
