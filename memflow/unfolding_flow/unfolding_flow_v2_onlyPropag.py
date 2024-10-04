import torch.nn as nn
import torch
import numpy as np
import utils
from memflow.unfolding_network.conditional_transformer_v3_OnlyPropagators import ConditioningTransformerLayer_v3_Propag

import zuko
from memflow.unfolding_flow.utils import Compute_ParticlesTensor as particle_tools
import memflow.phasespace.utils as ps_utils
from memflow.transfer_flow.periodicNSF_gaussian import NCSF_gaussian

from .custom_spline_flow_ps import Custom_spline_flow_ps
from .custom_spline_flow_eta import Custom_spline_flow_eta

from memflow.unfolding_flow.utils import Compute_ParticlesTensor
from memflow.phasespace.phasespace import PhaseSpace

class UnfoldingFlow_v2_onlyPropag(nn.Module):
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

                 DNN_condition=False,
                 DNN_layers=2,
                 DNN_dim=256,
                 DNN_output_dim=3,
                 
                 device=torch.device('cpu'),
                 dtype=torch.float32,
                 pretrained_model='',
                 load_conditioning_model=False):

        super(UnfoldingFlow_v2_onlyPropag, self).__init__()

        self.device = device
        self.dtype = dtype
        self.scaling_partons_CM_ps = scaling_partons_CM_ps
        
        self.cond_transformer = ConditioningTransformerLayer_v3_Propag(no_recoVars=4, # exist + 3-mom
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

            layers.append(nn.Linear(DNN_dim, DNN_output_dim, dtype=dtype))

            self.DNN_context = nn.Sequential(*layers)

        self.flow_logit_scaled_ps = Custom_spline_flow_ps(features=flow_nfeatures,
                              context=flow_ncond, 
                              transforms=flow_ntransforms, 
                              bins=flow_bins,
                              hidden_features=[flow_hiddenMLP_LayerDim]*flow_hiddenMLP_NoLayers, 
                              randperm=randPerm,
                              bound = flow_bound,
                              mean_gaussian = flow_base_first_arg,
                              std_gaussian = flow_base_second_arg,
                              passes= 2 if not flow_autoregressive else flow_nfeatures)

        if dtype == torch.float32:
            self.flow_logit_scaled_ps = self.flow_logit_scaled_ps.float()
        elif dtype == torch.float64:
            self.flow_logit_scaled_ps = self.flow_logit_scaled_ps.double()

    def disable_conditioner_regression_training(self):
        ''' Disable the conditioner regression training, but keep the
        latent space training'''
        self.cond_transformer.disable_regression_training()

    def enable_regression_training(self):
        self.cond_transformer.enable_regression_training()
        
    def forward(self,  logScaled_reco_Spanet, data_boost_reco,
                mask_recoParticles, mask_boost_reco,
                logit_ps_scaled_target, 
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
        boost_regressed_Epz_scaled = cond_X[1]
        free_latent_space = cond_X[2]

        H_thad_tlep_boost = [Hthadtlep_lab_ptetaphi_scaled[:,0], Hthadtlep_lab_ptetaphi_scaled[:,1],
                             Hthadtlep_lab_ptetaphi_scaled[:,2], boost_regressed_Epz_scaled[...,1]]

        # data_regressed_lab_ptetaphi = H thad tlep ISR pt eta phi
        (data_regressed_lab,
         data_regressed_lab_ptetaphi,
         boost_regressed) = particle_tools.get_HttISR_fromlab(H_thad_tlep_boost,
                                                              log_mean_parton_Hthad,
                                                              log_std_parton_Hthad,
                                                              log_mean_boost_parton,
                                                              log_std_boost_parton,
                                                              self.device, cartesian=True,
                                                              eps=1e-5,
                                                              return_both=True,
                                                              unscale_phi=False)

        data_regressed_lab_ptetaphi_scaled = data_regressed_lab_ptetaphi.clone()
        data_regressed_lab_ptetaphi_scaled[...,0] = torch.log(data_regressed_lab_ptetaphi[...,0] + 1)
        data_regressed_lab_ptetaphi_scaled[...,:2] = (data_regressed_lab_ptetaphi_scaled[...,:2] - log_mean_parton_Hthad[:2])/log_std_parton_Hthad[:2]

        mask = abs(Hthadtlep_lab_ptetaphi_scaled - data_regressed_lab_ptetaphi_scaled[:,:3]) > 1e-3
        if mask.any():
            raise Exception('Function `get_HttISR_fromlab` is not good')

        # Move to CM
        # Rescaling boost and gluon to return the scaled vectors for  regression losses
        # In principle there should be no problem here, but let's just keep it
        mask_wrong_boostE = torch.sqrt(boost_regressed[:, 0]**2 - boost_regressed[:, 3]**2) < particle_tools.M_MIN_TOT
        # print("N. events with wrong regressed boost", mask_wrong_boostE.sum())
        boost_regressed[mask_wrong_boostE, 0] = torch.sqrt(boost_regressed[mask_wrong_boostE, 3]**2 +  particle_tools.M_MIN_TOT**2 + 1e-3)

        # Note that we are not constraining x1x2 to be <1 . It is numerically constrained in get_PS
        boost_vectors_B  = ps_utils.boostVector_t(boost_regressed).unsqueeze(1) # [B, 1, 3]
        data_regressed_cm = ps_utils.boost_tt(data_regressed_lab , -boost_vectors_B) #[ B, 4particles,4]

        # Compute PS
        ps, detjinv_rambo_regr, mask_problematic =  particle_tools.get_PS(data_regressed_cm, boost_regressed)

        if ((ps<0)|(ps>1)).any():
            print("WRONG REGRESSED PS")
            breakpoint()

        # Now logit and scale PS
        logit_ps = torch.logit(ps, eps=5e-5) 
        logit_ps_scaled = (logit_ps - self.scaling_partons_CM_ps[0] ) / self.scaling_partons_CM_ps[1]

        if self.DNN_condition and free_latent_space.shape[2] > 0:
            free_latent_space = self.DNN_context(free_latent_space)
        
        # now we can concatenate the latent
        flow_cond_vector = torch.cat((logit_ps_scaled, free_latent_space[:,0]), dim=1)

        # And now we can use the flow model
        if flow_eval == "normalizing":
            flow_prob = self.flow_logit_scaled_ps(flow_cond_vector).log_prob(logit_ps_scaled_target)

            return data_regressed_lab_ptetaphi_scaled, boost_regressed_Epz_scaled, flow_prob

        
        elif flow_eval == "sampling":
            logit_scaled_ps_samples = self.flow_logit_scaled_ps(flow_cond_vector).rsample((Nsamples,))

            logit_unscaled_ps_samples = logit_scaled_ps_samples * self.scaling_partons_CM_ps[1] + self.scaling_partons_CM_ps[0]
            ps_samples = torch.sigmoid(logit_unscaled_ps_samples[:,0]) # NOW IT's TRUE RAMBO + now dims = [B, 10]
            #ps_samples = torch.clamp(ps_samples, min=0.001, max=0.999)

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
            propagators_unscaled_cartesian_lab_sampled = ps_utils.boost_tt(propagators_unscaled_cartesian_CM_sampled[:,-4:] , boost_vectors_B) #

            # get pt/eta/phi components and scale it and take the log for H/thad/tlep/ISR
            H_thad_tlep_ISR_sampled_unscaled_Eptetaphi_lab = Compute_ParticlesTensor.get_ptetaphi_comp_batch(propagators_unscaled_cartesian_lab_sampled)
            H_thad_tlep_ISR_sampled_scaled_ptetaphi_lab = H_thad_tlep_ISR_sampled_unscaled_Eptetaphi_lab[...,1:].clone() 
            H_thad_tlep_ISR_sampled_scaled_ptetaphi_lab[...,0] = torch.log(H_thad_tlep_ISR_sampled_unscaled_Eptetaphi_lab[...,1] + 1)
            H_thad_tlep_ISR_sampled_scaled_ptetaphi_lab[...,0:2] = (H_thad_tlep_ISR_sampled_scaled_ptetaphi_lab[...,0:2] - log_mean_parton_Hthad[:2])/log_std_parton_Hthad[:2]

            # get E/pz components of boost + take the log of E
            boost_sampled_scaled = boost_sampled_Epz_unscaled[...,[0,3]].clone()
            boost_sampled_scaled[...,0] = torch.log(boost_sampled_Epz_unscaled[...,0] + 1)
            boost_sampled_scaled = (boost_sampled_scaled - log_mean_boost_parton)/log_std_boost_parton
            
            return data_regressed_lab_ptetaphi_scaled, boost_regressed_Epz_scaled, H_thad_tlep_ISR_sampled_scaled_ptetaphi_lab, boost_sampled_scaled

        elif flow_eval == "both":
            flow_prob = self.flow_logit_scaled_ps(flow_cond_vector).log_prob(logit_ps_scaled_target)
            
            logit_scaled_ps_samples = self.flow_logit_scaled_ps(flow_cond_vector).rsample((Nsamples,))
            logit_scaled_ps_samples = torch.flatten(logit_scaled_ps_samples, start_dim=0, end_dim=1)

            logit_unscaled_ps_samples = logit_scaled_ps_samples * self.scaling_partons_CM_ps[1] + self.scaling_partons_CM_ps[0]
            ps_samples = torch.sigmoid(logit_unscaled_ps_samples) # NOW IT's TRUE RAMBO + now dims = [B, 10]
            ps_samples = torch.clamp(ps_samples, min=1e-3, max=1-1e-3)

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
            propagators_unscaled_cartesian_lab_sampled = ps_utils.boost_tt(propagators_unscaled_cartesian_CM_sampled[:,-4:] , boost_vectors_B) #

            # get pt/eta/phi components and scale it and take the log for H/thad/tlep
            H_thad_tlep_ISR_sampled_unscaled_Eptetaphi_lab = Compute_ParticlesTensor.get_ptetaphi_comp_batch(propagators_unscaled_cartesian_lab_sampled) # witoput ISR
            H_thad_tlep_ISR_sampled_scaled_ptetaphi_lab = H_thad_tlep_ISR_sampled_unscaled_Eptetaphi_lab[...,1:].clone() 
            H_thad_tlep_ISR_sampled_scaled_ptetaphi_lab[...,0] = torch.log(H_thad_tlep_ISR_sampled_unscaled_Eptetaphi_lab[...,1] + 1)
            H_thad_tlep_ISR_sampled_scaled_ptetaphi_lab[...,0:2] = (H_thad_tlep_ISR_sampled_scaled_ptetaphi_lab[...,0:2] - log_mean_parton_Hthad[:2])/log_std_parton_Hthad[:2]

            # get E/pz components of boost + take the log of E
            boost_sampled_scaled = boost_sampled_Epz_unscaled[...,[0,3]].clone()
            boost_sampled_scaled[...,0] = torch.log(boost_sampled_Epz_unscaled[...,0] + 1)
            boost_sampled_scaled = (boost_sampled_scaled - log_mean_boost_parton)/log_std_boost_parton
            
            return data_regressed_lab_ptetaphi_scaled, boost_regressed_Epz_scaled, flow_prob, \
                    H_thad_tlep_ISR_sampled_scaled_ptetaphi_lab, boost_sampled_scaled

        else:
            raise Exception(f"Invalid flow_eval mode {flow_eval}")
