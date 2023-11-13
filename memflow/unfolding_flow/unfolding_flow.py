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

class UnfoldingFlow(nn.Module):
    def __init__(self,
                 scaling_partons_lab,
                 scaling_boost_lab,
                 scaling_partons_CM_ps,
                 no_jets, no_lept,
                 input_features,
                 cond_hiddenFeatures=64,
                 cond_dimFeedForward=512,
                 cond_outFeatures=32,
                 cond_nheadEncoder=4,
                 cond_NoLayersEncoder=2,
                 cond_nheadDecoder=4,
                 cond_NoLayersDecoder=2,
                 cond_NoDecoders=3,
                 cond_aggregate=False,
                 cond_use_latent=False,
                 cond_out_features_latent=None,
                 cond_no_layers_decoder_latent=None,
                 
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
                 
                 device=torch.device('cpu'),
                 dtype=torch.float32,
                 pretrained_model='',
                 load_conditioning_model=False,
                 eps=1e-4):

        super(UnfoldingFlow, self).__init__()

        self.device = device
        self.dtype = dtype
        self.scaling_partons_lab = scaling_partons_lab
        self.scaling_boost_lab = scaling_boost_lab
        self.scaling_partons_CM_ps = scaling_partons_CM_ps
        
        self.cond_aggregate = cond_aggregate
        self.eps = eps # used for small values like the mass of the gluon for numerical reasons
        
        self.cond_transformer = ConditioningTransformerLayer(
                                    no_jets = no_jets,
                                    no_lept = no_lept,
                                    input_features=input_features,
                                    hidden_features=cond_hiddenFeatures,
                                    dim_feedforward_transformer=cond_dimFeedForward,
                                    out_features=cond_outFeatures,
                                    nhead_encoder=cond_nheadEncoder,
                                    no_layers_encoder=cond_NoLayersEncoder,
                                    nhead_decoder=cond_nheadDecoder,
                                    no_layers_decoder=cond_NoLayersDecoder,
                                    no_decoders=cond_NoDecoders,
                                    aggregate=cond_aggregate,
                                    use_latent=cond_use_latent,
                                    out_features_latent=cond_out_features_latent,
                                    no_layers_decoder_latent=cond_no_layers_decoder_latent, 
                                    dtype=dtype)

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


    def disable_conditioner_regression_training(self):
        ''' Disable the conditioner regression training, but keep the
        latent space training'''
        self.cond_transformer.disable_regression_training()

    def enable_regression_training(self):
        self.cond_transformer.enable_regression_training()
        
    def forward(self,  logScaled_reco, data_boost_reco,
                mask_recoParticles, mask_boost_reco,
                ps_target,  order=[0,1,2,3], disableGradConditioning =False,
                flow_eval="normalizing", Nsamples=0):


        if disableGradConditioning:  # do no train cond transformer at all with sampling epoch
            with torch.no_grad():
                cond_X = self.cond_transformer(logScaled_reco, data_boost_reco,
                                               mask_recoParticles,
                                               mask_boost_reco)
        else:
            cond_X = self.cond_transformer(logScaled_reco, data_boost_reco,
                                           mask_recoParticles, mask_boost_reco)


        # Now transforming the output
        higgs = cond_X[0]
        thad = cond_X[1]
        tlep = cond_X[2]
        
        (data_regressed_lab,
         data_regressed_lab_ptetaphi,
         boost_regressed) = particle_tools.get_HttISR_fromlab(cond_X, self.scaling_partons_lab[0],
                                                                                         self.scaling_partons_lab[1],
                                                                                         self.scaling_boost_lab[0],
                                                                                         self.scaling_boost_lab[1],
                                                                                         self.device, cartesian=True,
                                                                                         eps=self.eps,
                                                                                         return_both=True)

        # Move to CM
        # Rescaling boost and gluon to return the scaled vectors for  regression losses
        # In principle there should be no problem here, but let's just keep it
        mask_wrong_boostE = torch.sqrt(boost_regressed[:, 0]**2 - boost_regressed[:, 3]**2) < particle_tools.M_MIN_TOT
        # print("N. events with wrong regressed boost", mask_wrong_boostE.sum())
        boost_regressed[mask_wrong_boostE, 0] = torch.sqrt(boost_regressed[mask_wrong_boostE, 3]**2 +  particle_tools.M_MIN_TOT**2 + 1e-3)

        # Note that we are not constraining x1x2 to be <1 . It is numerically constrained in get_PS
        boost_vectors_B  = ps_utils.boostVector_t(boost_regressed).unsqueeze(1) # [B, 1, 3]
        data_regressed_cm = ps_utils.boost_tt(data_regressed_lab , -boost_vectors_B) #[ B, 4particles,4]

        
        boost_notscaled = boost_regressed[:, [0,-1]] # Only E and pz components are used for the 
        boost = boost_notscaled.clone()
        boost[:,0] = torch.log(boost_notscaled[:,0] + 1)
        boost = (boost - self.scaling_boost_lab[0]) / self.scaling_boost_lab[1]
        
        gluon_toscale = data_regressed_lab_ptetaphi[:,3] #pt, eta, phi
        gluon = gluon_toscale.clone()
        gluon[:,0] = torch.log(gluon_toscale[:,0] +1)
        gluon = (gluon - self.scaling_partons_lab[0]) / self.scaling_partons_lab[1]


        # Compute PS
        ps, detjinv_rambo_regr, mask_problematic =  particle_tools.get_PS(data_regressed_cm, boost_regressed)

        if ((ps<0)|(ps>1)).any():
            print("WRONG REGRESSED PS")
            breakpoint()

        # Now logit and scale PS
        logit_ps = torch.logit(ps, eps=5e-5) 
        logit_ps_scaled = (logit_ps - self.scaling_partons_CM_ps[0] ) / self.scaling_partons_CM_ps[1]

        # now we can concatenate the latent
        latent = cond_X[4]
        flow_cond_vector = torch.cat((logit_ps_scaled, latent), dim=1)
        
        # And now we can use the flow model
        if flow_eval == "normalizing":
            flow_prob = self.flow(flow_cond_vector).log_prob(ps_target)

            return [higgs, thad, tlep, gluon, boost], data_regressed_cm, \
                ps, logit_ps_scaled, flow_cond_vector, flow_prob, mask_problematic

        
        elif flow_eval == "sampling":
            samples = self.flow(flow_cond_vector).rsample((Nsamples,))
            
            return [higgs, thad, tlep, gluon, boost], data_regressed_cm, \
                ps, logit_ps_scaled, flow_cond_vector, samples, mask_problematic

        elif flow_eval == "conditioning_only":
            return [higgs, thad, tlep, gluon, boost], data_regressed_cm, \
                ps, logit_ps_scaled, flow_cond_vector, None, mask_problematic

        else:
            raise Exception(f"Invalid flow_eval mode {flow_eval}")
