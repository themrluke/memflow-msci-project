import torch.nn as nn
import torch
import numpy as np
import utils
from memflow.unfolding_network.conditional_transformer import ConditioningTransformerLayer
import zuko
from zuko.flows import TransformModule, SimpleAffineTransform
from zuko.distributions import BoxUniform
from zuko.distributions import DiagNormal
from memflow.unfolding_flow.utils import Compute_ParticlesTensor

class UnfoldingFlow(nn.Module):
    def __init__(self, log_mean, log_std, no_jets, no_lept, input_features, cond_hiddenFeatures=64,
                cond_dimFeedForward=512, cond_outFeatures=32, cond_nheadEncoder=4, cond_NoLayersEncoder=2,
                cond_nheadDecoder=4, cond_NoLayersDecoder=2, cond_NoDecoders=3, cond_aggregate=False,
                flow_nfeatures=12, flow_ncond=34, flow_ntransforms=5, flow_hiddenMLP_NoLayers=16,
                flow_hiddenMLP_LayerDim=128, flow_bins=16, flow_autoregressive=True, 
                flow_base=BoxUniform, flow_base_first_arg=-1, flow_base_second_arg=1, flow_bound=1., randPerm=False,
                affine_param_input1=0., affine_param_input2=1., affine_param_output1=-1., affine_param_output2=1.,
                device=torch.device('cpu'), dtype=torch.float64, model_path='', read_CondTransf=False, use_latent=False):

        super(UnfoldingFlow, self).__init__()

        self.log_mean = torch.tensor(log_mean, device=device)
        self.log_std = torch.tensor(log_std, device=device)
        self.cond_aggregate = cond_aggregate
        
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
                                    use_latent=use_latent,
                                    dtype=dtype)

        if read_CondTransf:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            self.cond_transformer.load_state_dict(state_dict['model_state_dict'])   
        
        self.flow = zuko.flows.NSF(features=flow_nfeatures,
                              context=flow_ncond, 
                              transforms=flow_ntransforms, 
                              bins=flow_bins, 
                              hidden_features=[flow_hiddenMLP_LayerDim]*flow_hiddenMLP_NoLayers, 
                              randperm=randPerm,
                              base=eval(flow_base),
                              base_args=[torch.ones(flow_nfeatures)*flow_base_first_arg, torch.ones(flow_nfeatures)*flow_base_second_arg],
                              univariate_kwargs={"bound": flow_bound }, # Keeping the flow in the [-1,1] box.
                              passes= 2 if not flow_autoregressive else flow_nfeatures)

        self.flow.transforms.insert(0, SimpleAffineTransform(affine_param_input1*torch.ones(flow_nfeatures),affine_param_input2*torch.ones(flow_nfeatures),
                                                     affine_param_output1*torch.ones(flow_nfeatures), affine_param_output2*torch.ones(flow_nfeatures)))
        
        
    def forward(self, mask_jets, mask_lepton_reco, mask_met, mask_boost_reco,
                logScaled_reco, data_boost_reco, 
                device, noProv, eps=0.0, order=[0,1,2,3], disableGradTransformer=False):

        mask_recoParticles = torch.cat((mask_jets, mask_lepton_reco, mask_met), dim=1)

        if (noProv):
            logScaled_reco = logScaled_reco[:,:,:-1]
        
        if disableGradTransformer:
            with torch.no_grad():
                cond_X = self.cond_transformer(logScaled_reco, data_boost_reco, mask_recoParticles, mask_boost_reco)
        else:
            cond_X = self.cond_transformer(logScaled_reco, data_boost_reco, mask_recoParticles, mask_boost_reco)


        if self.cond_aggregate:
            return cond_X
        else:
            
             HttISR_regressed, boost_regressed = Compute_ParticlesTensor.get_HttISR_numpy(cond_X, self.log_mean,
                                                                                         self.log_std, device, eps)

             # be careful at the order of phasespace_target and PS_regressed
             # order by default: H/thad/tlep/ISR
             PS_regressed, detjinv_regressed = Compute_ParticlesTensor.get_PS(HttISR_regressed, data_boost_reco)

             return cond_X, PS_regressed

