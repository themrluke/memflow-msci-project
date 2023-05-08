import torch.nn as nn
import torch
import numpy as np
from memflow.unfolding_network.conditional_transformer import ConditioningTransformerLayer
import zuko
from zuko.flows import TransformModule, SimpleAffineTransform
from zuko.distributions import BoxUniform


class UnfoldingFlow(nn.Module):
    def __init__(self, no_jets, no_lept, jets_features, lepton_features, 
                 nfeatures_flow=12, ncond_flow=34, ntransforms_flow=5, hidden_mlp_flow=[128]*4, bins_flow=16,
                 autoregressive_flow=True, out_features_cond=32, nhead_cond=4, no_layers_cond=3, dtype=torch.float64):
        super(UnfoldingFlow, self).__init__()
        
        self.cond_transformer = ConditioningTransformerLayer(
                                            no_jets = no_jets,
                                            jets_features=jets_features, 
                                            no_lept = no_lept,
                                            lepton_features=lepton_features, 
                                            out_features=out_features_cond,
                                            nhead=nhead_cond,
                                            no_layers=no_layers_cond,
                                            dtype=dtype)      
        
        self.flow = zuko.flows.NSF(features=nfeatures_flow,
                              context=ncond_flow, 
                              transforms=ntransforms_flow, 
                              bins=bins_flow, 
                              hidden_features=hidden_mlp_flow, 
                              randperm=False,
                              base=BoxUniform,
                              base_args=[torch.ones(nfeatures_flow)*(-1),torch.ones(nfeatures_flow)], 
                              univariate_kwargs={"bound": 1 }, # Keeping the flow in the [-1,1] box.
                              passes= 2 if not autoregressive_flow else nfeatures_flow)

        self.flow.transforms.insert(0, SimpleAffineTransform(0*torch.ones(nfeatures_flow),1*torch.ones(nfeatures_flow),
                                                     -1*torch.ones(nfeatures_flow), 1*torch.ones(nfeatures_flow)))
        
        
    def forward(self, data):
        
        
        (data_ps, data_ps_detjacinv, mask_lepton, data_lepton, mask_jets,
        data_jets, mask_met, data_met,
        mask_boost_reco, data_boost_reco) =  data
            
        cond_X = self.cond_transformer(data_jets,
                                    data_lepton,
                                    data_met,
                                    data_boost_reco, 
                                    mask_jets, 
                                    mask_lepton, 
                                    mask_met, 
                                    mask_boost_reco)

        flow_result = self.flow(cond_X).log_prob(data_ps)
        detjac = data_ps_detjacinv.log()

        return flow_result, detjac, cond_X