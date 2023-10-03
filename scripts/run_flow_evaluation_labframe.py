from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
import h5py
import torch
from memflow.read_data.dataset_all import DatasetCombined
from memflow.unfolding_flow.unfolding_flow import UnfoldingFlow

from memflow.unfolding_network.conditional_transformer import ConditioningTransformerLayer
from memflow.unfolding_flow.utils import *
from memflow.unfolding_flow.mmd_loss import MMD
from memflow.unfolding_flow.utils import Compute_ParticlesTensor

import numpy as np
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.functional import normalize
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from math import floor, ceil

import mdmm
# from tensorboardX import SummaryWriter
from omegaconf import OmegaConf
import sys
import argparse
import os
from pynvml import *
import vector

from utils import constrain_energy
from utils import total_mom
from rich.progress import track
from earlystop import EarlyStopper
from memflow.unfolding_flow.utils import Compute_ParticlesTensor

PI = torch.pi

    

def validation( device, config, model_weights, outputPath, N_samples, N_events, dtype):
 

    print("Loading datasets")
    test_dataset = DatasetCombined(config.input_dataset_test, dev=device,
                                    dtype=dtype, datasets=['partons_lab', 'reco_lab', 'partons_CM'],
                           reco_list_lab=['scaledLogRecoParticles', 'mask_lepton', 
                                      'mask_jets','mask_met',
                                      'mask_boost', 'scaledLogBoost'],
                           parton_list_lab=['logScaled_data_higgs_t_tbar_ISR',
                                        'logScaled_data_boost',
                                        'mean_log_data_higgs_t_tbar_ISR',
                                        'std_log_data_higgs_t_tbar_ISR',
                                        'mean_log_data_boost',
                                            'std_log_data_boost'],
                           parton_list_cm=['phasespace_intermediateParticles_onShell_logit',
                                           'phasespace_intermediateParticles_onShell_logit_scaled',
                                           'mean_phasespace_intermediateParticles_onShell_logit',
                                            'std_phasespace_intermediateParticles_onShell_logit'])


    log_mean_parton = test_dataset.partons_lab.mean_log_data_higgs_t_tbar_ISR
    log_std_parton = test_dataset.partons_lab.std_log_data_higgs_t_tbar_ISR
    log_mean_boost = test_dataset.partons_lab.mean_log_data_boost
    log_std_boost = test_dataset.partons_lab.std_log_data_boost
    mean_ps = test_dataset.partons_CM.mean_phasespace_intermediateParticles_onShell_logit
    scale_ps = test_dataset.partons_CM.std_phasespace_intermediateParticles_onShell_logit

    # Initialize model
    model = UnfoldingFlow(
                    pretrained_model=None,
                    load_conditioning_model=False,
                    scaling_partons_lab = [log_mean_parton, log_std_parton],
                    scaling_boost_lab = [log_mean_boost, log_std_boost],
                    scaling_partons_CM_ps = [mean_ps, scale_ps],

                    no_jets=config.input_shape.number_jets,
                    no_lept=config.input_shape.number_lept,
                    input_features=config.input_shape.input_features,
                    cond_hiddenFeatures=config.conditioning_transformer.hidden_features,
                    cond_dimFeedForward=config.conditioning_transformer.dim_feedforward_transformer,
                    cond_outFeatures=config.conditioning_transformer.out_features,
                    cond_nheadEncoder=config.conditioning_transformer.nhead_encoder,
                    cond_NoLayersEncoder=config.conditioning_transformer.no_layers_encoder,
                    cond_nheadDecoder=config.conditioning_transformer.nhead_decoder,
                    cond_NoLayersDecoder=config.conditioning_transformer.no_layers_decoder,
                    cond_NoDecoders=config.conditioning_transformer.no_decoders,
                    cond_aggregate=config.conditioning_transformer.aggregate,
                    cond_use_latent=config.conditioning_transformer.use_latent,
                    cond_out_features_latent=config.conditioning_transformer.out_features_latent,
                    cond_no_layers_decoder_latent=config.conditioning_transformer.no_layers_decoder_latent,   
        
                    flow_nfeatures=config.unfolding_flow.nfeatures,
                    flow_ncond=config.unfolding_flow.ncond, 
                    flow_ntransforms=config.unfolding_flow.ntransforms,
                    flow_hiddenMLP_NoLayers=config.unfolding_flow.hiddenMLP_NoLayers, 
                    flow_hiddenMLP_LayerDim=config.unfolding_flow.hiddenMLP_LayerDim,
                    flow_bins=config.unfolding_flow.bins,
                    flow_autoregressive=config.unfolding_flow.autoregressive,
                    flow_base=config.unfolding_flow.base,
                    flow_base_first_arg=config.unfolding_flow.base_first_arg,
                    flow_base_second_arg=config.unfolding_flow.base_second_arg,
                    flow_bound=config.unfolding_flow.bound,
                    randPerm=config.unfolding_flow.randPerm,
                    device=device,
                    dtype=dtype,
                    eps=config.training_params.eps)

    model.load_state_dict(torch.load(model_weights, map_location=torch.device("cpu"))["model_state_dict"])

    # Setting up DDP
    model = model.to(device)
    model.eval()
    
    B =  config.training_params.batch_size_validation

    # Datasets
    testLoader = DataLoader(
        test_dataset,
        batch_size=B,
        shuffle=False,
    )

    N_eval = len(test_dataset)
    if N_events == None:
        N_events = N_eval


    out_ps = torch.zeros((N_events, 10), dtype=torch.float64, device="cpu")
    out_partons_lab = torch.zeros((N_events, 4, 3), dtype=torch.float64, device="cpu") 
    out_partons_CM = torch.zeros((N_events, 4, 4), dtype=torch.float64, device="cpu") 
    out_boost = torch.zeros((N_events, 2), dtype=torch.float64, device="cpu") 
    out_flow_prob = torch.zeros((N_events,), dtype=torch.float64, device="cpu") 
    out_flow_samples = torch.zeros((N_events, N_samples, 10),dtype= torch.float64, device="cpu") 

    # training loop    
    iterator = iter(testLoader)
    max_iter = ceil(N_eval / B)
    for i in track(range(max_iter+1), "Evaluating"):
        with torch.no_grad():

            (logScaled_partons, logScaled_boost,
             logScaled_reco, mask_lepton_reco, 
             mask_jets, mask_met, 
             mask_boost_reco, data_boost_reco,
             ps_target, ps_target_scaled) = next(iterator)


            mask_recoParticles = torch.cat((mask_jets, mask_lepton_reco, mask_met), dim=1)
            if True : # Always remove true proveance TODO REMOVE THE HACK 
                logScaled_reco = logScaled_reco[:,:,:-1]


            # The provenance is remove in the model
            (data_regressed, data_regressed_cm, ps_regr,
                 logit_ps_regr, flow_cond_vector,
                 flow_logprob, mask_problematic)   = model(logScaled_reco,
                                                                          data_boost_reco,
                                                                          mask_recoParticles,
                                                                          mask_boost_reco,
                                                                          ps_target_scaled,
                                                                          disableGradTransformer=False,
                                                                          flow_eval="normalizing")
            ps_samples = model.flow(flow_cond_vector).sample((N_samples,))
            breakpoint()
            if i<max_iter:
                out_ps[i*B: (i+1)*B] = ps_regr.cpu()
                out_partons_lab[i*B:(i+1)*B] = torch.stack(data_regressed[:-1], dim=1).cpu()
                out_partons_CM[i*B:(i+1)*B] = data_regressed_cm.cpu()
                out_boost[i*B:(i+1)*B] = data_regressed[-1].cpu()

                out_flow_samples[i*B:(i+1)*B] = ps_samples.transpose(1,0).cpu()
                out_flow_prob[i*B:(i+1)*B] = flow_logprob.cpu()
            else:
                out_ps[i*B: ] = ps_regr.cpu()
                out_partons_lab[i*B:] = torch.stack(data_regressed[:-1], dim=1).cpu()
                out_partons_CM[i*B:] = data_regressed_cm.cpu()
                out_boost[i*B:] = data_regressed[-1].cpu()

                out_flow_samples[i*B:] = ps_samples.transpose(1,0).cpu()
                out_flow_prob[i*B:] = flow_logprob.cpu()

            if (i+1)*B > N_events:
                break

    
    with h5py.File(outputPath, 'a') as f:
        # Create a dataset in the file
        out_ps = f.create_dataset("ps", (N_events, 10 ), dtype='f', data=out_ps)
        out_partons_lab =  f.create_dataset("part_lab_ptetaphi", (N_events, 4, 3 ), dtype='f', data=out_partons_lab)
        out_partons_CM =  f.create_dataset("part_lab_Epxpypz", (N_events, 4, 4), dtype='f', data=out_partons_CM)
        out_boost =  f.create_dataset("boost", (N_events, 2), dtype='f', data=out_boost)
        out_flow_prob = f.create_dataset("logprob", (N_events,), dtype='f', data=out_flow_prob)
        out_flow_samples = f.create_dataset("flow_samples", (N_events, N_samples, 10), dtype='f', data=out_flow_samples)
        
              
    print('Evaluation finished!!')
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-config', type=str, required=True, help='path to config.yaml File')
    parser.add_argument('--output-path', type=str, required=True, help='Output directory')
    parser.add_argument('--on-GPU', action="store_true",  help='run on GPU boolean')
    parser.add_argument('--nsamples', type=int, required=True, help="Number of samples for event")
    parser.add_argument("--nevents", type=int,  help="Number of events to evaluate. ")
    parser.add_argument("--batch-size", type=int,  help="Batch size")
    parser.add_argument('--model-weights', type=str, help="model .pt file")
    args = parser.parse_args()
    
    path_to_conf = args.path_config
    on_GPU = args.on_GPU # by default run on CPU
    outputPath = args.output_path
    model_weights = os.path.dirname(path_to_conf)+ "/" + args.model_weights

    # Read config file in 'conf'
    with open(path_to_conf) as f:
        conf = OmegaConf.load(path_to_conf)
    
    print("Evaluating with cfg: \n", OmegaConf.to_yaml(conf))

    outputDir = os.path.dirname(outputPath)
    os.makedirs(outputDir, exist_ok=True)    

    if conf.training_params.dtype == "float32":
        dtype = torch.float32
    elif conf.training_params.dtype == "float64":
        dtype = torch.float64

    if args.batch_size:
        conf.training_params.batch_size_validation = args.batch_size

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    validation(device, conf, model_weights, outputPath, args.nsamples, args.nevents, dtype)
    
    print(f"Flow training finished succesfully! Version: {conf.version}")
    
    
    
    
    
    
    
    
    
