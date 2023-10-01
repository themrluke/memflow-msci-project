from memflow.read_data.dataset_all import DatasetCombined
from memflow.unfolding_network.conditional_transformer import ConditioningTransformerLayer
from memflow.unfolding_flow.utils import *

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.functional import normalize
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from tensorboardX import SummaryWriter
from omegaconf import OmegaConf
import sys
import argparse
from rich.progress import track
import os
from pynvml import *
import glob

def UnscaleTensor(config, model, dataLoader, Nsamples, outputDir, batch_size, device,
                  log_mean_parton, log_std_parton,
                  log_mean_boost, log_std_boost):
            
    outputDir = os.path.abspath(outputDir)
            
    output  = torch.empty((Nsamples, 5,4 ), device="cpu")


    iterator = iter(dataLoader)
    for i in track(range(len(dataLoader)), "Evaluating"):
        data = next(iterator)
        with torch.no_grad():                
            (logScaled_reco, mask_lepton_reco, 
            mask_jets, mask_met, 
            mask_boost_reco, data_boost_reco) = data

            #print(logScaled_reco[0])
                
            mask_recoParticles = torch.cat((mask_jets, mask_lepton_reco, mask_met), dim=1)

            # remove prov
            if (config.noProv):
                logScaled_reco = logScaled_reco[:,:,:-1]

            #print(mask_recoParticles.shape)
            #print(logScaled_reco.shape)

            out = model(logScaled_reco, data_boost_reco, mask_recoParticles, mask_boost_reco)

            #print(out)
            # if config.conditioning_transformer.use_latent:
            #     no_particles = len(out) - 1
            # else:
            #     no_particles = len(out)


            data_regressed, boost_regressed = Compute_ParticlesTensor.get_HttISR_fromlab(out, log_mean_parton,
                                  log_std_parton,
                                  log_mean_boost, log_std_boost,
                                device, cartesian=True, eps=1e-5)


            if out[0].shape[0] == batch_size:
                output[i*batch_size:(i+1)*batch_size] = torch.cat((data_regressed, boost_regressed.unsqueeze(1)), dim=1).cpu()
            else:
                output[i*batch_size:] = torch.cat((data_regressed, boost_regressed.unsqueeze(1)), dim=1).cpu()

                    

    
    torch.save((output), f'{outputDir}/unscaledRegressedPartonsTensor.pt')



        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, required=True, help='path to model directory')
    parser.add_argument('--conf', type=str, required=True, help='path config')
    parser.add_argument('--on-GPU', action="store_true",  help='run on GPU boolean')
    parser.add_argument('--data-validation', type=str, required=False, help="Validation dataset overwrite")
    args = parser.parse_args()
    
    if(args.model_dir[-1] != '/'):
        raise("--model-dir argument must be directory, finishing in `/`")
    else:
        args.model_dir = args.model_dir[:len(args.model_dir)-1] # remove '/' from input 

    path_to_dir = args.model_dir
    path_to_conf = f"{args.model_dir}/{args.conf}"
    model_path = glob.glob(f"{path_to_dir}/model*.pt")[0]
    
    # Read config file in 'conf'
    conf = OmegaConf.load(path_to_conf)
    if args.data_validation:
        conf.input_dataset_validation = args.data_validation

    on_GPU = args.on_GPU # by default run on CPU
    outputDir = args.model_dir
    
    if on_GPU:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')

    if (device == torch.device('cuda')):
        env_var = os.environ.get("CUDA_VISIBLE_DEVICES")
        if env_var:
            actual_devices = env_var.split(",")
            actual_devices = [int(d) for d in actual_devices]
        else:
            actual_devices = list(range(torch.cuda.device_count()))
        print("Actual devices: ", actual_devices)

    if (device == torch.device('cuda')):

        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
        print('')
    
    # READ data
    if (conf.cartesian):
        data = DatasetCombined(conf.input_dataset_validation, dev=device, dtype=torch.float64,
                               datasets=["partons_lab", "reco_lab"],
                                reco_list_lab=['scaledLogRecoParticlesCartesian', 'mask_lepton', 
                                            'mask_jets','mask_met',
                                            'mask_boost', 'scaledLogBoost'
                                          ],
                                parton_list_lab=[ 'mean_log_data_higgs_t_tbar_ISR',
                                             'std_log_data_higgs_t_tbar_ISR',
                                             'mean_log_data_boost',
                                             'std_log_data_boost'])
    else:
        data = DatasetCombined(conf.input_dataset_validation, dev=device, dtype=torch.float64,
                               datasets=["partons_lab", "reco_lab"],
                                reco_list_lab=['scaledLogRecoParticles', 'mask_lepton', 
                                            'mask_jets','mask_met',
                                            'mask_boost', 'scaledLogBoost'
                                          ],
                                parton_list_lab=[ 'mean_log_data_higgs_t_tbar_ISR',
                                             'std_log_data_higgs_t_tbar_ISR',
                                             'mean_log_data_boost',
                                             'std_log_data_boost'])

    print(f"Working with {len(data)} samples")
    batch_size = 2048
    data_loader = DataLoader(dataset=data, shuffle=False, batch_size=batch_size)

    if (device == torch.device('cuda')):
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(actual_devices[0])
        # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
        info = nvmlDeviceGetMemoryInfo(h)
        print(f'\ntotal    : {info.total}')
        print(f'free     : {info.free}')
        print(f'used     : {info.used}')
        nvmlShutdown()

        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
        print('')

    # Initialize model
    model = ConditioningTransformerLayer(no_jets = conf.input_shape.number_jets,
                                    no_lept = conf.input_shape.number_lept,
                                    input_features=conf.input_shape.input_features, 
                                    hidden_features=conf.conditioning_transformer.hidden_features,
                                    dim_feedforward_transformer= conf.conditioning_transformer.dim_feedforward_transformer,
                                    out_features=conf.conditioning_transformer.out_features,
                                    nhead_encoder=conf.conditioning_transformer.nhead_encoder,
                                    no_layers_encoder=conf.conditioning_transformer.no_layers_encoder,
                                    nhead_decoder=conf.conditioning_transformer.nhead_decoder,
                                    no_layers_decoder=conf.conditioning_transformer.no_layers_decoder,
                                    no_decoders=conf.conditioning_transformer.no_decoders,
                                    aggregate=conf.conditioning_transformer.aggregate,
                                    use_latent=conf.conditioning_transformer.use_latent,
                                    out_features_latent=conf.conditioning_transformer.out_features_latent,
                                    no_layers_decoder_latent=conf.conditioning_transformer.no_layers_decoder_latent,   
                                    dtype=torch.float64)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    print("Model weight evaluated")
    
    if (device == torch.device('cuda')):
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(actual_devices[0])
        # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
        info = nvmlDeviceGetMemoryInfo(h)
        print(f'\ntotal    : {info.total}')
        print(f'free     : {info.free}')
        print(f'used     : {info.used}')
        nvmlShutdown()

        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
        print('')

    # Copy model on GPU memory
    if (device == torch.device('cuda')):
        if len(actual_devices)>1:
            model = torch.nn.DataParallel(model)

        model = model.to(device)

    print(f"parameters total:{count_parameters(model)}")
    model.eval()
    UnscaleTensor(conf, model, data_loader, len(data), outputDir, batch_size, device,
                  data.partons_lab.mean_log_data_higgs_t_tbar_ISR,
                  data.partons_lab.std_log_data_higgs_t_tbar_ISR,
                             data.partons_lab.mean_log_data_boost,
                             data.partons_lab.std_log_data_boost)
        
    
    print("Unscale tensor finished!")
    
    
    
    
    
    
    
    
    
    
    
    

