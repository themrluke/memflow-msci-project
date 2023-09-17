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
import os
from pynvml import *
import glob

def UnscaleTensor(config, model, dataLoader, outputDir, batch_size):
            
    outputDir = os.path.abspath(outputDir)
            
    total_sample = config.training_params.training_sample + config.training_params.validation_sample

    N = len(dataLoader)

    unscaledRegressedPartonsTensor = torch.empty((total_sample,
                                            config.conditioning_transformer.no_decoders,
                                            config.conditioning_transformer.out_features))

    log_mean = torch.tensor(config.scaling_params.log_mean, device=device)
    log_std = torch.tensor(config.scaling_params.log_std, device=device)
                
    for i, data in enumerate(dataLoader):

        with torch.no_grad():
            if (i % 100 == 0):
                print(i)
                
            (logScaled_reco, mask_lepton_reco, 
            mask_jets, mask_met, 
            mask_boost_reco, data_boost_reco) = data
                
            mask_recoParticles = torch.cat((mask_jets, mask_lepton_reco, mask_met), dim=1)

            # remove prov
            if (config.noProv):
                logScaled_reco = logScaled_reco[:,:,:-1]

            out = model(logScaled_reco, data_boost_reco, mask_recoParticles, mask_boost_reco)

            if config.conditioning_transformer.use_latent:
                no_particles = len(out) - 1
            else:
                no_particles = len(out)

            for particle in range(no_particles):
                if out[particle].shape[0] == batch_size:
                    unscaledRegressedPartonsTensor[i*batch_size:(i+1)*batch_size,particle,:] = out[particle]
                else:
                    unscaledRegressedPartonsTensor[i*batch_size:,particle,:] = out[particle]

    
    torch.save((unscaledRegressedPartonsTensor), f'{outputDir}/unscaledRegressedPartonsTensor.pt')



        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, required=True, help='path to model directory')
    parser.add_argument('--on-GPU', action="store_true",  help='run on GPU boolean')
    args = parser.parse_args()
    
    if(args.model_dir[-1] != '/'):
        raise("--model-dir argument must be directory, finishing in `/`")
    else:
        args.model_dir = args.model_dir[:len(args.model_dir)-1] # remove '/' from input 

    path_to_dir = args.model_dir
    path_to_conf = glob.glob(f"{path_to_dir}/*.yaml")[0]
    model_path = glob.glob(f"{path_to_dir}/model*.pt")[0]
    
    # Read config file in 'conf'
    with open(path_to_conf) as f:
        conf = OmegaConf.load(path_to_conf)

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
        data = DatasetCombined(conf.input_dataset, dev=device, dtype=torch.float64,
                                reco_list=['scaledLogRecoParticlesCartesian', 'mask_lepton', 
                                            'mask_jets','mask_met',
                                            'mask_boost', 'data_boost'],
                                parton_list=[])
    else:
        data = DatasetCombined(conf.input_dataset, dev=device, dtype=torch.float64,
                                reco_list=['scaledLogRecoParticles', 'mask_lepton', 
                                            'mask_jets','mask_met',
                                            'mask_boost', 'data_boost'],
                                parton_list=[])
    
    batch_size = 2048
    data_loader = DataLoader(dataset=data, shuffle=False, batch_size=batch_size)

    if (device == torch.device('cuda')):
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(0)
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
                                    dtype=torch.float64)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()
    
    if (device == torch.device('cuda')):
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(0)
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
        model = model.cuda()

    print(f"parameters total:{count_parameters(model)}")
    if (device == torch.device('cuda')):
        
        # TODO: split the data for multi-GPU processing
        if len(actual_devices) > 1:
            #world_size = torch.cuda.device_count()
            # make a dictionary with k: rank, v: actual device
            #dev_dct = {i: actual_devices[i] for i in range(world_size)}
            #print(f"Devices dict: {dev_dct}")
            #mp.spawn(
            #    TrainingAndValidLoop,
            #    args=(conf, model, train_loader, val_loader, world_size),
            #    nprocs=world_size,
            #    join=True,
            #)
            UnscaleTensor(conf, model, data_loader, outputDir, batch_size)
        else:
            UnscaleTensor(conf, model, data_loader, outputDir, batch_size)
    else:
        UnscaleTensor(conf, model, data_loader, outputDir, batch_size)
        
    
    print("Unscale tensor finished!")
    
    
    
    
    
    
    
    
    
    
    
    

