from memflow.unfolding_flow.unfolding_flow import UnfoldingFlow
from memflow.read_data.dataset_all import DatasetCombined
from memflow.unfolding_network.conditional_transformer import ConditioningTransformerLayer
import yaml
from yaml.loader import SafeLoader
import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import memflow.unfolding_flow.utils as utils
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch.multiprocessing as mp

import sys
import argparse
import os

def TrainingAndValidLoop(config, model, trainingLoader, validLoader):
    optimizer = optim.Adam(list(model.parameters()) , lr=config.training_params.lr)
    
    N_train = len(trainingLoader)
    N_valid = len(validLoader)
    

    name_dir = f'{os.getcwd()}/results/runs/{config.version}'
    writer = SummaryWriter(name_dir)
    
    for e in range(config.training_params.nepochs):
        
        sum_loss = 0.
    
        # training loop    
        print("Before training loop")
        for i, data in enumerate(trainingLoader):
            
            if (i % 100 == 0):
                print(i)

            optimizer.zero_grad()

            logp_g, detjac, cond_X = model(data)

            inf_mask = torch.isinf(logp_g)
            nonzeros = torch.count_nonzero(inf_mask)
            writer.add_scalar(f"Number_INF_flow_{e}", nonzeros.item(), i)

            logp_g = torch.nan_to_num(logp_g, posinf=20, neginf=-20)
            detjac = torch.nan_to_num(detjac, posinf=20, neginf=-20)

            detjac_mean = -detjac.nanmean()
            writer.add_scalar(f"detJacOnly_epoch_{e}", detjac_mean.item(), i)

            loss = -logp_g.mean()
            loss.backward()

            optimizer.step() 
            sum_loss += loss.item()

            writer.add_scalar(f"Loss_step_train_epoch_{e}", loss.item(), i)

        writer.add_scalar('Loss_epoch_train', sum_loss/N_train, e)
        valid_loss = 0

        # validation loop (don't update weights and gradients)
        print("Before validation loop")
        for i, data in enumerate(validLoader):

            logp_g, detjac, cond_X  = model(data)
            logp_g = torch.nan_to_num(logp_g, posinf=20, neginf=-20)
            loss =  -logp_g.mean()
            valid_loss += loss.item()

            writer.add_scalar(f"Loss_step_validation_epoch_{e}", loss.item(), i)

            if i == 0:

                (data_ps, data_ps_detjacinv, 
                mask_lepton, data_lepton, mask_jets,
                data_jets, mask_met, data_met,
                mask_boost_reco, data_boost_reco) =  data

                ps_new = model.flow(cond_X).sample((config.training_params.sampling_points,))

                data_ps_cpu = data_ps.detach().cpu()
                ps_new_cpu = ps_new.detach().cpu()

                for x in range(data_ps_cpu.size(1)):
                    fig, ax = plt.subplots()
                    h = ax.hist2d(data_ps_cpu[:,x].tile(config.training_params.sampling_points,1,1).flatten().numpy(),
                                  ps_new_cpu[:,:,x].flatten().numpy(),
                                  bins=50, range=((0, 1),(0, 1)))
                    fig.colorbar(h[3], ax=ax)
                    writer.add_figure(f"Validation_ramboentry_Plot_{x}", fig, e)

                    fig, ax = plt.subplots()
                    h = ax.hist(
                        (data_ps_cpu[:,x].tile(config.training_params.sampling_points,1,1) - ps_new_cpu[:,:,x]).flatten().numpy(),
                        bins=100)
                    writer.add_figure(f"Validation_ramboentry_Diff_{x}", fig, e)

        writer.add_scalar('Loss_epoch_val', valid_loss/N_valid, e)

        writer.close()
        
        resultsDir = f"{os.getcwd()}/results/logs/result_{config.name}_{config.version}"
        
        torch.save({
            'modelA_state_dict': model.state_dict(),
            'optimizerA_state_dict': optimizer.state_dict()
            }, resultsDir)
        
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-config', type=str, required=True, help='path to config.yaml File')
    parser.add_argument('--on-GPU', action="store_true",  help='run on GPU boolean')
    args = parser.parse_args()
    
    path_to_conf = args.path_config
    on_GPU = args.on_GPU # by default run on CPU

    # Read config file in 'conf'
    with open(path_to_conf) as f:
        conf = OmegaConf.load(path_to_conf)
        print(conf)
    
    print("Training with cfg: \n", OmegaConf.to_yaml(conf))
    
    if on_GPU:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')

    if (device == torch.device('cuda')):
        torch.cuda.empty_cache()
        env_var = os.environ.get("CUDA_VISIBLE_DEVICES")
        if env_var:
            actual_devices = env_var.split(",")
            actual_devices = [int(d) for d in actual_devices]
        else:
            actual_devices = list(range(torch.cuda.device_count()))
        print("Actual devices: ", actual_devices)
        
        
    # READ data
    data = DatasetCombined(conf.input_dataset, dev=device, dtype=torch.float64)
    
    # split data for training sample and validation sample
    train_subset, val_subset = torch.utils.data.random_split(
        data, [conf.training_params.traning_sample, conf.training_params.validation_sample],
        generator=torch.Generator().manual_seed(1))
    
    train_loader = DataLoader(dataset=train_subset, shuffle=True, batch_size=conf.training_params.batch_size_training)
    val_loader = DataLoader(dataset=val_subset, shuffle=True, batch_size=conf.training_params.batch_size_validation)
    
    # Initialize model
    model = UnfoldingFlow(no_jets=conf.input_shape.number_jets,
                    no_lept=conf.input_shape.number_lept,
                    jets_features=conf.input_shape.jets_features,
                    lepton_features=conf.input_shape.lepton_features, 
                    nfeatures_flow=conf.unfolding_flow.nfeatures,
                    ncond_flow=conf.unfolding_flow.ncond, 
                    ntransforms_flow=conf.unfolding_flow.ntransforms,
                    hidden_mlp_flow=conf.unfolding_flow.hidden_mlp, 
                    bins_flow=conf.unfolding_flow.bins,
                    autoregressive_flow=conf.unfolding_flow.autoregressive, 
                    out_features_cond=conf.conditioning_transformer.out_features, 
                    nhead_cond=conf.conditioning_transformer.nhead, 
                    no_layers_cond=conf.conditioning_transformer.no_layers,
                    dtype=torch.float64)
    
    # Copy model on GPU memory
    if (device == torch.device('cuda')):
        model = model.cuda()

        
    print(f"parameters total:{utils.count_parameters(model)}")
    
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
            TrainingAndValidLoop(conf, model, train_loader, val_loader)
        else:
            TrainingAndValidLoop(conf, model, train_loader, val_loader)
    else:
        TrainingAndValidLoop(conf, model, train_loader, val_loader)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
