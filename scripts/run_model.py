from memflow.unfolding_flow.unfolding_flow import UnfoldingFlow
from memflow.read_data.dataset_all import DatasetCombined
from memflow.unfolding_network.conditional_transformer import ConditioningTransformerLayer
from torch.optim.lr_scheduler import CosineAnnealingLR
from earlystop import EarlyStopper
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

from torch.cuda.amp import GradScaler
from torch import autocast
from memflow.unfolding_flow.mmd_loss import MMD

import glob
import sys
import argparse
import os

def TrainingAndValidLoop(config, model, trainingLoader, validLoader, outputDir, alternativeTr, disableGradTransformer):
    optimizer = optim.AdamW(list(model.parameters()) , lr=config.training_params.lr, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    N_train = len(trainingLoader)
    N_valid = len(validLoader)
    
    name_dir = f'{outputDir}/results_{config.unfolding_flow.base}_FirstArg:{config.unfolding_flow.base_first_arg}_Autoreg:{config.unfolding_flow.autoregressive}_NoTransf:{config.unfolding_flow.ntransforms}_NoBins:{config.unfolding_flow.bins}_DNN:{config.unfolding_flow.hiddenMLP_NoLayers}:{config.unfolding_flow.hiddenMLP_LayerDim}'
    modelName = f"{name_dir}/model_flow.pt"
    writer = SummaryWriter(name_dir)
    with open(f"{name_dir}/config_{config.name}_{config.version}.yaml", "w") as fo:
        fo.write(OmegaConf.to_yaml(config))

    ii = 0
    early_stopper = EarlyStopper(patience=config.training_params.nEpochsPatience, min_delta=0.001)
    torch.autograd.set_detect_anomaly(True)
    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler()

    sampling_Forward = config.training_params.sampling_Forward
    
    for e in range(config.training_params.nepochs):
        
        sum_loss = 0.
        if alternativeTr:
            if e % 2 == 0:
                sampling_Forward = False
            else:
                sampling_Forward = True
    
        # training loop    
        print("Before training loop")
        model.train()
        for i, data in enumerate(trainingLoader):
            
            ii += 1

            if (i % 100 == 0):
                print(i)

            optimizer.zero_grad()

            (PS_target,
            PS_rambo_detjacobian,
            logScaled_reco, mask_lepton_reco, 
            mask_jets, mask_met, 
            mask_boost_reco, data_boost_reco) = data

            mask0 = PS_target < 0
            mask1 = PS_target > 1
            if (mask0.any() or mask1.any()):
                print('PS target < 0 or > 1')
                exit(0)

            # Runs the forward pass with autocasting.
            with autocast(device_type='cuda', dtype=torch.float16):
            #if True:
                cond_X, PS_regressed = model(mask_jets=mask_jets, mask_lepton_reco=mask_lepton_reco,
                                            mask_met=mask_met, mask_boost_reco=mask_boost_reco,
                                            logScaled_reco=logScaled_reco, data_boost_reco=data_boost_reco, 
                                            device=device, noProv=config.noProv, eps=config.training_params.eps,
                                            order=config.training_params.order, disableGradTransformer=disableGradTransformer)

                detjac = PS_rambo_detjacobian.log()
                flow_prob = model.flow(PS_regressed).log_prob(PS_target)

                if sampling_Forward:

                    flow_sample = model.flow(PS_regressed).rsample()

                    sample_mask_all = (flow_sample>=0) & (flow_sample<=1)
                    sample_mask = torch.all(sample_mask_all, dim=1)

                    flow_sample = flow_sample[sample_mask]
                    PS_target_masked = PS_target[sample_mask]
                    
                    # kernel: kernel type such as "multiscale" or "rbf"
                    flow_loss = MMD(x=flow_sample, y=PS_target_masked, kernel='rbf', device=device)        
                else:
                    flow_loss = -flow_prob

                inf_mask = torch.isinf(flow_loss)
                nonzeros = torch.count_nonzero(inf_mask)
                writer.add_scalar(f"Number_INF_flow_{e}", nonzeros.item(), i)
                #logp_g = torch.nan_to_num(logp_g, posinf=20, neginf=-20)
                #detjac = torch.nan_to_num(detjac, posinf=20, neginf=-20)

                detjac_mean = -detjac.nanmean()
                
                if sampling_Forward:
                    loss = flow_loss
                else:
                    loss = flow_loss[torch.logical_not(inf_mask)].mean() # dont take infinities into consideration

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            #loss.backward()
            #optimizer.step()

            sum_loss += loss.item()

            writer.add_scalar(f"detJacOnly_epoch_step", detjac_mean.item(), i)
            if sampling_Forward:
                writer.add_scalar(f"Loss_step_train_epoch_step_SamplingDir", loss.item(), i)
            else:
                writer.add_scalar(f"Loss_step_train_epoch_step_Normalizing_dir", loss.item(), i)


        if sampling_Forward:
            writer.add_scalar(f"Loss_epoch_train_SamplingDir", sum_loss/N_train, e)
        else:
            writer.add_scalar(f"Loss_epoch_train_NormalizingDir", sum_loss/N_train, e)

        valid_loss = 0

        # validation loop (don't update weights and gradients)
        print("Before validation loop")
        model.eval()
        for i, data in enumerate(validLoader):

            (PS_target,
            PS_rambo_detjacobian,
            logScaled_reco, mask_lepton_reco, 
            mask_jets, mask_met, 
            mask_boost_reco, data_boost_reco) = data

            with torch.no_grad():

                cond_X, PS_regressed = model(mask_jets=mask_jets, mask_lepton_reco=mask_lepton_reco,
                                            mask_met=mask_met, mask_boost_reco=mask_boost_reco,
                                            logScaled_reco=logScaled_reco, data_boost_reco=data_boost_reco, 
                                            device=device, noProv=config.noProv, eps=config.training_params.eps,
                                            order=config.training_params.order, disableGradTransformer=disableGradTransformer)

                detjac = PS_rambo_detjacobian.log()
                flow_prob = model.flow(PS_regressed).log_prob(PS_target)

                if sampling_Forward:

                    flow_sample = model.flow(PS_regressed).sample()

                    sample_mask_all = (flow_sample>=0) & (flow_sample<=1)
                    sample_mask = torch.all(sample_mask_all, dim=1)

                    flow_sample = flow_sample[sample_mask]
                    PS_target_masked = PS_target[sample_mask]
                    
                    # kernel: kernel type such as "multiscale" or "rbf"
                    flow_loss = MMD(x=flow_sample, y=PS_target_masked, kernel='rbf', device=device)        
                else:
                    flow_loss = -flow_prob

                inf_mask = torch.isinf(flow_loss)
                nonzeros = torch.count_nonzero(inf_mask)
                detjac_mean = -detjac.nanmean()
                
                if sampling_Forward:
                    loss = flow_loss
                else:
                    loss = flow_loss[torch.logical_not(inf_mask)].mean() # dont take infinities into consideration
                
                valid_loss += loss.item()

                if i == 0:

                    N_samples = 100 # 100 x 1024 x 10
                    ps_new = model.flow(PS_regressed).sample((N_samples,))

                    data_ps_cpu = PS_target.detach().cpu()
                    ps_new_cpu = ps_new.detach().cpu()

                    for x in range(data_ps_cpu.size(1)):
                        fig, ax = plt.subplots()
                        h = ax.hist2d(data_ps_cpu[:,x].tile(N_samples,1,1).flatten().numpy(),
                                        ps_new_cpu[:,:,x].flatten().numpy(),
                                        bins=50, range=((-1.5, 1.5),(-1.5, 1.5)))
                        fig.colorbar(h[3], ax=ax)
                        writer.add_figure(f"CheckElemOutside_Plot_{x}", fig, e)

                        fig, ax = plt.subplots()
                        h = ax.hist2d(data_ps_cpu[:,x].tile(N_samples,1,1).flatten().numpy(),
                                    ps_new_cpu[:,:,x].flatten().numpy(),
                                    bins=50, range=((-0.1, 1.1),(-0.1, 1.1)))
                        fig.colorbar(h[3], ax=ax)
                        writer.add_figure(f"Validation_ramboentry_Plot_{x}", fig, e)

                        fig, ax = plt.subplots()
                        h = ax.hist(
                            (data_ps_cpu[:,x].tile(N_samples,1,1) - ps_new_cpu[:,:,x]).flatten().numpy(),
                            bins=100)
                        writer.add_figure(f"Validation_ramboentry_Diff_{x}", fig, e)

        valid_loss = valid_loss/N_valid
        if sampling_Forward:
            writer.add_scalar(f"Loss_epoch_val_SamplingDir", valid_loss, e)
        else:
            writer.add_scalar(f"Loss_epoch_val_NormalizingDir", valid_loss, e)

        if early_stopper.early_stop(valid_loss, model.state_dict(), optimizer.state_dict(), modelName):
            print(f"Model converges at epoch {e} !!!")         
            break

        scheduler.step() # reduce lr if the model is not improving anymore

    writer.close()
        
    print('Training finished!!')
        
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, required=True, help='path to config.yaml File')
    parser.add_argument('--output-dir', type=str, required=True, help='path to output directory')
    parser.add_argument('--on-GPU', action="store_true",  help='run on GPU boolean')
    parser.add_argument('--alternativeTr', action="store_true",  help='Use Alternative Training')
    parser.add_argument('--path-config', type=str, help='by default use the file config from the pretraining directory')
    parser.add_argument('--disable-grad-CondTransformer', action="store_true",  help='Propagate gradients for ConditionalTransformer')
    args = parser.parse_args()
    
    path_to_dir = args.model_dir
    output_dir = f'{args.model_dir}/{args.output_dir}'
    on_GPU = args.on_GPU # by default run on CPU
    alternativeTr = args.alternativeTr # by default do the training in a specific direction
    disableGradTransf = args.disable_grad_CondTransformer

    path_to_conf = glob.glob(f"{path_to_dir}/*.yaml")[0]
    path_to_model = glob.glob(f"{path_to_dir}/model*.pt")[0]

    if (args.path_config is not None):
        path_to_conf = args.path_config

    print(path_to_conf)
    # Read config file in 'conf'
    with open(path_to_conf) as f:
        conf = OmegaConf.load(path_to_conf)
        print(conf)
    
    print("Training with cfg: \n", OmegaConf.to_yaml(conf))

    if on_GPU:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu') 
        
    # READ data
    if (conf.cartesian):
        data = DatasetCombined(conf.input_dataset, dev=device, dtype=torch.float64,
                                reco_list=['scaledLogRecoParticlesCartesian', 'mask_lepton', 
                                            'mask_jets','mask_met',
                                            'mask_boost', 'data_boost'],
                                parton_list=['phasespace_intermediateParticles_onShell',
                                            'phasespace_rambo_detjacobian_onShell'])
    else:
        data = DatasetCombined(conf.input_dataset, dev=device, dtype=torch.float64,
                                reco_list=['scaledLogRecoParticles', 'mask_lepton', 
                                            'mask_jets','mask_met',
                                            'mask_boost', 'data_boost'],
                                parton_list=['phasespace_intermediateParticles_onShell',
                                            'phasespace_rambo_detjacobian_onShell'])
    
    # split data for training sample and validation sample
    train_subset, val_subset = torch.utils.data.random_split(
        data, [conf.training_params.training_sample, conf.training_params.validation_sample],
        generator=torch.Generator().manual_seed(1))
    
    train_loader = DataLoader(dataset=train_subset, shuffle=True, batch_size=conf.training_params.batch_size_training)
    val_loader = DataLoader(dataset=val_subset, shuffle=True, batch_size=conf.training_params.batch_size_validation)
    
    # Initialize model
    model = UnfoldingFlow(model_path=path_to_model,
                    read_CondTransf=True,
                    log_mean = conf.scaling_params.log_mean,
                    log_std = conf.scaling_params.log_std,
                    no_jets=conf.input_shape.number_jets,
                    no_lept=conf.input_shape.number_lept,
                    input_features=conf.input_shape.input_features,
                    cond_hiddenFeatures=conf.conditioning_transformer.hidden_features,
                    cond_dimFeedForward=conf.conditioning_transformer.dim_feedforward_transformer,
                    cond_outFeatures=conf.conditioning_transformer.out_features,
                    cond_nheadEncoder=conf.conditioning_transformer.nhead_encoder,
                    cond_NoLayersEncoder=conf.conditioning_transformer.no_layers_encoder,
                    cond_nheadDecoder=conf.conditioning_transformer.nhead_decoder,
                    cond_NoLayersDecoder=conf.conditioning_transformer.no_layers_decoder,
                    cond_NoDecoders=conf.conditioning_transformer.no_decoders,
                    cond_aggregate=conf.conditioning_transformer.aggregate,
                    flow_nfeatures=conf.unfolding_flow.nfeatures,
                    flow_ncond=conf.unfolding_flow.ncond, 
                    flow_ntransforms=conf.unfolding_flow.ntransforms,
                    flow_hiddenMLP_NoLayers=conf.unfolding_flow.hiddenMLP_NoLayers, 
                    flow_hiddenMLP_LayerDim=conf.unfolding_flow.hiddenMLP_LayerDim,
                    flow_bins=conf.unfolding_flow.bins,
                    flow_autoregressive=conf.unfolding_flow.autoregressive,
                    flow_base=conf.unfolding_flow.base,
                    flow_base_first_arg=conf.unfolding_flow.base_first_arg,
                    flow_base_second_arg=conf.unfolding_flow.base_second_arg,
                    flow_bound=conf.unfolding_flow.bound,
                    use_latent=conf.conditioning_transformer.use_latent,
                    device=device,
                    dtype=torch.float64)
    
    # Copy model on GPU memory
    if (device == torch.device('cuda')):
        model = model.cuda()
        
    print(f"parameters total:{utils.count_parameters(model)}")
    
    if (device == torch.device('cuda')):
        
        # TODO: split the data for multi-GPU processing
        #if len(actual_devices) > 1:
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
        TrainingAndValidLoop(conf, model, train_loader, val_loader, output_dir, alternativeTr, disableGradTransf)
        #else:
        #    TrainingAndValidLoop(conf, model, train_loader, val_loader, output_dir, alternativeTr, disableGradTransf)
    else:
        TrainingAndValidLoop(conf, model, train_loader, val_loader, output_dir, alternativeTr, disableGradTransf)
        
    print("Training finished succesfully!")
    
    
    
    
    
    
    
    
    
    
    
    
    

