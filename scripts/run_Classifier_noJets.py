from comet_ml import Experiment
#from comet_ml.integration.pytorch import log_model

import torch
from memflow.read_data.dataset_all import DatasetCombined
from memflow.classifier_nojets.Classifier_nojets import Classifier_nojets
from memflow.unfolding_flow.utils import *

import numpy as np
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.functional import normalize
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from math import floor

import mdmm
# from tensorboardX import SummaryWriter
from omegaconf import OmegaConf
import sys
import argparse
import os
from pynvml import *
import vector

from utils import alter_variables

from earlystop import EarlyStopper
from memflow.unfolding_flow.utils import Compute_ParticlesTensor

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.profiler import profile, record_function, ProfilerActivity

from random import randint
PI = torch.pi    

def ddp_setup(rank, world_size, port):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    import socket
    print("Setting up ddp for device: ", rank)
    os.environ["MASTER_ADDR"] = socket.gethostname()
    os.environ["MASTER_PORT"] = f"{port}"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train( device, name_dir, config,  outputDir, dtype,
           world_size=None, device_ids=None):
    # device is device when not distributed and rank when distributed
    print("START OF RANK:", device)
    if world_size is not None:
        ddp_setup(device, world_size, config.ddp_port)

    device_id = device_ids[device] if device_ids is not None else device
 
    modelName = f"{name_dir}/model_{config.name}_{config.version}.pt"

    #print("Loading datasets")
    train_dataset = DatasetCombined(config.input_dataset_train, dev=device,
                                    dtype=dtype, datasets=["partons_lab", "reco_lab"],
                                    reco_list_lab=['mask_jets'],
                                    parton_list_lab=['logScaled_data_higgs_t_tbar_ISR'])

    val_dataset = DatasetCombined(config.input_dataset_validation, dev=device,
                                  dtype=dtype, datasets=["partons_lab", "reco_lab"],
                                    reco_list_lab=['mask_jets'],
                                    parton_list_lab=['logScaled_data_higgs_t_tbar_ISR'])

    # Initialize model
    model = Classifier_nojets(hidden_features=config.input_shape.no_partonVars,
                               dim_feedforward_transformer=config.transformerClassifier.dim_feedforward,
                               nhead_encoder=config.transformerClassifier.nhead_encoder,
                               no_layers_encoder=config.transformerClassifier.no_encoder_layers,
                               dtype=dtype)
         
    

    # Experiment logging
    if device == 0 or world_size is None:
        # Loading comet_ai logging
        exp = Experiment(
            api_key=config.comet_token,
            project_name="Classifier_NoJets",
            workspace="antoniopetre",
            auto_output_logging = "simple",
            # disabled=True
        )
        exp.add_tags([config.name, config.version])
        exp.log_parameters(config.training_params)
        exp.log_parameters(config.transformerClassifier)
        exp.log_parameters({"model_param_tot":count_parameters(model)})
    else:
        exp = None

    # Setting up DDP
    model = model.to(device)

    if world_size is not None:
        ddp_model = DDP(
            model,
            device_ids=[device],
            output_device=device,
            # find_unused_parameters=True,
        )
        #print(ddp_model)
        model = ddp_model.module
    else:
        ddp_model = model


    # Datasets
    trainingLoader = DataLoader(
        train_dataset,
        batch_size= config.training_params.batch_size_training,
        shuffle=False if world_size is not None else True,
        sampler=DistributedSampler(train_dataset) if world_size is not None else None,
    )
    validLoader = DataLoader(
        val_dataset,
        batch_size=config.training_params.batch_size_training,
        shuffle=False,        
    )

    loss = torch.nn.CrossEntropyLoss(weight=None, ignore_index=-1, reduction='mean', label_smoothing=0.0)
    optimizer = optim.RAdam(list(model.parameters()) , lr=config.training_params.lr)
    
    scheduler_type = config.training_params.scheduler
    
    if scheduler_type == "cosine_scheduler":
        scheduler = CosineAnnealingLR(optimizer,
                                        T_max=config.training_params.cosine_scheduler.Tmax,
                                        eta_min=config.training_params.cosine_scheduler.eta_min)
    elif scheduler_type == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                factor=config.training_params.reduce_on_plateau.factor,
                                                                patience=config.training_params.reduce_on_plateau.patience,
                                                                threshold=config.training_params.reduce_on_plateau.threshold,
                                                                min_lr=config.training_params.reduce_on_plateau.get("min_lr", 1e-7),
                                                                verbose=True)
    elif scheduler_type == "cyclic_lr":
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                        base_lr= config.training_params.cyclic_lr.base_lr,
                                                        max_lr= config.training_params.cyclic_lr.max_lr,
                                                        step_size_up=config.training_params.cyclic_lr.step_size_up,
                                                        step_size_down=None,
                                                        cycle_momentum=False,
                                                        gamma=config.training_params.cyclic_lr.gamma,
                                                        mode=config.training_params.cyclic_lr.mode,
                                                        verbose=False)
        
    
    
    early_stopper = EarlyStopper(patience=config.training_params.nEpochsPatience, min_delta=0.0001)

    ii = 0
    ii_valid = 0

    for e in range(config.training_params.nepochs):
        
        N_train = 0
        N_valid = 0
        if world_size is not None:
            print(
                f"[GPU{device_id}] | Rank {device} | Epoch {e} | Batchsize: {config.training_params.batch_size_training*len(device_ids)} | Steps: {len(trainingLoader)}"
            )
            trainingLoader.sampler.set_epoch(e)
            
        sum_loss = 0.
    
        # training loop    
        print("Before training loop")
        ddp_model.train()

        for i, data_batch in enumerate(trainingLoader):
            N_train += 1
            ii+=1

            optimizer.zero_grad()

            (logScaled_data_higgs_t_tbar_ISR,
             mask_jets) = data_batch
            
             # need to substract 4 because classes are 0...N-1 not 4 ... N+3
            no_objects = torch.count_nonzero(mask_jets, dim=1) + 2 - 4 # add lepton/met
            prediction = model(logScaled_data_higgs_t_tbar_ISR)

            loss_main = loss(prediction, no_objects)

            loss_main.backward()
            optimizer.step()
            sum_loss += loss_main.item()

            if scheduler_type == "cyclic_lr": #cycle each step
                scheduler.step()

            with torch.no_grad():
                if exp is not None and device==0 or world_size is None:
                    if i % config.training_params.interval_logging_steps == 0:
                        exp.log_metric('loss_step', loss_main, step=ii)
                        exp.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=ii)

        ### END of training 
        if exp is not None and device==0 or world_size is None:
            exp.log_metric("full_loss_epoch_training", sum_loss/N_train, epoch=e)

        total_valid_loss = 0.
        
        # validation loop (don't update weights and gradients)
        print("Before validation loop")
        ddp_model.eval()
        
        for i, data_batch in enumerate(validLoader):
            N_valid += 1
            ii_valid += 1
            # Move data to device
            with torch.no_grad():

                (logScaled_data_higgs_t_tbar_ISR,
                 mask_jets) = data_batch
                
                # need to substract 4 because classes are 0...N-1 not 4 ... N+3
                no_objects = torch.count_nonzero(mask_jets, dim=1) + 2 - 4 # add lepton/met
                prediction = model(logScaled_data_higgs_t_tbar_ISR)
        
                loss_main = loss(prediction, no_objects)
                total_valid_loss += loss_main.item()

                if i == 0:
                    # iau cu argmax nr de jeturi prezis si vad comparatia
                    # ma uit la histograma 1d si la diferenta
                    # si fac o histograma 2d unde ma uit la diferenta vs target no jets
                    noJets_predicted = torch.argmax(prediction, dim=1) + 4
                    noJets_target = no_objects + 4
                    
                    # Check 1d distrib of noJets_predicted and true_noJets
                    fig, ax = plt.subplots(figsize=(7,6), dpi=100)
                    ax.hist(noJets_predicted.detach().cpu().numpy(), range=(4,19), bins=15, histtype='step', label='predicted', color='r', stacked=False, fill=False)
                    ax.hist(noJets_target.detach().cpu().numpy(), range=(4,19), bins=15, histtype='step', label='target', color='b', stacked=False, fill=False)
                    plt.legend()
                    ax.set_xlabel('noJets')
                    exp.log_figure(f"validation_figure_1", fig, step=e)
                    
                    # Check 1d distrib of diff between noJets_predicted and true_noJets
                    fig, ax = plt.subplots(figsize=(7,6), dpi=100)
                    ax.hist((noJets_predicted - noJets_target).detach().cpu().numpy(), range=(-16,16), bins=32, histtype='step', color='b', stacked=False, fill=False)
                    ax.set_xlabel('predicted - target (noJets)')
                    exp.log_figure(f"validation_figure_2", fig, step=e)
                    
                    # Check 2d distrib difference vs target
                    fig, ax = plt.subplots(figsize=(7,6), dpi=100)
                    h = ax.hist2d(noJets_target.detach().cpu().numpy(),
                                  (noJets_predicted - noJets_target).detach().cpu().numpy(),
                                  bins=[15,32],
                                  range=[(4,19),(-16,16)], cmin=1)
                    fig.colorbar(h[3], ax=ax)
                    ax.set_ylabel('predicted - target (noJets)')
                    ax.set_xlabel('target (noJets)')
                    exp.log_figure(f"validation_figure_3", fig, step=e)
                    

        if exp is not None and device==0 or world_size is None:
            exp.log_metric("total_valid_loss_epoch", total_valid_loss/N_valid, epoch=e)
            
            
        if device == 0 or world_size is None:
            if early_stopper.early_stop(total_valid_loss/N_valid,
                                    model.state_dict(), optimizer.state_dict(), modelName, exp):
                print(f"Model converges at epoch {e} !!!")         
                break

        # Step the scheduler at the end of the val
        if scheduler_type == "cosine_scheduler":
            after_N_epochs = config.training_params.cosine_scheduler.get("after_N_epochs", 0)
            if e > after_N_epochs:
                scheduler.step()
                
        elif scheduler_type == "reduce_on_plateau":
            # Step the scheduler at the end of the val
            scheduler.step(total_valid_loss/N_valid)

    # exp_log.end()
    destroy_process_group()
    print('preTraining finished!!')
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-config', type=str, required=True, help='path to config.yaml File')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--on-GPU', action="store_true",  help='run on GPU boolean')
    parser.add_argument('--distributed', action="store_true")
    args = parser.parse_args()
    
    path_to_conf = args.path_config
    on_GPU = args.on_GPU # by default run on CPU
    outputDir = args.output_dir

    # Read config file in 'conf'
    with open(path_to_conf) as f:
        conf = OmegaConf.load(path_to_conf)
    
    print("Training with cfg: \n", OmegaConf.to_yaml(conf))

    env_var = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_var:
        actual_devices = env_var.split(",")
    else:
        actual_devices = list(range(torch.cuda.device_count()))
    print("Actual devices: ", actual_devices)
    world_size = len(actual_devices)

    outputDir = os.path.abspath(outputDir)
    name_dir = f'{outputDir}/Classifier_noJets_{conf.name}_{conf.version}_dim_feedforward_transformer_{conf.transformerClassifier.dim_feedforward}_nhead_{conf.transformerClassifier.nhead_encoder}_numLayers_{conf.transformerClassifier.no_encoder_layers}'
    

    os.makedirs(name_dir, exist_ok=True)
    
    with open(f"{name_dir}/config_{conf.name}_{conf.version}.yaml", "w") as fo:
        fo.write(OmegaConf.to_yaml(conf)) 

    if conf.training_params.dtype == "float32":
        dtype = torch.float32
    elif conf.training_params.dtype == "float64":
        dtype = torch.float64
        
    if len(actual_devices) > 1 and args.distributed:
        # make a dictionary with k: rank, v: actual device
        dev_dct = {i: actual_devices[i] for i in range(world_size)}
        print(f"Devices dict: {dev_dct}")
        mp.spawn(
            train,
            args=(name_dir, conf,  outputDir, dtype,
                    world_size, dev_dct),
            nprocs=world_size,
            # join=True
        )
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        train(device, name_dir, conf,  outputDir, dtype)
    
    print(f"Classifier no_jets training finished succesfully! Version: {conf.version}")
    
    
    
    
    
    
    
    
    
