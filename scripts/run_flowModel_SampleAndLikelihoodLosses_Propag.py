from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
from random import randint

from datetime import datetime

import torch
from memflow.read_data.dataset_all import DatasetCombined

from memflow.unfolding_flow.unfolding_flow_v2_onlyPropag import UnfoldingFlow_v2_onlyPropag
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
from math import floor

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

from memflow.phasespace.utils import *

from earlystop import EarlyStopper
from memflow.unfolding_flow.utils import Compute_ParticlesTensor

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.profiler import profile, record_function, ProfilerActivity

from random import randint
PI = torch.pi

#torch.autograd.set_detect_anomaly(True) # todo

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

def sinusoidal_positional_embedding(token_sequence_size, token_embedding_dim, device, n=10000.0):

    if token_embedding_dim % 2 != 0:
        raise ValueError("Sinusoidal positional embedding cannot apply to odd token embedding dim (got dim={:d})".format(token_embedding_dim))

    T = token_sequence_size
    d = token_embedding_dim #d_model=head_num*d_k, not d_q, d_k, d_v

    positions = torch.arange(0, T).unsqueeze_(1)
    embeddings = torch.zeros(T, d, device=device)

    denominators = torch.pow(n, 2*torch.arange(0, d//2)/d) # 10000^(2i/d_model), i is the index of embedding
    embeddings[:, 0::2] = torch.sin(positions/denominators) # sin(pos/10000^(2i/d_model))
    embeddings[:, 1::2] = torch.cos(positions/denominators) # cos(pos/10000^(2i/d_model))

    return embeddings

def loss_fn_periodic(inp, target, loss_fn, device, dtype):

    deltaPhi = torch.abs(target - inp)
    loss_total_1 = loss_fn(deltaPhi, torch.zeros(deltaPhi.shape, device=device, dtype=dtype))
    loss_total_2 = loss_fn(2*np.pi - deltaPhi, torch.zeros(deltaPhi.shape, device=device, dtype=dtype))
    loss_total = torch.minimum(loss_total_1, loss_total_2)
    
    return loss_total # + overflow_delta*0.5

def compute_regr_losses(MMD_input, MMD_target, boost_input, boost_target,
                        cartesian, loss_fn,
                        device, dtype, split=False):
    
    loss_pt = 4*loss_fn(MMD_input[...,0], MMD_target[...,0]) # reduction = none
    loss_eta = 2*loss_fn(MMD_input[...,1], MMD_target[...,1])
    loss_phi = 5*loss_fn_periodic(MMD_input[...,2], MMD_target[...,2], loss_fn, device, dtype)

    loss_boost_E = loss_fn(boost_input[...,0], boost_target[...,0])
    loss_boost_pz = loss_fn(boost_input[...,1], boost_target[...,1])

    loss_Event = torch.cat((loss_pt, loss_eta, loss_phi, loss_boost_E[:,None], loss_boost_pz[:,None]), dim=1)

    if split:
        loss_higgs = torch.sum(loss_Event[:,[0,3,6]], dim=0) # pt eta phi
        loss_thad = torch.sum(loss_Event[:,[1,4,7]], dim=0) # pt eta phi
        loss_tlep = torch.sum(loss_Event[:,[2,5,8]], dim=0) # pt eta phi
        loss_boost = torch.sum(loss_Event[:,[9,10]], dim=0) # E pz
        return loss_higgs.mean() / MMD_input.shape[0], \
                loss_thad.mean() / MMD_input.shape[0], \
                loss_tlep.mean() / MMD_input.shape[0], \
                loss_boost.mean() / MMD_input.shape[0]
    else:
        loss = torch.sum(loss_Event, dim=1).mean()
        return loss

def compute_mmd_loss(mmd_input, mmd_target, kernel, device, dtype, total=False, split=False):
    mmds = []
    for particle in range(len(mmd_input)):
        for feature in range(mmd_input[particle].shape[-1]):
            mmds.append(MMD(mmd_input[particle][...,feature:feature+1], mmd_target[particle][...,feature:feature+1], kernel, device, dtype))
    # total MMD
    if total:
        mmds.append(MMD(torch.cat(mmd_input, dim=1), torch.cat(mmd_target, dim=1), kernel, device, dtype))

    if split:
        return mmds
    else:
        return sum(mmds)/len(mmds)
    

def train( device, name_dir, config,  outputDir, dtype,
           world_size=None, device_ids=None, path_regression=None, disable_grad_conditioning=True):
    # device is device when not distributed and rank when distributed
    print("START OF RANK:", device)
    if world_size is not None:
        ddp_setup(device, world_size, config.ddp_port)

    device_id = device_ids[device] if device_ids is not None else device

    train_dataset = DatasetCombined(config.input_dataset_train, dev=device, new_higgs=True,
                                    dtype=dtype, datasets=['partons_lab', 'reco_lab', 'partons_CM'],
                           reco_list_lab=['scaledLogReco_sortedBySpanet',
                                          'mask_scaledLogReco_sortedBySpanet',
                                          'mask_boost', 'scaledLogBoost'],
                           parton_list_cm=['phasespace_intermediateParticles_onShell_logit_scaled'],
                                           #'phasespace_rambo_detjacobian_onShell'],
                           parton_list_lab=['logScaled_data_higgs_t_tbar_ISR',
                                           'logScaled_data_boost'])

    val_dataset = DatasetCombined(config.input_dataset_validation,dev=device, new_higgs=True,
                                  dtype=dtype, datasets=['partons_lab', 'reco_lab', 'partons_CM'],
                           reco_list_lab=['scaledLogReco_sortedBySpanet',
                                          'mask_scaledLogReco_sortedBySpanet',
                                          'mask_boost', 'scaledLogBoost'],
                           parton_list_cm=['phasespace_intermediateParticles_onShell_logit_scaled'],
                                          # 'phasespace_rambo_detjacobian_onShell'],
                           parton_list_lab=['logScaled_data_higgs_t_tbar_ISR',
                                           'logScaled_data_boost'])

    no_recoObjs = train_dataset.reco_lab.scaledLogReco_sortedBySpanet.shape[1]

    log_mean_reco = train_dataset.reco_lab.meanRecoParticles
    log_std_reco = train_dataset.reco_lab.stdRecoParticles
    log_mean_parton_Hthad = train_dataset.partons_lab.mean_log_data_higgs_t_tbar_ISR
    log_std_parton_Hthad = train_dataset.partons_lab.std_log_data_higgs_t_tbar_ISR
    log_mean_boost_parton = train_dataset.partons_lab.mean_log_data_boost
    log_std_boost_parton = train_dataset.partons_lab.std_log_data_boost
    mean_ps = train_dataset.partons_CM.mean_phasespace_intermediateParticles_onShell_logit
    scale_ps = train_dataset.partons_CM.std_phasespace_intermediateParticles_onShell_logit
    
    if device == torch.device('cuda'):
        log_mean_reco = log_mean_reco.cuda()
        log_std_reco = log_std_reco.cuda()

        log_mean_parton_Hthad = log_mean_parton_Hthad.cuda()
        log_std_parton_Hthad = log_std_parton_Hthad.cuda()
        log_mean_boost_parton = log_mean_boost_parton.cuda()
        log_std_boost_parton = log_std_boost_parton.cuda()

        mean_ps = mean_ps.cuda()
        scale_ps = scale_ps.cuda()


    model = UnfoldingFlow_v2_onlyPropag(scaling_partons_CM_ps=[mean_ps, scale_ps],

                                 regression_hidden_features=config.conditioning_transformer.hidden_features,
                                 regression_DNN_input=config.conditioning_transformer.hidden_features + 1,
                                 regression_dim_feedforward=config.conditioning_transformer.dim_feedforward_transformer,
                                 regression_nhead_encoder=config.conditioning_transformer.nhead_encoder,
                                 regression_noLayers_encoder=config.conditioning_transformer.no_layers_encoder,
                                 regression_noLayers_decoder=config.conditioning_transformer.no_layers_decoder,
                                 regression_DNN_layers=config.conditioning_transformer.DNN_layers,
                                 regression_DNN_nodes=config.conditioning_transformer.DNN_nodes,
                                 regression_aggregate=config.conditioning_transformer.aggregate,
                                 regression_atanh=True,
                                 regression_angles_CM=True,
                                 
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
                                 randPerm=True,

                                 DNN_condition=False,
                                 DNN_layers=2,
                                 DNN_dim=256,
                                 DNN_output_dim=3,
                                 
                                 device=device,
                                 dtype=dtype,
                                 pretrained_model=path_regression,
                                 #load_conditioning_model=False) #todo
                                 load_conditioning_model=True)
    
    
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

    
    modelName = f"{name_dir}/model_{config.name}_{config.version}.pt"

    if device == 0 or world_size is None:
        # Loading comet_ai logging
        exp = Experiment(
            api_key=config.comet_token,
            project_name="memflow",
            workspace="antoniopetre",
            auto_output_logging = "simple",
            # disabled=True
        )
        exp.add_tags([config.name,config.version, 'Unfolding Flow', 'Propag', 'likelihood+Samples',
                     f'autoreg={config.unfolding_flow.autoregressive}', f'freeze_pretrain={disable_grad_conditioning}'])
        exp.log_parameters({"model_param_tot": count_parameters(model)})
        exp.log_parameters(config)
        
        exp.set_name(f"UnfoldingFlow_Propag_SampleLikelihoodLosses_freeze:{disable_grad_conditioning}_{randint(0, 1000)}")
    else:
        exp = None

    # Datasets
    trainingLoader = DataLoader(
        train_dataset,
        batch_size= config.training_params.batch_size_training,
        shuffle=False if world_size is not None else True,
        sampler=DistributedSampler(train_dataset) if world_size is not None else None,
    )
    validLoader = DataLoader(
        val_dataset,
        batch_size=config.training_params.batch_size_validation,
        shuffle=False,        
    )
        
    if disable_grad_conditioning:
        
        constraints = []
        # regression huber constraint
        constraints.append(mdmm.MaxConstraint(
                                            compute_regr_losses,
                                            1000.0, # to be modified based on the regression
                                            scale=config.MDMM.scale_mmd,
                                            damping=config.MDMM.dumping_mmd,
                                            ))
        
        # regression MMD constraint
        for i in range(4):
            for j in range(3):
                if i == 3 and j == 2:
                    break
                constraints.append(mdmm.MaxConstraint(
                                            compute_mmd_loss,
                                            1000.0, # to be modified based on the regression
                                            scale=config.MDMM.scale_mmd,
                                            damping=config.MDMM.dumping_mmd,
                                            ))


        # sampled huber constraint
        constraints.append(mdmm.MaxConstraint(
                                            compute_regr_losses,
                                            config.MDMM.max_sampled_huber, # to be modified based on the regression
                                            scale=config.MDMM.scale_mmd,
                                            damping=config.MDMM.dumping_mmd,
                                            ))
        
        # sampled MMD constraint
        for i in range(4):
            for j in range(3):
                if i == 3 and j == 2:
                    break
                constraints.append(mdmm.MaxConstraint(
                                            compute_mmd_loss,
                                            config.MDMM.max_mmd_sampled_1d[j], # to be modified based on the regression
                                            scale=config.MDMM.scale_mmd,
                                            damping=config.MDMM.dumping_mmd,
                                            ))

    else:
        constraints = []
        # regression huber constraint
        constraints.append(mdmm.MaxConstraint(
                                            compute_regr_losses,
                                            config.MDMM.max_regression_huber, # to be modified based on the regression
                                            scale=config.MDMM.scale_mmd,
                                            damping=config.MDMM.dumping_mmd,
                                            ))
        
        # regression MMD constraint
        for i in range(4):
            for j in range(3):
                if i == 3 and j == 2:
                    break
                constraints.append(mdmm.MaxConstraint(
                                            compute_mmd_loss,
                                            config.MDMM.max_mmd_1d[j], # to be modified based on the regression
                                            scale=config.MDMM.scale_mmd,
                                            damping=config.MDMM.dumping_mmd,
                                            ))


        # sampled huber constraint
        constraints.append(mdmm.MaxConstraint(
                                            compute_regr_losses,
                                            config.MDMM.max_sampled_huber, # to be modified based on the regression
                                            scale=config.MDMM.scale_mmd,
                                            damping=config.MDMM.dumping_mmd,
                                            ))
        
        # sampled MMD constraint
        for i in range(4):
            for j in range(3):
                if i == 3 and j == 2:
                    break
                constraints.append(mdmm.MaxConstraint(
                                            compute_mmd_loss,
                                            config.MDMM.max_mmd_sampled_1d[j], # to be modified based on the regression
                                            scale=config.MDMM.scale_mmd,
                                            damping=config.MDMM.dumping_mmd,
                                            ))

    print(len(constraints))
        
    # Create the optimizer
    if dtype == torch.float32:
        MDMM_module = mdmm.MDMM(constraints).float() # support many constraints
    else:
        MDMM_module = mdmm.MDMM(constraints)

    loss_fn = torch.nn.HuberLoss(delta=config.training_params.huber_delta, reduction='none')

    # optimizer = optim.Adam(list(model.parameters()) , lr=config.training_params.lr)
    optimizer = MDMM_module.make_optimizer(model.parameters(), lr=config.training_params.lr)

    # Scheduler selection
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

    # attach one-hot encoded position for jets
    pos_jets_lepton_MET = [pos for pos in range(8)] # 6 jets + lepton + MET
    pos_other_jets = [8 for pos in range(no_recoObjs - 8)]
    
    pos_jets_lepton_MET = torch.tensor(pos_jets_lepton_MET, device=device, dtype=dtype)
    pos_other_jets = torch.tensor(pos_other_jets, device=device, dtype=dtype)
    pos_logScaledReco = torch.cat((pos_jets_lepton_MET, pos_other_jets), dim=0).unsqueeze(dim=1)

    # attach one-hot encoded position for partons
    pos_partons = torch.tensor([pos for pos in range(4)], device=device, dtype=dtype).unsqueeze(dim=1) # higgs, t1, t2, ISR

    # sin_cos embedding
    pos_logScaledReco = sinusoidal_positional_embedding(token_sequence_size=no_recoObjs,
                                                        token_embedding_dim=config.conditioning_transformer.hidden_features,
                                                        device=device,
                                                        n=10000.0)

    No_regressed_vars = 4
    # 9 partons
    pos_partons = sinusoidal_positional_embedding(token_sequence_size=No_regressed_vars,
                                                  token_embedding_dim=config.conditioning_transformer.hidden_features,
                                                  device=device,
                                                  n=10000.0)

    
    # new order: lepton MET higgs1 higgs2 etc
    new_order_list = [6,7,0,1,2,3,4,5]
    lastElems = [i+8 for i in range(no_recoObjs - 8)]
    new_order_list = new_order_list + lastElems
    new_order = torch.LongTensor(new_order_list)

    partons_order = [0,1,2,3]

    partons_name = ['higgs', 'thad', 'tlep', 'ISR']
    partons_var = ['pt','eta','phi']
    boost_var = ['E', 'pz']

    E_CM = 13000
    rambo = PhaseSpace(E_CM, [21,21], [25,6,-6,21], dev=device) 
    
    posLogParton = torch.linspace(0, 1, No_regressed_vars, device=device, dtype=dtype)

    ii = 0
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
            
            (logScaled_partons,
             logScaled_parton_boost,
             logScaled_reco_sortedBySpanet, mask_recoParticles,
             mask_boost_reco, data_boost_reco,
             ps_onShell_logit_scaled) = data_batch
                                        
            # exist + 3-mom
            logScaled_reco_sortedBySpanet = logScaled_reco_sortedBySpanet[:,:,[0,1,2,3]]
            # The provenance is remove in the model

            # put the lepton first:
            logScaled_reco_sortedBySpanet = logScaled_reco_sortedBySpanet[:,new_order,:]
            mask_recoParticles = mask_recoParticles[:,new_order]

            # same order for partons:
            logScaled_partons = logScaled_partons[:,partons_order,:]

            # remove prov from partons
            logScaled_partons = logScaled_partons[:,:,[0,1,2]] # [pt,eta,phi,parton_id, type] -> skip type=1/2 for partons/leptons
    
            regressed_HthadtlepISR_lab_ptetaphi_scaled, boost_regressed_Epz_scaled, flow_prob, \
            sampled_HthadtlepISR_lab_ptetaphi_scaled, boost_sampled_scaled  = ddp_model(logScaled_reco_sortedBySpanet, data_boost_reco,
                                                                    mask_recoParticles, mask_boost_reco,
                                                                    logit_ps_scaled_target = ps_onShell_logit_scaled,
                                                                                                                    
                                                                    log_mean_boost_parton=log_mean_boost_parton,
                                                                    log_std_boost_parton=log_std_boost_parton,
                                                                    log_mean_parton_Hthad=log_mean_parton_Hthad,
                                                                    log_std_parton_Hthad=log_std_parton_Hthad,
                                                                                     
                                                                    order=[0,1,2,3],
                                                                    disableGradConditioning=disable_grad_conditioning,
                                                                    flow_eval="both",
                                                                    Nsamples=1, No_regressed_vars=No_regressed_vars,
                                                                    sin_cos_embedding=True, sin_cos_reco=pos_logScaledReco,
                                                                    sin_cos_partons=pos_partons,
                                                                    attach_position_regression=posLogParton,
                                                                    rambo=rambo)
            

            flow_prob_total = -1*flow_prob.mean()
            
            MMD_input_regr = [regressed_HthadtlepISR_lab_ptetaphi_scaled[:,i] for i in range(3)] # pt eta phi
            MMD_input_regr.append(boost_regressed_Epz_scaled[:,0])

            MMD_input_sampled = [sampled_HthadtlepISR_lab_ptetaphi_scaled[:,i] for i in range(3)] # pt eta phi
            MMD_input_sampled.append(boost_sampled_scaled)

            MMD_target = [logScaled_partons[:,0], logScaled_partons[:,1], logScaled_partons[:,2]]
            MMD_target.append(logScaled_parton_boost)

            mdmm_return = MDMM_module(flow_prob_total,
                                      # regression huber constraint
                                      [(regressed_HthadtlepISR_lab_ptetaphi_scaled[:,:3], logScaled_partons[:,[0,1,2]],
                                       boost_regressed_Epz_scaled[:,0], logScaled_parton_boost,
                                       config.cartesian, loss_fn,
                                       device, dtype, False),
                                      # regression MMD constraint
                                      *[([MMD_input_regr[particle][...,feature:feature+1]], [MMD_target[particle][...,feature:feature+1]], config.training_params.mmd_kernel, device, dtype, False, False) for feature in range(3) for particle in range(4) if feature != 2 or particle != 3],
                                      # sampled huber constraint
                                      (sampled_HthadtlepISR_lab_ptetaphi_scaled[:,:3], logScaled_partons[:,[0,1,2]],
                                       boost_sampled_scaled, logScaled_parton_boost,
                                       config.cartesian, loss_fn,
                                       device, dtype, False),
                                      # sampled MMD constraint
                                      *[([MMD_input_sampled[particle][...,feature:feature+1]], [MMD_target[particle][...,feature:feature+1]], config.training_params.mmd_kernel, device, dtype, False, False) for feature in range(3) for particle in range(4) if feature != 2 or particle != 3]])
                
                
            loss_final = mdmm_return.value

            loss_final.backward()
            optimizer.step()

            with torch.no_grad():

                regr_huber_total =  compute_regr_losses(regressed_HthadtlepISR_lab_ptetaphi_scaled[:,:3], logScaled_partons[:,[0,1,2]],
                                            boost_regressed_Epz_scaled[:,0], logScaled_parton_boost,
                                            config.cartesian, loss_fn,
                                            split=False, dtype=dtype,
                                            device=device)

                sampled_huber_total =  compute_regr_losses(sampled_HthadtlepISR_lab_ptetaphi_scaled[:,:3], logScaled_partons[:,[0,1,2]],
                                            boost_sampled_scaled, logScaled_parton_boost,
                                            config.cartesian, loss_fn,
                                            split=False, dtype=dtype,
                                            device=device)

                regr_mmds = compute_mmd_loss(MMD_input_regr, MMD_target,
                                             kernel=config.training_params.mmd_kernel,
                                             device=device, total=False, dtype=dtype, split=True)

                sampled_mmds = compute_mmd_loss(MMD_input_sampled, MMD_target,
                                                kernel=config.training_params.mmd_kernel,
                                                device=device, total=False, dtype=dtype, split=True)
                
                if exp is not None and device==0 or world_size is None:
                    if i % 10 == 0:
                        for kk in range(11):
                            exp.log_metric(f"loss_regr_mmd_{kk}", regr_mmds[kk].item(),step=ii)
                            exp.log_metric(f"loss_sampled_mmd_{kk}", sampled_mmds[kk].item(),step=ii)
                        
                        exp.log_metric('loss_mmd_tot', sum(regr_mmds)/11,step=ii)
                        exp.log_metric('loss_regr_Huber_tot', regr_huber_total.item(),step=ii)
                        exp.log_metric('loss_tot_train', loss_final.item(),step=ii)
                        exp.log_metric('loss_flow_ps', flow_prob_total.mean(), step=ii)
                        exp.log_metric('loss_sampled_huber_total', sampled_huber_total, step=ii)
                        exp.log_metric('loss_sampled_mmds_total', sum(sampled_mmds)/11, step=ii)
                
                sum_loss += loss_final.item()
            

        ### END of training 
        if exp is not None and device==0 or world_size is None:
            exp.log_metric("loss_epoch_total_train", sum_loss/N_train, epoch=e, step=ii)
            exp.log_metric("learning_rate", optimizer.param_groups[0]['lr'], epoch=e, step=ii)

        valid_loss_mmd_perObj = valid_loss_sampled_mmd_perObj = [0. for i in range(11)]
        valid_loss_regr_huber_total = valid_loss_mmd_total = valid_loss_total = 0.
        valid_loss_sampled_huber_total = valid_loss_sampled_mmd_total = 0.
        valid_loss_flow_ps = 0.
        
        # validation loop (don't update weights and gradients)
        print("Before validation loop")
        ddp_model.eval()
        
        for i, data_batch in enumerate(validLoader):
            N_valid +=1
            # Move data to device
            with torch.no_grad():
                    
                (logScaled_partons,
                 logScaled_parton_boost,
                 logScaled_reco_sortedBySpanet, mask_recoParticles,
                 mask_boost_reco, data_boost_reco,
                 ps_onShell_logit_scaled) = data_batch
                                            
                # exist + 3-mom
                logScaled_reco_sortedBySpanet = logScaled_reco_sortedBySpanet[:,:,[0,1,2,3]]
                # The provenance is remove in the model
    
                # put the lepton first:
                logScaled_reco_sortedBySpanet = logScaled_reco_sortedBySpanet[:,new_order,:]
                mask_recoParticles = mask_recoParticles[:,new_order]
    
                # same order for partons:
                logScaled_partons = logScaled_partons[:,partons_order,:]
    
                # remove prov from partons
                logScaled_partons = logScaled_partons[:,:,[0,1,2]] # [pt,eta,phi,parton_id, type] -> skip type=1/2 for partons/leptons
        
                regressed_HthadtlepISR_lab_ptetaphi_scaled, boost_regressed_Epz_scaled, flow_prob, \
                sampled_HthadtlepISR_lab_ptetaphi_scaled, boost_sampled_scaled  = ddp_model(logScaled_reco_sortedBySpanet, data_boost_reco,
                                                                        mask_recoParticles, mask_boost_reco,
                                                                        logit_ps_scaled_target = ps_onShell_logit_scaled,
                                                                                                                        
                                                                        log_mean_boost_parton=log_mean_boost_parton,
                                                                        log_std_boost_parton=log_std_boost_parton,
                                                                        log_mean_parton_Hthad=log_mean_parton_Hthad,
                                                                        log_std_parton_Hthad=log_std_parton_Hthad,
                                                                                         
                                                                        order=[0,1,2,3],
                                                                        disableGradConditioning=disable_grad_conditioning,
                                                                        flow_eval="both",
                                                                        Nsamples=1, No_regressed_vars=No_regressed_vars,
                                                                        sin_cos_embedding=True, sin_cos_reco=pos_logScaledReco,
                                                                        sin_cos_partons=pos_partons,
                                                                        attach_position_regression=posLogParton,
                                                                        rambo=rambo)
                
    
                flow_prob_total = -1*flow_prob.mean()
                
                MMD_input_regr = [regressed_HthadtlepISR_lab_ptetaphi_scaled[:,i] for i in range(3)] # pt eta phi
                MMD_input_regr.append(boost_regressed_Epz_scaled[:,0])
    
                MMD_input_sampled = [sampled_HthadtlepISR_lab_ptetaphi_scaled[:,i] for i in range(3)] # pt eta phi
                MMD_input_sampled.append(boost_sampled_scaled)
    
                MMD_target = [logScaled_partons[:,0], logScaled_partons[:,1], logScaled_partons[:,2]]
                MMD_target.append(logScaled_parton_boost)
    
                mdmm_return = MDMM_module(flow_prob_total,
                                          # regression huber constraint
                                          [(regressed_HthadtlepISR_lab_ptetaphi_scaled[:,:3], logScaled_partons[:,[0,1,2]],
                                           boost_regressed_Epz_scaled[:,0], logScaled_parton_boost,
                                           config.cartesian, loss_fn,
                                           device, dtype, False),
                                          # regression MMD constraint
                                          *[([MMD_input_regr[particle][...,feature:feature+1]], [MMD_target[particle][...,feature:feature+1]], config.training_params.mmd_kernel, device, dtype, False, False) for feature in range(3) for particle in range(4) if feature != 2 or particle != 3],
                                          # sampled huber constraint
                                          (sampled_HthadtlepISR_lab_ptetaphi_scaled[:,:3], logScaled_partons[:,[0,1,2]],
                                           boost_sampled_scaled, logScaled_parton_boost,
                                           config.cartesian, loss_fn,
                                           device, dtype, False),
                                          # sampled MMD constraint
                                          *[([MMD_input_sampled[particle][...,feature:feature+1]], [MMD_target[particle][...,feature:feature+1]], config.training_params.mmd_kernel, device, dtype, False, False) for feature in range(3) for particle in range(4) if feature != 2 or particle != 3]])
                    
                    
                loss_final = mdmm_return.value

                regr_huber_total =  compute_regr_losses(regressed_HthadtlepISR_lab_ptetaphi_scaled[:,:3], logScaled_partons[:,[0,1,2]],
                                            boost_regressed_Epz_scaled[:,0], logScaled_parton_boost,
                                            config.cartesian, loss_fn,
                                            split=False, dtype=dtype,
                                            device=device)

                sampled_huber_total =  compute_regr_losses(sampled_HthadtlepISR_lab_ptetaphi_scaled[:,:3], logScaled_partons[:,[0,1,2]],
                                            boost_sampled_scaled, logScaled_parton_boost,
                                            config.cartesian, loss_fn,
                                            split=False, dtype=dtype,
                                            device=device)

                regr_mmds = compute_mmd_loss(MMD_input_regr, MMD_target,
                                             kernel=config.training_params.mmd_kernel,
                                             device=device, total=False, dtype=dtype, split=True)

                sampled_mmds = compute_mmd_loss(MMD_input_sampled, MMD_target,
                                                kernel=config.training_params.mmd_kernel,
                                                device=device, total=False, dtype=dtype, split=True)

                valid_loss_mmd_perObj = [x + y for x, y in zip(valid_loss_mmd_perObj, regr_mmds)]
                valid_loss_sampled_mmd_perObj = [x + y for x, y in zip(valid_loss_sampled_mmd_perObj, sampled_mmds)]
                valid_loss_regr_huber_total += regr_huber_total
                valid_loss_mmd_total += sum(regr_mmds)/len(regr_mmds)
                valid_loss_sampled_huber_total += sampled_huber_total
                valid_loss_sampled_mmd_total += sum(sampled_mmds)/len(sampled_mmds)
                valid_loss_total += loss_final
                valid_loss_flow_ps += flow_prob_total

                MMD_input_regr.append(regressed_HthadtlepISR_lab_ptetaphi_scaled[:,-1]) # attach ISR
                MMD_input_sampled.append(sampled_HthadtlepISR_lab_ptetaphi_scaled[:,-1]) # attach ISR
                MMD_target.append(logScaled_partons[:,-1]) # attach ISR
                
                if i == 0 and ( exp is not None and device==0 or world_size is None):
                    for particle in range(4):

                        if particle == 3:
                            index_particle = -1
                        else:
                            index_particle = particle

                        # PLOT DECAY PRODUCTS -> PT/eta/phi
                        for feature in range(3):
                            fig, ax = plt.subplots(figsize=(7,6), dpi=100)
                            h = ax.hist2d(MMD_input_regr[index_particle][:,feature].detach().cpu().numpy().flatten(),
                                          MMD_target[index_particle][:,feature].cpu().detach().numpy().flatten(),
                                          bins=40, range=((-4, 4),(-4, 4)), cmin=1)
                            fig.colorbar(h[3], ax=ax)
                            ax.set_xlabel(f"regressed {partons_var[feature]}")
                            ax.set_ylabel(f"target {partons_var[feature]}")
                            ax.set_title(f"particle {partons_name[particle]} feature {partons_var[feature]}")
                            exp.log_figure(f"2D_{partons_name[particle]}_{partons_var[feature]}", fig, step=e)

                            fig, ax = plt.subplots(figsize=(7,6), dpi=100)
                            h = ax.hist2d(MMD_input_sampled[index_particle][:,feature].detach().cpu().numpy().flatten(),
                                          MMD_target[index_particle][:,feature].cpu().detach().numpy().flatten(),
                                          bins=40, range=((-4, 4),(-4, 4)), cmin=1)
                            fig.colorbar(h[3], ax=ax)
                            ax.set_xlabel(f"sampled {partons_var[feature]}")
                            ax.set_ylabel(f"target {partons_var[feature]}")
                            ax.set_title(f"particle {partons_name[particle]} feature {partons_var[feature]}")
                            exp.log_figure(f"2D_sampled_{partons_name[particle]}_{partons_var[feature]}", fig, step=e)

                    
                            fig, ax = plt.subplots(figsize=(7,6), dpi=100)
                            ax.hist(MMD_input_regr[index_particle][:,feature].detach().cpu().numpy().flatten(),
                                          bins=30, range=(-3.2, 3.2), label="regressed", histtype="step")
                            ax.hist(MMD_target[index_particle][:,feature].cpu().detach().numpy().flatten(),
                                          bins=30, range=(-3.2, 3.2), label="target",histtype="step")
                            ax.legend()
                            ax.set_xlabel(f"{partons_name[particle]} feature {partons_var[feature]}")
                            exp.log_figure(f"1D_{partons_name[particle]}_{partons_var[feature]}", fig, step=e)

                            fig, ax = plt.subplots(figsize=(7,6), dpi=100)
                            ax.hist(MMD_input_sampled[index_particle][:,feature].detach().cpu().numpy().flatten(),
                                          bins=30, range=(-3.2, 3.2), label="sampled", histtype="step")
                            ax.hist(MMD_target[index_particle][:,feature].cpu().detach().numpy().flatten(),
                                          bins=30, range=(-3.2, 3.2), label="target",histtype="step")
                            ax.legend()
                            ax.set_xlabel(f"{partons_name[particle]} feature {partons_var[feature]}")
                            exp.log_figure(f"1D_sampled_{partons_name[particle]}_{partons_var[feature]}", fig, step=e)
                    
                    # PLOT BOOST
                    for feature in range(2):  
                        fig, ax = plt.subplots(figsize=(7,6), dpi=100)
                        h = ax.hist2d(MMD_input_regr[3][:,feature].detach().cpu().numpy().flatten(),
                                      MMD_target[3][:,feature].cpu().detach().numpy().flatten(),
                                      bins=40, range=((-3, 3),(-3, 3)), cmin=1)
                        fig.colorbar(h[3], ax=ax)
                        ax.set_xlabel(f"regressed {boost_var[feature]}")
                        ax.set_ylabel(f"target {boost_var[feature]}")
                        ax.set_title(f"boost: feature {boost_var[feature]}")
                        exp.log_figure(f"boost_2D_{boost_var[feature]}", fig,step=e)

                        fig, ax = plt.subplots(figsize=(7,6), dpi=100)
                        h = ax.hist2d(MMD_input_sampled[3][:,feature].detach().cpu().numpy().flatten(),
                                      MMD_target[3][:,feature].cpu().detach().numpy().flatten(),
                                      bins=40, range=((-3, 3),(-3, 3)), cmin=1)
                        fig.colorbar(h[3], ax=ax)
                        ax.set_xlabel(f"sampled {boost_var[feature]}")
                        ax.set_ylabel(f"target {boost_var[feature]}")
                        ax.set_title(f"boost: feature {boost_var[feature]}")
                        exp.log_figure(f"boost_2D_sampled_{boost_var[feature]}", fig,step=e)

                
                        fig, ax = plt.subplots(figsize=(7,6), dpi=100)
                        ax.hist(MMD_input_regr[3][:,feature].detach().cpu().numpy().flatten(),
                                      bins=30, range=(-2, 2), label="regressed", histtype="step")
                        ax.hist(MMD_target[3][:,feature].cpu().detach().numpy().flatten(),
                                      bins=30, range=(-2, 2), label="target",histtype="step")
                        ax.legend()
                        ax.set_xlabel(f"boost: feature {boost_var[feature]}")
                        exp.log_figure(f"boost_1D_{boost_var[feature]}", fig, step=e)

                        fig, ax = plt.subplots(figsize=(7,6), dpi=100)
                        ax.hist(MMD_input_sampled[3][:,feature].detach().cpu().numpy().flatten(),
                                      bins=30, range=(-2, 2), label="sampled", histtype="step")
                        ax.hist(MMD_target[3][:,feature].cpu().detach().numpy().flatten(),
                                      bins=30, range=(-2, 2), label="target",histtype="step")
                        ax.legend()
                        ax.set_xlabel(f"boost: feature {boost_var[feature]}")
                        exp.log_figure(f"boost_1D_sampled_{boost_var[feature]}", fig, step=e)

        if exp is not None and device==0 or world_size is None:
            for kk in range(11):
                exp.log_metric(f"loss_mmd_val_{kk}", valid_loss_mmd_perObj[kk]/N_valid,epoch= e)
                exp.log_metric(f"loss_mmd_sampled_val_{kk}", valid_loss_sampled_mmd_perObj[kk]/N_valid,epoch= e)
            
            exp.log_metric('valid_loss_regr_total', valid_loss_regr_huber_total/N_valid,epoch= e)
            exp.log_metric('valid_loss_mmd_total', valid_loss_mmd_total/N_valid,epoch= e)
            exp.log_metric('valid_loss_sampled_huber', valid_loss_sampled_huber_total/N_valid,epoch= e)
            exp.log_metric('valid_loss_mmd_sampled_total', valid_loss_sampled_mmd_total/N_valid,epoch= e)
            
            exp.log_metric('valid_loss_total', valid_loss_total/N_valid,epoch= e)
            exp.log_metric('valid_loss_flow_ps', valid_loss_flow_ps/N_valid,epoch= e)
            

        if device == 0 or world_size is None:
            if early_stopper.early_stop(valid_loss_total/N_valid,
                                    model.state_dict(), optimizer.state_dict(), modelName, exp):
                print(f"Model converges at epoch {e} !!!")         
                break

        # Step the scheduler at the end of the val
        # after_N_epochs = config.training_params.cosine_scheduler.get("after_N_epochs", 0)
        # if e > after_N_epochs:
        #     scheduler.step()
        
        scheduler.step(valid_loss_total/N_valid) # reduce lr if the model is not improving anymore
        

    # writer.close()
    # exp_log.end()
    destroy_process_group()
    print('preTraining finished!!')
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-config', type=str, required=True, help='path to config.yaml File')
    parser.add_argument('--path-regression', type=str, required=True, help='path to regression File')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--on-GPU', action="store_true",  help='run on GPU boolean')
    parser.add_argument('--disable-grad-conditioning', action="store_true",  help='run on GPU boolean')
    parser.add_argument('--distributed', action="store_true")
    args = parser.parse_args()
    
    path_to_conf = args.path_config
    path_regression = args.path_regression
    on_GPU = args.on_GPU # by default run on CPU
    disable_grad_conditioning = args. disable_grad_conditioning
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
    latentSpace = conf.conditioning_transformer.use_latent
    name_dir = f'{outputDir}/UnfoldingFlow_Propag_SampleLikelihoodLosses_freeze:{disable_grad_conditioning}_date&Hour_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}'

    os.makedirs(name_dir, exist_ok=True)
    
    with open(f"{name_dir}/config_{conf.name}_{conf.version}.yaml", "w") as fo:
        fo.write(OmegaConf.to_yaml(conf)) 

    if conf.training_params.dtype == "float32":
        dtype = torch.float32
    elif conf.training_params.dtype == "float64":
        dtype = torch.float64
        
    
    if args.distributed:
        
        # make a dictionary with k: rank, v: actual device
        dev_dct = {i: actual_devices[i] for i in range(world_size)}
        print(f"Devices dict: {dev_dct}")
        mp.spawn(
            train,
            args=(name_dir, conf,  outputDir, dtype,
                    world_size, dev_dct, path_regression, disable_grad_conditioning),
            nprocs=world_size,
            # join=True
        )
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        train(device,name_dir, conf,  outputDir, dtype, path_regression=path_regression, disable_grad_conditioning=disable_grad_conditioning)
    
    print(f"Flow finished succesfully! Version: {conf.version}")