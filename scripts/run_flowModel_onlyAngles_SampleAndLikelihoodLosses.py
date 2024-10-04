from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
from random import randint

from datetime import datetime

import torch
from memflow.read_data.dataset_all import DatasetCombined

from memflow.unfolding_flow.unfolding_flow_withDecay_v2_onlyAngles import UnfoldingFlow_withDecay_v2
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

def compute_regr_losses_regression(MMD_input, MMD_target, boost_input, boost_target,
                        cartesian, loss_fn,
                        device, dtype, split=False):
    
    loss = torch.zeros(13, device=device, dtype=dtype)

    for feature in range(MMD_input[0].shape[1]):
        # if feature != phi
        if feature != 2:

            if feature == 0:
                loss_all = 4*loss_fn(MMD_input[...,feature], MMD_target[...,feature])
            else:
                loss_all = 2*loss_fn(MMD_input[...,feature], MMD_target[...,feature])
                
            loss_boost = loss_fn(boost_input[...,feature], boost_target[...,feature]).unsqueeze(dim=1)

            loss += torch.sum(torch.cat((loss_all, loss_boost), dim=1), dim=0)/loss_all.shape[0]
            
        # case when feature is equal to phi (phi is periodic variable)
        else:
            loss_phi = 5*loss_fn_periodic(MMD_input[...,feature],
                                          MMD_target[...,feature], loss_fn, device, dtype)
            
            zero_phi = torch.zeros((loss_phi.shape[0], 1), device=device, dtype=dtype)
                                    
            loss += torch.sum(torch.cat((loss_phi, zero_phi), dim=1), dim=0)/loss_all.shape[0]
    
    if split:
        return loss
    else:
        return  sum(loss)/36

def compute_regr_losses(MMD_input, MMD_target,
                        cartesian, loss_fn,
                        device, dtype, split=False):
    
    loss = torch.zeros(5, device=device, dtype=dtype)

    for feature in range(2):
        # if feature != phi
        if feature == 0:

            loss_all = loss_fn(MMD_input[...,feature], MMD_target[...,feature])
    
            loss += torch.sum(loss_all, dim=0)/loss_all.shape[0]
            
        # case when feature is equal to phi (phi is periodic variable)
        else:
            loss_phi = 2*loss_fn_periodic(MMD_input[...,feature],
                                          MMD_target[...,feature], loss_fn, device, dtype)
                                                
            loss += torch.sum(loss_phi, dim=0)/loss_all.shape[0]
    
    if split:
        return loss
    else:
        return  sum(loss)/10
        


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
                                    dtype=dtype, datasets=['partons_lab', 'reco_lab'],
                           reco_list_lab=['scaledLogReco_sortedBySpanet',
                                          'mask_scaledLogReco_sortedBySpanet',
                                          'mask_boost', 'scaledLogBoost'],
                           parton_list_lab=['tensor_AllPartons',
                                           'logScaled_data_boost',
                                           'Hb1_thad_b_q1_tlep_b_el_CM'])

    val_dataset = DatasetCombined(config.input_dataset_validation,dev=device, new_higgs=True,
                                  dtype=dtype, datasets=['partons_lab', 'reco_lab'],
                           reco_list_lab=['scaledLogReco_sortedBySpanet',
                                          'mask_scaledLogReco_sortedBySpanet',
                                          'mask_boost', 'scaledLogBoost'],
                           parton_list_lab=['tensor_AllPartons',
                                           'logScaled_data_boost',
                                           'Hb1_thad_b_q1_tlep_b_el_CM'])

    no_recoObjs = train_dataset.reco_lab.scaledLogReco_sortedBySpanet.shape[1]

    log_mean_reco = train_dataset.reco_lab.meanRecoParticles
    log_std_reco = train_dataset.reco_lab.stdRecoParticles
    log_mean_parton = train_dataset.partons_lab.mean_log_partonsLeptons
    log_std_parton = train_dataset.partons_lab.std_log_partonsLeptons
    log_mean_parton_Hthad = train_dataset.partons_lab.mean_log_data_higgs_t_tbar_ISR
    log_std_parton_Hthad = train_dataset.partons_lab.std_log_data_higgs_t_tbar_ISR
    log_mean_boost_parton = train_dataset.partons_lab.mean_log_data_boost
    log_std_boost_parton = train_dataset.partons_lab.std_log_data_boost
    
    if device == torch.device('cuda'):
        log_mean_reco = log_mean_reco.cuda()
        log_std_reco = log_std_reco.cuda()
        log_mean_parton = log_mean_parton.cuda()
        log_std_parton = log_std_parton.cuda()

        log_mean_parton_Hthad = log_mean_parton_Hthad.cuda()
        log_std_parton_Hthad = log_std_parton_Hthad.cuda()
        log_mean_boost_parton = log_mean_boost_parton.cuda()
        log_std_boost_parton = log_std_boost_parton.cuda()


    model = UnfoldingFlow_withDecay_v2(
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
                                 
                                 flow_context_angles=config.unfolding_flow_anglesCM.ncond,
                                 flow_ntransforms_angles=config.unfolding_flow_anglesCM.ntransforms,
                                 flow_nbins_angles=config.unfolding_flow_anglesCM.bins,
                                 flow_hiddenMLP_LayerDim_angles=config.unfolding_flow_anglesCM.hiddenMLP_LayerDim,
                                 flow_hiddenMLP_NoLayers_angles=config.unfolding_flow_anglesCM.hiddenMLP_NoLayers,
                                 flow_base_anglesCM=config.unfolding_flow_anglesCM.base,
                                 flow_base_first_arg_anglesCM=config.unfolding_flow_anglesCM.base_first_arg,
                                 flow_base_second_arg_anglesCM=config.unfolding_flow_anglesCM.base_second_arg,
                                 randPerm_angles=True,
                                 
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
        exp.add_tags([config.name,config.version, 'only angles', 'Unfolding Flow', 'Sample Loss', 'withDecay', 'angles_CM_withTheta', 'likelihood+Samples',
                     f'autoreg={config.unfolding_flow.autoregressive}', f'freeze_pretrain={disable_grad_conditioning}'])
        exp.log_parameters({"model_param_tot": count_parameters(model)})
        exp.log_parameters(config)
        
        exp.set_name(f"UnfoldingFlow_withDecay_SampleLoss_onlyAngles_freeze:{disable_grad_conditioning}_{randint(0, 1000)}")
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
        constraints.append(mdmm.MaxConstraint(
                                            compute_regr_losses_regression,
                                            100000.0, # to be modified based on the regression
                                            scale=config.MDMM.scale_mmd,
                                            damping=config.MDMM.dumping_mmd,
                                            ))
        for i in range(5):
            # eta/phi
            for j in range(2):
                constraints.append(mdmm.MaxConstraint(
                                            compute_mmd_loss,
                                            100000.0, # to be modified based on the regression
                                            scale=config.MDMM.scale_mmd,
                                            damping=config.MDMM.dumping_mmd,
                                            ))

    else:
        constraints = []
        constraints.append(mdmm.MaxConstraint(
                                            compute_regr_losses_regression,
                                            config.MDMM.max_regression_huber, # to be modified based on the regression
                                            scale=config.MDMM.scale_mmd,
                                            damping=config.MDMM.dumping_mmd,
                                            ))
        for i in range(5):
            # eta/phi
            for j in range(2):
                constraints.append(mdmm.MaxConstraint(
                                            compute_mmd_loss,
                                            config.MDMM.max_mmd_1d[j], # to be modified based on the regression
                                            scale=config.MDMM.scale_mmd,
                                            damping=config.MDMM.dumping_mmd,
                                            ))
        
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
    pos_partons = torch.tensor([pos for pos in range(9)], device=device, dtype=dtype).unsqueeze(dim=1) # higgs, t1, t2, ISR

    # sin_cos embedding
    pos_logScaledReco = sinusoidal_positional_embedding(token_sequence_size=no_recoObjs,
                                                        token_embedding_dim=config.conditioning_transformer.hidden_features,
                                                        device=device,
                                                        n=10000.0)

    No_regressed_vars = 9
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

    
    # lepton1/neutrino/higgs 1,2/thad 1,2,3/tlep/ISR/...
    # higgs and thad ordered by pt
    #partons_order = [7,8,0,1,2,4,5,3,6,9,10,11]
    # need new order for partons: [higgs, higgs_b1, ...] --> check utils file
    # [higgs, higgs_b1, tlep, tlep_e, tlep_b, thad, thad_b, thad_q1, higgs_b2, tlep_nu, thad_q2, ISR]
    partons_order = [9, 0, 11, 7, 3, 10, 2, 4, 1, 8, 5, 6]

    partons_name = ['higgs B1', 'thad B', 'thad Q1', 'tlep B', 'tlep e']
    partons_var = ['eta','phi']
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
             Hb1_thad_b_q1_tlep_b_el_CM,
             logScaled_reco_sortedBySpanet, mask_recoParticles,
             mask_boost_reco, data_boost_reco) = data_batch
                                        
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
    
            regressed_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab, boost_regressed_Epz_scaled, \
            flow_prob_higgs, flow_prob_thad_b, flow_prob_thad_W, flow_prob_tlep_b, flow_prob_tlep_W, \
            higgs_etaPhi_unscaled_CM_sampled, thad_b_etaPhi_unscaled_CM_sampled, thad_W_etaPhi_unscaled_CM_sampled, \
            tlep_b_etaPhi_unscaled_CM_sampled, tlep_W_etaPhi_unscaled_CM_sampled  = ddp_model(logScaled_reco_sortedBySpanet,
                                                                                              data_boost_reco,
                                                                    mask_recoParticles, mask_boost_reco,
                                                          
                                                                    higgs_etaPhi_unscaled_CM_target = Hb1_thad_b_q1_tlep_b_el_CM[:,0:1,1:3],
                                                                    thad_etaPhi_unscaled_CM_target = Hb1_thad_b_q1_tlep_b_el_CM[:,1:3,1:3],
                                                                    tlep_etaPhi_unscaled_CM_target = Hb1_thad_b_q1_tlep_b_el_CM[:,3:5,1:3],
                                                          
                                                                    log_mean_parton=log_mean_parton, 
                                                                    log_std_parton=log_std_parton,
                                                                    log_mean_boost_parton=log_mean_boost_parton,
                                                                    log_std_boost_parton=log_std_boost_parton,
                                                                    log_mean_parton_Hthad=log_mean_parton_Hthad,
                                                                    log_std_parton_Hthad=log_std_parton_Hthad,
                                                                    order=[0,1,2,3],
                                                                    disableGradConditioning=disable_grad_conditioning,
                                                                    flow_eval="both",
                                                                    Nsamples=1, No_regressed_vars=9,
                                                                    sin_cos_embedding=True, sin_cos_reco=pos_logScaledReco,
                                                                    sin_cos_partons=pos_partons,
                                                                    attach_position_regression=posLogParton,
                                                                    rambo=rambo)


            sampled_higgsb1_thad_b_W_tlep_b_W_angles = torch.cat((higgs_etaPhi_unscaled_CM_sampled,
                                                                   thad_b_etaPhi_unscaled_CM_sampled,
                                                                   thad_W_etaPhi_unscaled_CM_sampled,
                                                                   tlep_b_etaPhi_unscaled_CM_sampled,
                                                                   tlep_W_etaPhi_unscaled_CM_sampled), dim=1)            

            flow_prob_total = -1*(flow_prob_higgs + flow_prob_thad_b + flow_prob_thad_W + \
                                flow_prob_tlep_b + flow_prob_tlep_W).mean()
                                    
            MMD_input_regression = [regressed_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab[:,i,1:] for i in range(12)] # pt eta phi
            MMD_input_samples = [sampled_higgsb1_thad_b_W_tlep_b_W_angles[:,i] for i in range(5)] # pt eta 
            
            zeros_boost = torch.zeros((logScaled_parton_boost.shape[0], 1), device=device, dtype=dtype)
            boost_regressed_Epz_scaled = torch.cat((boost_regressed_Epz_scaled[:,0], zeros_boost), dim=1)
            MMD_input_regression.append(boost_regressed_Epz_scaled)

            MMD_target = [logScaled_partons[:,0],
                         logScaled_partons[:,1],
                         logScaled_partons[:,8],
                         logScaled_partons[:,5],
                         logScaled_partons[:,6],
                         logScaled_partons[:,7],
                         logScaled_partons[:,10],
                         logScaled_partons[:,2],
                         logScaled_partons[:,4],
                         logScaled_partons[:,3],
                         logScaled_partons[:,9],
                         logScaled_partons[:,11]]
            logScaled_parton_boost = torch.cat((logScaled_parton_boost, zeros_boost), dim=1)
            MMD_target.append(logScaled_parton_boost)
            MMD_target_angles = [Hb1_thad_b_q1_tlep_b_el_CM[:,i,1:] for i in range(5)] # pt eta            

            sampled_loss_hubber_perObj =  compute_regr_losses(sampled_higgsb1_thad_b_W_tlep_b_W_angles,
                                                              Hb1_thad_b_q1_tlep_b_el_CM[...,1:3],
                                                                config.cartesian, loss_fn,
                                                                device, dtype, split=True)

            sampled_loss_huber =  torch.sum(sampled_loss_hubber_perObj)/10

            # MMDS ORDER: 8 partons + boost + total = 10
            sampled_mmds = compute_mmd_loss(MMD_input_samples, MMD_target_angles,
                                              kernel=config.training_params.mmd_kernel,
                                              device=device, total=False, dtype=dtype, split=True)

            sampled_mmds_total = sum(sampled_mmds)/len(sampled_mmds)

            #print('check nan')
            #print(torch.isnan(sampled_loss_hubber_perObj).any())
            #for i in range(38):
            #    print(torch.isnan(sampled_mmds[i]).any())
            #print(torch.isnan(sampled_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab).any())
            #print()

            full_loss = flow_prob_total + sampled_loss_huber + sampled_mmds_total

            mdmm_return = MDMM_module(full_loss, [(regressed_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab[...,1:], logScaled_partons[:,[0,1,8,5,6,7,10,2,4,3,9,11]], \
                                                          boost_regressed_Epz_scaled, logScaled_parton_boost, \
                                                          config.cartesian, loss_fn, device, dtype, False), \
                                                        *[([MMD_input_regression[particle][...,feature:feature+1]],
                                                           [MMD_target[particle][...,feature:feature+1]],
                                                           config.training_params.mmd_kernel, device, dtype, False, False) for feature in range(3) for particle in range(len(MMD_target)) if feature != 2 and particle != len(MMD_target)-1]])

            loss_final = mdmm_return.value

            loss_final.backward()
            optimizer.step()

            with torch.no_grad():

                regr_loss_hubber_perObj =  compute_regr_losses_regression(regressed_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab[...,1:],
                                            logScaled_partons[:, [0,1,8,5,6,7,10,2,4,3,9,11]],
                                            boost_regressed_Epz_scaled, logScaled_parton_boost,
                                            config.cartesian, loss_fn,
                                            split=True, dtype=dtype,
                                            device=device)

                regr_loss_huber =  torch.sum(regr_loss_hubber_perObj)/36

                # MMDS ORDER: 8 partons + boost + total = 10
                mmds = compute_mmd_loss(MMD_input_regression, MMD_target,
                                      kernel=config.training_params.mmd_kernel,
                                      device=device, total=False, dtype=dtype, split=True)
                
                if exp is not None and device==0 or world_size is None:
                    if i % 10 == 0:
                        for kk in range(38):
                            exp.log_metric(f"loss_regr_mmd_{kk}", mmds[kk].item(),step=ii)
                            if kk < 10:
                                exp.log_metric(f"loss_sampled_mmd_{kk}", sampled_mmds[kk].item(),step=ii)
                        for kk in range(13):
                            exp.log_metric(f"loss_regr_huber_{kk}", regr_loss_hubber_perObj[kk],step=ii)
                            if kk < 5:
                                exp.log_metric(f"loss_sampled_huber_{kk}", sampled_loss_hubber_perObj[kk],step=ii)
                        
                        exp.log_metric('loss_mmd_tot', sum(mmds)/13,step=ii)
                        exp.log_metric('loss_regr_Huber_tot', regr_loss_huber.item(),step=ii)
                        exp.log_metric('loss_tot_train', loss_final.item(),step=ii)
                        exp.log_metric('loss_flow_higgs', -1*flow_prob_higgs.mean(), step=ii)
                        exp.log_metric('loss_flow_thad_b', -1*flow_prob_thad_b.mean(), step=ii)
                        exp.log_metric('loss_flow_thad_W', -1*flow_prob_thad_W.mean(), step=ii)
                        exp.log_metric('loss_flow_tlep_b', -1*flow_prob_tlep_b.mean(), step=ii)
                        exp.log_metric('loss_flow_tlep_W', -1*flow_prob_tlep_W.mean(), step=ii)
                        exp.log_metric('loss_flow_total', full_loss, step=ii)
                        exp.log_metric('loss_sampled_huber_total', sampled_loss_huber, step=ii)
                        exp.log_metric('loss_sampled_mmds_total', sampled_mmds_total, step=ii)
                
                sum_loss += loss_final.item()
            

        ### END of training 
        if exp is not None and device==0 or world_size is None:
            exp.log_metric("loss_epoch_total_train", sum_loss/N_train, epoch=e, step=ii)
            exp.log_metric("learning_rate", optimizer.param_groups[0]['lr'], epoch=e, step=ii)

        valid_loss_regr_huber_perObj = torch.zeros(13, device=device, dtype=dtype)
        valid_sampled_regr_huber_perObj = torch.zeros(5, device=device, dtype=dtype)
        valid_loss_mmd_perObj = [0. for i in range(38)]
        valid_loss_sampled_mmd_perObj = [0. for i in range(10)]
        valid_loss_regr_huber_total = valid_loss_mmd_total = valid_loss_total = 0.
        valid_loss_sampled_huber_total = valid_loss_sampled_mmd_total = 0.
        valid_loss_flow_ps = valid_loss_flow_total = valid_loss_flow_higgs = 0.
        valid_loss_flow_thad_b = valid_loss_flow_thad_W = valid_loss_flow_tlep_b = valid_loss_flow_tlep_W = 0.
        
        # validation loop (don't update weights and gradients)
        print("Before validation loop")
        ddp_model.eval()
        
        for i, data_batch in enumerate(validLoader):
            N_valid +=1
            # Move data to device
            with torch.no_grad():
                    
                (logScaled_partons,
                 logScaled_parton_boost,
                 Hb1_thad_b_q1_tlep_b_el_CM,
                 logScaled_reco_sortedBySpanet, mask_recoParticles,
                 mask_boost_reco, data_boost_reco) = data_batch
                                            
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
        
                regressed_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab, boost_regressed_Epz_scaled, \
                flow_prob_higgs, flow_prob_thad_b, flow_prob_thad_W, flow_prob_tlep_b, flow_prob_tlep_W, \
                higgs_etaPhi_unscaled_CM_sampled, thad_b_etaPhi_unscaled_CM_sampled, thad_W_etaPhi_unscaled_CM_sampled, \
                tlep_b_etaPhi_unscaled_CM_sampled, tlep_W_etaPhi_unscaled_CM_sampled  = ddp_model(logScaled_reco_sortedBySpanet,
                                                                                                  data_boost_reco,
                                                                        mask_recoParticles, mask_boost_reco,
                                                              
                                                                        higgs_etaPhi_unscaled_CM_target = Hb1_thad_b_q1_tlep_b_el_CM[:,0:1,1:3],
                                                                        thad_etaPhi_unscaled_CM_target = Hb1_thad_b_q1_tlep_b_el_CM[:,1:3,1:3],
                                                                        tlep_etaPhi_unscaled_CM_target = Hb1_thad_b_q1_tlep_b_el_CM[:,3:5,1:3],
                                                              
                                                                        log_mean_parton=log_mean_parton, 
                                                                        log_std_parton=log_std_parton,
                                                                        log_mean_boost_parton=log_mean_boost_parton,
                                                                        log_std_boost_parton=log_std_boost_parton,
                                                                        log_mean_parton_Hthad=log_mean_parton_Hthad,
                                                                        log_std_parton_Hthad=log_std_parton_Hthad,
                                                                        order=[0,1,2,3],
                                                                        disableGradConditioning=disable_grad_conditioning,
                                                                        flow_eval="both",
                                                                        Nsamples=1, No_regressed_vars=9,
                                                                        sin_cos_embedding=True, sin_cos_reco=pos_logScaledReco,
                                                                        sin_cos_partons=pos_partons,
                                                                        attach_position_regression=posLogParton,
                                                                        rambo=rambo)
    
                sampled_higgsb1_thad_b_W_tlep_b_W_angles = torch.cat((higgs_etaPhi_unscaled_CM_sampled,
                                                                       thad_b_etaPhi_unscaled_CM_sampled,
                                                                       thad_W_etaPhi_unscaled_CM_sampled,
                                                                       tlep_b_etaPhi_unscaled_CM_sampled,
                                                                       tlep_W_etaPhi_unscaled_CM_sampled), dim=1)            
    
                flow_prob_total = -1*(flow_prob_higgs + flow_prob_thad_b + flow_prob_thad_W + \
                                    flow_prob_tlep_b + flow_prob_tlep_W).mean()
                                    
                MMD_input_regression = [regressed_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab[:,i,1:] for i in range(12)] # pt eta phi
                MMD_input_samples = [sampled_higgsb1_thad_b_W_tlep_b_W_angles[:,i] for i in range(5)] # pt eta 
                
                zeros_boost = torch.zeros((logScaled_parton_boost.shape[0], 1), device=device, dtype=dtype)
                boost_regressed_Epz_scaled = torch.cat((boost_regressed_Epz_scaled[:,0], zeros_boost), dim=1)
                MMD_input_regression.append(boost_regressed_Epz_scaled)
    
                MMD_target = [logScaled_partons[:,0],
                             logScaled_partons[:,1],
                             logScaled_partons[:,8],
                             logScaled_partons[:,5],
                             logScaled_partons[:,6],
                             logScaled_partons[:,7],
                             logScaled_partons[:,10],
                             logScaled_partons[:,2],
                             logScaled_partons[:,4],
                             logScaled_partons[:,3],
                             logScaled_partons[:,9],
                             logScaled_partons[:,11]]
                logScaled_parton_boost = torch.cat((logScaled_parton_boost, zeros_boost), dim=1)
                MMD_target.append(logScaled_parton_boost)
                MMD_target_angles = [Hb1_thad_b_q1_tlep_b_el_CM[:,i,1:] for i in range(5)] # pt eta 
    
                sampled_loss_hubber_perObj =  compute_regr_losses(sampled_higgsb1_thad_b_W_tlep_b_W_angles,
                                                                  Hb1_thad_b_q1_tlep_b_el_CM[...,1:3],
                                                                    config.cartesian, loss_fn,
                                                                    device, dtype, split=True)
    
                sampled_loss_huber =  torch.sum(sampled_loss_hubber_perObj)/10
    
                # MMDS ORDER: 8 partons + boost + total = 10
                sampled_mmds = compute_mmd_loss(MMD_input_samples, MMD_target_angles,
                                                  kernel=config.training_params.mmd_kernel,
                                                  device=device, total=False, dtype=dtype, split=True)
    
                sampled_mmds_total = sum(sampled_mmds)/len(sampled_mmds)

                full_loss = flow_prob_total + sampled_loss_huber + sampled_mmds_total
    
                mdmm_return = MDMM_module(full_loss, [(regressed_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab[...,1:], logScaled_partons[:,[0,1,8,5,6,7,10,2,4,3,9,11]], \
                                                              boost_regressed_Epz_scaled, logScaled_parton_boost, \
                                                              config.cartesian, loss_fn, device, dtype, False), \
                                                            *[([MMD_input_regression[particle][...,feature:feature+1]],
                                                               [MMD_target[particle][...,feature:feature+1]],
                                                               config.training_params.mmd_kernel, device, dtype, False, False) for feature in range(3) for particle in range(len(MMD_target)) if feature != 2 and particle != len(MMD_target)-1]])

                loss_final = mdmm_return.value

            

                regr_loss_hubber_perObj =  compute_regr_losses_regression(regressed_Hb1b2_thad_b_q1q2_tlep_b_el_nu_ISR_Eptetaphi_scaled_lab[...,1:],
                                            logScaled_partons[:, [0,1,8,5,6,7,10,2,4,3,9,11]],
                                            boost_regressed_Epz_scaled, logScaled_parton_boost,
                                            config.cartesian, loss_fn,
                                            split=True, dtype=dtype,
                                            device=device)

                regr_loss_huber =  torch.sum(regr_loss_hubber_perObj)/36

                # MMDS ORDER: 8 partons + boost + total = 10
                mmds = compute_mmd_loss(MMD_input_regression, MMD_target,
                                      kernel=config.training_params.mmd_kernel,
                                      device=device, total=False, dtype=dtype, split=True)

                valid_loss_regr_huber_perObj += regr_loss_hubber_perObj
                valid_sampled_regr_huber_perObj += sampled_loss_hubber_perObj
                valid_loss_mmd_perObj = [x + y for x, y in zip(valid_loss_mmd_perObj, mmds)]
                valid_loss_sampled_mmd_perObj = [x + y for x, y in zip(valid_loss_sampled_mmd_perObj, sampled_mmds)]
                valid_loss_regr_huber_total += regr_loss_huber
                valid_loss_sampled_huber_total += sampled_loss_huber
                valid_loss_mmd_total += sum(mmds)/len(mmds)
                valid_loss_sampled_mmd_total += sum(sampled_mmds)/len(sampled_mmds)

                valid_loss_total += loss_final
                valid_loss_flow_total += flow_prob_total + valid_loss_sampled_mmd_total + valid_loss_sampled_huber_total
                valid_loss_flow_higgs += -1*flow_prob_higgs.mean()
                valid_loss_flow_thad_b += -1*flow_prob_thad_b.mean()
                valid_loss_flow_thad_W += -1*flow_prob_thad_W.mean()
                valid_loss_flow_tlep_b += -1*flow_prob_tlep_b.mean()
                valid_loss_flow_tlep_W += -1*flow_prob_tlep_W.mean()
            
                if i == 0 and ( exp is not None and device==0 or world_size is None):
                    for particle in range(5):

                        # PLOT DECAY PRODUCTS -> PT/eta/phi
                        for feature in range(2):  

                            # plot samples
                            fig, ax = plt.subplots(figsize=(7,6), dpi=100)
                            h = ax.hist2d(MMD_input_samples[particle][:,feature].detach().cpu().numpy().flatten(),
                                          MMD_target_angles[particle][:,feature].cpu().detach().numpy().flatten(),
                                          bins=40, range=((-4, 4),(-4, 4)), cmin=1)
                            fig.colorbar(h[3], ax=ax)
                            ax.set_xlabel(f"sampled flow: {partons_var[feature]}")
                            ax.set_ylabel(f"target {partons_var[feature]}")
                            ax.set_title(f"particle {partons_name[particle]} feature {partons_var[feature]}")
                            exp.log_figure(f"2D_sampled_{partons_name[particle]}_{partons_var[feature]}", fig, step=e)

                    
                            fig, ax = plt.subplots(figsize=(7,6), dpi=100)
                            ax.hist(MMD_input_samples[particle][:,feature].detach().cpu().numpy().flatten(),
                                          bins=30, range=(-3.2, 3.2), label="sampled flow", histtype="step")
                            ax.hist(MMD_target_angles[particle][:,feature].cpu().detach().numpy().flatten(),
                                          bins=30, range=(-3.2, 3.2), label="target",histtype="step")
                            ax.legend()
                            ax.set_xlabel(f"{partons_name[particle]} feature {partons_var[feature]}")
                            exp.log_figure(f"1D_sampled_{partons_name[particle]}_{partons_var[feature]}", fig, step=e)
                  

        if exp is not None and device==0 or world_size is None:
            for kk in range(13):
                exp.log_metric(f"valid_loss_regr_perObj_{kk}", valid_loss_regr_huber_perObj[kk]/N_valid, epoch=e )
                if kk < 5:
                    exp.log_metric(f"valid_loss_sampled_huber_perObj_{kk}", valid_sampled_regr_huber_perObj[kk]/N_valid, epoch=e )
            for kk in range(38):
                exp.log_metric(f"loss_mmd_val_{kk}", valid_loss_mmd_perObj[kk]/N_valid,epoch= e)
                if kk < 10:
                    exp.log_metric(f"loss_mmd_sampled_val_{kk}", valid_loss_sampled_mmd_perObj[kk]/N_valid,epoch= e)
            
            exp.log_metric('valid_loss_regr_total', valid_loss_regr_huber_total/N_valid,epoch= e)
            exp.log_metric('valid_loss_mmd_total', valid_loss_mmd_total/N_valid,epoch= e)
            exp.log_metric('valid_loss_sampled_huber', valid_loss_sampled_huber_total/N_valid,epoch= e)
            exp.log_metric('valid_loss_mmd_sampled_total', valid_loss_sampled_mmd_total/N_valid,epoch= e)
            
            exp.log_metric('valid_loss_total', valid_loss_total/N_valid,epoch= e)
            exp.log_metric('valid_loss_flow_total', valid_loss_flow_total/N_valid,epoch= e)
            exp.log_metric('valid_loss_flow_higgs', valid_loss_flow_higgs/N_valid,epoch= e)
            exp.log_metric('valid_loss_flow_thad_b', valid_loss_flow_thad_b/N_valid,epoch= e)
            exp.log_metric('valid_loss_flow_thad_W', valid_loss_flow_thad_W/N_valid,epoch= e)
            exp.log_metric('valid_loss_flow_tlep_b', valid_loss_flow_tlep_b/N_valid,epoch= e)
            exp.log_metric('valid_loss_flow_tlep_W', valid_loss_flow_tlep_W/N_valid,epoch= e)
            

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
    name_dir = f'{outputDir}/UnfoldingFlow_withDecay_SampleLoss_onlyAngles_freeze:{disable_grad_conditioning}_date&Hour_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}'

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