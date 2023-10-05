from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

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

from earlystop import EarlyStopper
from memflow.unfolding_flow.utils import Compute_ParticlesTensor

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.profiler import profile, record_function, ProfilerActivity

from random import randint
PI = torch.pi

# torch.autograd.set_detect_anomaly(True)

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


def loss_fn_periodic(inp, target, loss_fn, device):
    # penalty term for going further than PI
    overflow_delta = torch.clamp(inp - PI, min=0.).mean()  - torch.clamp(inp + PI, max=0.).mean()
    # rescale to pi for the loss itself
    inp[inp>PI] = inp[inp>PI]-2*PI
    inp[inp<-PI] = inp[inp<-PI] + 2*PI
    
    deltaPhi = target - inp
    deltaPhi = torch.where(deltaPhi > PI, deltaPhi - 2*PI, deltaPhi)
    deltaPhi = torch.where(deltaPhi <= -PI, deltaPhi + 2*PI, deltaPhi)
    return loss_fn(deltaPhi, torch.zeros(deltaPhi.shape, device=device)) + 5. * overflow_delta


def compute_regr_losses(logScaled_partons, logScaled_boost, higgs, thad, tlep, gluon, boost, cartesian, loss_fn,
                   scaling_phi, device, split=False):
    lossH = 0.
    lossThad = 0.
    lossTlep = 0.
    lossGluon = 0. 

    if cartesian:
        lossH = loss_fn(logScaled_partons[:,0], higgs)
        lossThad =  loss_fn(logScaled_partons[:,1], thad)
        lossTlep =  loss_fn(logScaled_partons[:,2], tlep)
    else:
        for feature in range(higgs.shape[1]):
            # if feature != phi
            if feature != 2:
                lossH += loss_fn(logScaled_partons[:,0,feature], higgs[:,feature])
                lossThad +=  loss_fn(logScaled_partons[:,1,feature], thad[:,feature])
                lossTlep +=  loss_fn(logScaled_partons[:,2,feature], tlep[:,feature])
                lossGluon +=  loss_fn(logScaled_partons[:,3,feature], gluon[:,feature])
            # case when feature is equal to phi (phi is periodic variable)
            else:
                lossH += loss_fn_periodic(higgs[:,feature]*scaling_phi[1] + scaling_phi[0],
                                          logScaled_partons[:,0,feature]*scaling_phi[1] + scaling_phi[0], loss_fn, device)
                lossThad +=  loss_fn_periodic(thad[:,feature]*scaling_phi[1] + scaling_phi[0],
                                              logScaled_partons[:,1,feature]*scaling_phi[1] + scaling_phi[0], loss_fn, device)
                lossTlep +=  loss_fn_periodic(tlep[:,feature]*scaling_phi[1] + scaling_phi[0],
                                              logScaled_partons[:,2,feature]*scaling_phi[1] + scaling_phi[0], loss_fn, device)
                lossGluon +=  loss_fn_periodic(gluon[:,feature]*scaling_phi[1] + scaling_phi[0],
                                               logScaled_partons[:,3,feature]*scaling_phi[1] + scaling_phi[0], loss_fn, device)

        lossBoost = loss_fn(logScaled_boost, boost)

    if split:
        return lossH, lossThad, lossTlep, lossGluon, lossBoost
    else:
        return  (lossH + lossThad + lossTlep + lossGluon + lossBoost)/13


def compute_mmd_regr_loss(mmd_input, mmd_target, kernel, device, dtype, total=False, split=False):
    mmds = []
    for particle in range(len(mmd_input)):
        mmds.append(MMD(mmd_input[particle], mmd_target[particle], kernel, device, dtype))
    # total MMD
    if total:
        mmds.append(MMD(torch.cat(mmd_input, dim=1), torch.cat(mmd_target, dim=1), kernel, device, dtype))

    if split:
        return mmds
    else:
        return sum(mmds)/len(mmds)

def compute_mmd_ps_samples(ps_samples, ps_target_scaled, kernel, device, dtype, is_sampling_epoch=True):
    if is_sampling_epoch:
        return MMD(ps_samples, ps_target_scaled, kernel, device, dtype)
    else:
        return 0.
    

def train( device, name_dir, config,  outputDir, dtype,
           world_size=None, device_ids=None):
    # device is device when not distributed and rank when distributed
    print("START OF RANK:", device)
    if world_size is not None:
        ddp_setup(device, world_size, config.ddp_port)

    device_id = device_ids[device] if device_ids is not None else device

 
    modelName = f"{name_dir}/model_{config.name}_{config.version}.pt"

    #print("Loading datasets")
    train_dataset = DatasetCombined(config.input_dataset_train,dev=device,
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

    val_dataset = DatasetCombined(config.input_dataset_validation,dev=device,
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

    log_mean_parton = train_dataset.partons_lab.mean_log_data_higgs_t_tbar_ISR
    log_std_parton = train_dataset.partons_lab.std_log_data_higgs_t_tbar_ISR
    log_mean_boost = train_dataset.partons_lab.mean_log_data_boost
    log_std_boost = train_dataset.partons_lab.std_log_data_boost
    mean_ps = train_dataset.partons_CM.mean_phasespace_intermediateParticles_onShell_logit
    scale_ps = train_dataset.partons_CM.std_phasespace_intermediateParticles_onShell_logit

    # Initialize model
    model = UnfoldingFlow(
                    pretrained_model=config.conditioning_transformer.weights,
                    load_conditioning_model=config.unfolding_flow.load_conditioning_model,
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

    # Experiment logging
    if device == 0 or world_size is None:
        # Loading comet_ai logging
        exp = Experiment(
            api_key=config.comet_token,
            project_name="memflow",
            workspace="valsdav",
            auto_output_logging = "simple",
            # disabled=True
        )
        exp.add_tags([config.name,config.version])
        exp.log_parameters(config.training_params)
        exp.log_parameters(config.conditioning_transformer)
        exp.log_parameters(config.unfolding_flow)
        exp.log_parameters(config.MDMM)
        exp.log_parameters({"model_param_tot":count_parameters(model)})
        exp.log_parameters({"model_param_conditioner":count_parameters(model.cond_transformer)})
        exp.log_parameters({"model_param_flow":count_parameters(model.flow)})
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
        #pin_memory=True,
        # collate_fn=my_collate, # use custom collate function here
        #pin_memory=True,
    )
    validLoader = DataLoader(
        val_dataset,
        batch_size=config.training_params.batch_size_training,
        shuffle=False,
        # collate_fn=my_collate,
        
    )

    # Constraints
    constraint_regr_huber = mdmm.MaxConstraint(
        compute_regr_losses,
        max=config.MDMM.huber_max, # to be modified based on the regression
        scale=config.MDMM.huber_scale,
        damping=config.MDMM.huber_damping,
    )
    constraint_regr_mmd = mdmm.MaxConstraint(
        compute_mmd_regr_loss,
        max=config.MDMM.mmd_regr_max, # to be modified based on the regression
        scale=config.MDMM.mmd_regr_scale,
        damping=config.MDMM.mmd_regr_damping,
    )

    constraint_ps_samples = mdmm.MaxConstraint(
        compute_mmd_ps_samples,
        max=config.MDMM.mmd_ps_samples_max,
        scale=config.MDMM.mmd_ps_samples_scale,
        damping=config.MDMM.mmd_ps_samples_damping
    )
    # constraint_ps_regr_huber = mdmm.MaxConstraint(
    #                 compute_ps_regr_loss,
    #                 max=config.MDMM.ps_regr_max, # to be modified based on the regression
    #                 scale=config.MDMM.ps_regr_scale,
    #                 damping=config.MDMM.ps_regr_damping,
    #                 )
    # constraint_ps_regr_mmd = mdmm.MaxConstraint(
    #                 compute_mmd_ps_regr_loss,
    #                 max=config.MDMM.ps_mmd_max, # to be modified based on the regression
    #                 scale=config.MDMM.ps_mmd_scale,
    #                 damping=config.MDMM.ps_mmd_damping,
    #                 ) 

    # Create the optimizer
    if dtype == torch.float32:
        MDMM_module = mdmm.MDMM([constraint_regr_huber,
                                 constraint_regr_mmd,
                                 constraint_ps_samples]).float() # support many constraints TODO: use a constraint for every particle
    else:
        MDMM_module = mdmm.MDMM([constraint_regr_huber,
                                 constraint_regr_mmd,
                                 constraint_ps_samples])
 
    loss_fn = torch.nn.HuberLoss(delta=config.training_params.huber_delta)

    # optimizer = optim.Adam(list(model.parameters()) , lr=config.training_params.lr)
    optimizer = MDMM_module.make_optimizer(model.parameters(), lr=config.training_params.lr)
    for gr in optimizer.param_groups:
        gr["initial_lr"] = config.training_params.lr

    scheduler_type = config.training_params.scheduler
    
    if scheduler_type == "cosine_scheduler":
        scheduler = CosineAnnealingLR(optimizer,
                                  **config.training_params.cosine_scheduler)
    elif scheduler_type == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               verbose=True,
                                                               **config.training_params.reduce_on_plateau)
    elif scheduler_type == "cyclic_lr":
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                     step_size_down=None,
                                                     cycle_momentum=False,
                                                     **config.training_params.cyclic_lr
                                                     )
        
    
    early_stopper = EarlyStopper(patience=config.training_params.nEpochsPatience,
                                 min_delta=config.training_params.get("minDeltaPatience", 1e-4))

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
        # Sampling loss on epoch 2
        sampling_training = e % 2 == 0 
        
        # training loop    
        print("Before training loop")
        ddp_model.train()
        
        for i, data_batch in enumerate(trainingLoader):
            N_train += 1
            ii+=1


            optimizer.zero_grad()
            
            (logScaled_partons, logScaled_boost,
             logScaled_reco, mask_lepton_reco, 
             mask_jets, mask_met, 
             mask_boost_reco, data_boost_reco,
             ps_target, ps_target_scaled) = data_batch

                
            mask_recoParticles = torch.cat((mask_jets, mask_lepton_reco, mask_met), dim=1)
            if True : # Always remove true proveance TODO REMOVE THE HACK 
                logScaled_reco = logScaled_reco[:,:,:-1]

            if sampling_training:
                # The provenance is remove in the model
                (data_regressed, data_regressed_cm,
                 ps_regr, logit_ps_regr, flow_cond_vector,
                 flow_samples, mask_problematic)   = ddp_model(logScaled_reco,
                                                               data_boost_reco,
                                                               mask_recoParticles,
                                                               mask_boost_reco,
                                                               ps_target_scaled,
                                                               disableGradTransformer=False,
                                                               flow_eval="sampling",
                                                               Nsamples=1)
                flow_samples = flow_samples.squeeze(0)
                
            else:
                # The provenance is remove in the model
                (data_regressed, data_regressed_cm,
                 ps_regr, logit_ps_regr, flow_cond_vector,
                 flow_logprob, mask_problematic)   = ddp_model(logScaled_reco,
                                                               data_boost_reco,
                                                               mask_recoParticles,
                                                               mask_boost_reco,
                                                               ps_target_scaled,
                                                               disableGradTransformer=False,
                                                               flow_eval="normalizing")

            
            higgs = data_regressed[0] 
            thad = data_regressed[1]
            tlep = data_regressed[2]
            gluon = data_regressed[3]
            boost = data_regressed[4]
                        
            MMD_input = data_regressed
            MMD_target = [logScaled_partons[:,0], logScaled_partons[:,1],
                          logScaled_partons[:,2], logScaled_partons[:,3],
                          logScaled_boost]

            if sampling_training:
                inf_mask = torch.isinf(flow_samples) | torch.isnan(flow_samples)
                # In the sampling direction the main loss is a distance between the sample and the target
                loss_main = loss_fn(flow_samples, ps_target_scaled)
            else:
                inf_mask = torch.isinf(flow_logprob) | torch.isnan(flow_logprob)
                loss_main = -flow_logprob[(~inf_mask) & (~mask_problematic)].mean()
                flow_samples = None
                    
            mdmm_return = MDMM_module(loss_main, [
                # huberloos
                (logScaled_partons,
                 logScaled_boost,
                 higgs, thad,
                 tlep, gluon, boost,
                 config.cartesian, loss_fn,
                 [log_mean_parton[2], log_std_parton[2]],   #scaling phi parton lab
                 device,
                 False #Split
                 ),
                #MMD rereg
                (MMD_input, MMD_target,
                 config.training_params.mmd_kernel,
                 device, dtype, True, False),
                # PS regr scaled
                # (logit_ps_regr,
                #  ps_target_scaled,
                #  loss_fn),
                # # PS regr mmd
                # (logit_ps_regr,
                #  ps_target_scaled,
                #  config.training_params.mmd_kernel,
                #  device, dtype),

                # PS samples mmd loss
                (flow_samples, ps_target_scaled,
                 config.training_params.mmd_kernel,
                 device, dtype, sampling_training)
            ])

            # Use the mdmm constraints on the loss
            loss_final = mdmm_return.value

            loss_final.backward()
            optimizer.step()
            sum_loss += loss_final.item()

            # Scheduler step
            if scheduler_type == "cyclic_lr": #cycle each step
                scheduler.step()

            if scheduler_type == "cosine_scheduler":
                after_N_epochs = config.training_params.cosine_scheduler.get("after_N_epochs", 0)
                if e > after_N_epochs:
                    scheduler.step()

            with torch.no_grad():
                if exp is not None and device==0 or world_size is None:
                    if i % config.training_params.interval_logging_steps == 0:
                        (mmd_loss_H, mmd_loss_thad,
                         mmd_loss_tlep,
                         mmd_loss_gluon,
                         mmd_loss_boost,
                         mmd_loss_all) = compute_mmd_regr_loss(MMD_input, MMD_target,
                                                               kernel=config.training_params.mmd_kernel,
                                                               device=device,total=True, dtype=dtype, split=True)
                        
                        lossH, lossThad, lossTlep, lossGluon, lossBoost = compute_regr_losses(logScaled_partons,
                                                                                  logScaled_boost,
                                                                                  higgs, thad,
                                                                                  tlep, gluon, boost,
                                                                                  config.cartesian, loss_fn,
                                                                                  scaling_phi=[log_mean_parton[2], log_std_parton[2]], # Added scaling for phi
                                                                                  split=True,
                                                                                  device=device)
                        # loss_huber_PS_cm = compute_ps_regr_loss(logit_ps_regr,
                        #                      ps_target_scaled,
                        #                      loss_fn)
                    
                        

                        # loss_mmd_PS_cm = compute_mmd_ps_regr_loss(logit_ps_regr,
                        #                                           ps_target_scaled,
                        #                                           config.training_params.mmd_kernel,
                        #                                           device, dtype)

                        # total loss is split in two
                        if sampling_training:
                            exp.log_metric('loss_tot_train_sampling', loss_final.item(),step=ii) # including mmd
                            exp.log_metric('loss_flow_sampling', loss_main.item(), step=ii)
                            # compute mmd sampling
                            mmd_loss_samples = compute_mmd_ps_samples(flow_samples, ps_target_scaled,
                                                   config.training_params.mmd_kernel,
                                                   device, dtype, is_sampling_epoch=True)
                            exp.log_metric("loss_mmd_samples", mmd_loss_samples.item(), step=ii)
                            
                        else:
                            exp.log_metric('loss_tot_train_density', loss_final.item(),step=ii) # including mmd
                            exp.log_metric('loss_flow_density', loss_main.item(), step=ii)
                        
                        
                        exp.log_metric("loss_mmd_H", mmd_loss_H.item(),step=ii)
                        exp.log_metric('loss_mmd_thad', mmd_loss_thad.item(),step=ii)
                        exp.log_metric('loss_mmd_tlep', mmd_loss_tlep.item(),step=ii)
                        exp.log_metric('loss_mmd_gluon', mmd_loss_gluon.item(),step=ii)
                        exp.log_metric('loss_mmd_boost', mmd_loss_boost.item(),step=ii)
                        exp.log_metric('loss_mmd_all', mmd_loss_all.item(),step=ii)
                        # exp.log_metric('loss_mmd_PSregr', loss_mmd_PS_cm.item(), step=ii)

                        exp.log_metric('loss_mmd_tot', (mmd_loss_H+mmd_loss_thad + mmd_loss_tlep + mmd_loss_boost + mmd_loss_gluon +  mmd_loss_all).item()/6,step=ii)
                        exp.log_metric('loss_huber_H', lossH.item()/3,step=ii)
                        exp.log_metric('loss_huber_Thad', lossThad.item()/3,step=ii)
                        exp.log_metric('loss_huber_Tlep', lossTlep.item()/3,step=ii)
                        exp.log_metric('loss_huber_gluon', lossGluon.item()/3,step=ii)
                        exp.log_metric('loss_huber_boost', lossBoost.item(),step=ii)
                        exp.log_metric('loss_huber_tot', (lossH + lossThad+ lossTlep + lossGluon+lossBoost).item()/13,step=ii)
                        # exp.log_metric('loss_huber_PSregr', loss_huber_PS_cm.item(), step=ii)
                        
                        exp.log_metric('flow_ninf_train', inf_mask.sum(), step=ii)
                        exp.log_metric('flow_nproblems', mask_problematic.sum(), step=ii )
                        # Log the average difference of ps points
                        exp.log_metric('train_PSregr_avgdiff', (logit_ps_regr - ps_target_scaled).abs().mean(), step=ii)
                        exp.log_metric('train_PSregr_stddiff', (logit_ps_regr - ps_target_scaled).std().mean(), step=ii)
                        exp.log_metric('train_PSregr_mmd', MMD(logit_ps_regr, ps_target_scaled, config.training_params.mmd_kernel, device, dtype), step=ii)
                        
                        exp.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=ii)

        ### END of training 
        if exp is not None and device==0 or world_size is None:
            exp.log_metric("loss_epoch_total_train", sum_loss/N_train, epoch=e, step=ii)
            # if scheduler_type == "cyclic_lr":
            #     exp.log_metric("learning_rate", scheduler.get_last_lr(), epoch=e, step=ii)

        
        valid_loss_huber = 0.
        valid_loss_mmd = 0.
        valid_lossH = 0.
        valid_lossTlep = 0.
        valid_lossThad = 0.
        valid_lossGluon = 0.
        valid_lossBoost = 0.
        valid_loss_final = 0.
        valid_mmd_H = 0.
        valid_mmd_thad = 0.
        valid_mmd_tlep = 0.
        valid_mmd_gluon= 0.
        valid_mmd_boost = 0.
        valid_mmd_all = 0.
        valid_loss_flow = 0.
        valid_ninf_flow = 0
        valid_nproblems = 0
        # valid_huber_PS_cm = 0.
        # valid_mmd_PS_cm = 0.
        valid_mmd_ps_samples = 0.

        
        # validation loop (don't update weights and gradients)
        print("Before validation loop")
        ddp_model.eval()
        
        for i, data_batch in enumerate(validLoader):
            N_valid +=1
            # Move data to device
            with torch.no_grad():

                (logScaled_partons, logScaled_boost,
                 logScaled_reco, mask_lepton_reco, 
                 mask_jets, mask_met, 
                 mask_boost_reco, data_boost_reco,
                 ps_target, ps_target_scaled) = data_batch
                
                mask_recoParticles = torch.cat((mask_jets, mask_lepton_reco, mask_met), dim=1)

                # remove prov
                if True:
                    logScaled_reco = logScaled_reco[:,:,:-1]

                if sampling_training:
                    # The provenance is remove in the model
                    (data_regressed, data_regressed_cm,
                     ps_regr, logit_ps_regr, flow_cond_vector,
                     flow_samples, mask_problematic)   = ddp_model(logScaled_reco,
                                                                   data_boost_reco,
                                                                   mask_recoParticles,
                                                                   mask_boost_reco,
                                                                   ps_target_scaled,
                                                                   disableGradTransformer=False,
                                                                   flow_eval="sampling",
                                                                   Nsamples=1)
                    flow_samples = flow_samples.squeeze(0)

                else:
                    # The provenance is remove in the model
                    (data_regressed, data_regressed_cm,
                     ps_regr, logit_ps_regr, flow_cond_vector,
                     flow_logprob, mask_problematic)   = ddp_model(logScaled_reco,
                                                                   data_boost_reco,
                                                                   mask_recoParticles,
                                                                   mask_boost_reco,
                                                                   ps_target_scaled,
                                                                   disableGradTransformer=False,
                                                                   flow_eval="normalizing")

                    
                higgs = data_regressed[0] 
                thad = data_regressed[1]
                tlep = data_regressed[2]
                gluon = data_regressed[3]
                boost = data_regressed[4]
            
                MMD_input = data_regressed
                MMD_target = [logScaled_partons[:,0], logScaled_partons[:,1],
                              logScaled_partons[:,2], logScaled_partons[:,3],
                              logScaled_boost]

                if sampling_training:
                    inf_mask = torch.isinf(flow_samples) | torch.isnan(flow_samples)
                    # In the sampling direction the main loss is a distance between the sample and the target
                    loss_main = loss_fn(flow_samples, ps_target_scaled)
                else:
                    inf_mask = torch.isinf(flow_logprob) | torch.isnan(flow_logprob)
                    loss_main = -flow_logprob[(~inf_mask) & (~mask_problematic)].mean()
                    flow_samples = None

                
                lossH, lossThad, lossTlep, lossGluon, lossBoost = compute_regr_losses(logScaled_partons,
                                                                                      logScaled_boost,
                                                                                      higgs, thad, tlep,gluon, boost,
                                                                                      config.cartesian, loss_fn,
                                                                                      scaling_phi=[log_mean_parton[2], log_std_parton[2]], # Added scaling for phi
                                                                                      split=True,
                                                                                      device=device)

                regr_loss =  (lossH + lossThad + lossTlep+lossGluon + lossBoost)/13

                # loss_huber_PS_cm = compute_ps_regr_loss(logit_ps_regr,
                #                              ps_target_scaled,
                #                              loss_fn)

                # mmd_loss_PS_cm = compute_mmd_ps_regr_loss(logit_ps_regr,
                #                                           ps_target_scaled,
                #                                           config.training_params.mmd_kernel,
                #                                           device, dtype)
                
                (mmd_loss_H,
                 mmd_loss_thad,
                 mmd_loss_tlep,
                 mmd_loss_gluon,
                 mmd_loss_boost,
                 mmd_loss_all)= compute_mmd_regr_loss(MMD_input, MMD_target,
                            kernel=config.training_params.mmd_kernel,
                                                   device=device,
                                                   total=True,
                                                   dtype=dtype,
                                                   split=True)
                
                mmd_loss = (mmd_loss_H+ mmd_loss_thad + mmd_loss_tlep + mmd_loss_gluon +  mmd_loss_boost + mmd_loss_all)/6


                if sampling_training:
                    # compute mmd sampling
                    mmd_loss_samples = compute_mmd_ps_samples(flow_samples, ps_target_scaled,
                                                              config.training_params.mmd_kernel,
                                                              device, dtype, is_sampling_epoch=True)
                    valid_mmd_ps_samples += mmd_loss_samples.item()
                                
                valid_loss_flow += loss_main.item()
                valid_ninf_flow += inf_mask.sum()
                valid_nproblems += mask_problematic.sum()
                valid_loss_huber += regr_loss.item()
                valid_lossH += lossH.item()/3
                valid_lossTlep += lossTlep.item()/3
                valid_lossThad += lossThad.item()/3
                valid_lossBoost += lossBoost.item()
                valid_lossGluon += lossGluon.item()/3
                #valid_huber_PS_cm += loss_huber_PS_cm.item()
                valid_loss_mmd += mmd_loss.item()
                valid_mmd_H += mmd_loss_H.item()
                valid_mmd_thad += mmd_loss_thad.item()
                valid_mmd_tlep += mmd_loss_tlep.item()
                valid_mmd_gluon += mmd_loss_gluon.item()
                valid_mmd_boost += mmd_loss_boost.item()
                # valid_mmd_PS_cm += mmd_loss_PS_cm.item()
                valid_mmd_all += mmd_loss_all.item()
                
                valid_loss_final += loss_main.item()   # using only the main loss, not MDMM

                particle_list = [higgs, thad, tlep, gluon]
                
                if i == 0 and ( exp is not None and device==0 or world_size is None):
                    for particle in range(len(particle_list)): # 4 or 3 particles: higgs/thad/tlep/gluonISR
                        
                        # 4 or 3 features
                        for feature in range(3):  
                            fig, ax = plt.subplots(figsize=(7,6), dpi=100)
                            h = ax.hist2d(logScaled_partons[:,particle,feature].detach().cpu().numpy(),
                                          particle_list[particle][:,feature].cpu().detach().numpy().flatten(),
                                          bins=30, range=((-2.5, 2.5),(-2.5, 2.5)), cmin=1)
                            fig.colorbar(h[3], ax=ax)
                            exp.log_figure(f"particle_2D_{particle}_{feature}", fig,step=e)

                    
                            fig, ax = plt.subplots(figsize=(7,6), dpi=100)
                            ax.hist(logScaled_partons[:,particle,feature].detach().cpu().numpy(),
                                          bins=30, range=(-2, 2), label="truth", histtype="step")
                            ax.hist(particle_list[particle][:,feature].cpu().detach().numpy().flatten(),
                                          bins=30, range=(-2, 2), label="regressed",histtype="step")
                            ax.legend()
                            ax.set_xlabel(f"particle {particle} feature {feature}")
                            exp.log_figure(f"particle_1D_{particle}_{feature}", fig, step=e)

                    ranges_boost = [(-3,3),(-2,2)]
                    for feature in range(2):
                        fig, ax = plt.subplots(figsize=(7,6), dpi=100)
                        h = ax.hist2d(logScaled_boost[:,feature].detach().cpu().numpy(),
                                      boost[:,feature].cpu().detach().numpy().flatten(),
                                      bins=30, range=(ranges_boost[feature],ranges_boost[feature] ), cmin=1)
                        fig.colorbar(h[3], ax=ax)
                        exp.log_figure(f"boost_2D_{feature}", fig,step=e)


                        fig, ax = plt.subplots(figsize=(7,6), dpi=100)
                        ax.hist(logScaled_boost[:,feature].detach().cpu().numpy(),
                                      bins=30, range=ranges_boost[feature], label="truth", histtype="step")
                        ax.hist(boost[:,feature].cpu().detach().numpy().flatten(),
                                      bins=30, range=ranges_boost[feature], label="regressed",histtype="step")
                        ax.legend()
                        ax.set_xlabel(f"boost feature {feature}")
                        exp.log_figure(f"boost_1D_{feature}", fig, step=e)

                    # Plots for the flow quality
                    # plots of regressed ps
                    for k in range(10):
                        fig, ax = plt.subplots(figsize=(7,6), dpi=100)
                        h = ax.hist2d(ps_target_scaled[:,k].cpu().numpy(),
                                      logit_ps_regr[:,k].cpu().numpy(),
                                      bins=40, range=((-0.75, 0.75),(-0.75,0.75)), cmin=1)
                        fig.colorbar(h[3], ax=ax)
                        exp.log_figure(f"ps_regr_CM_2D_{k}", fig,step=e)


                    # Sampling
                    N_samples = config.training_params.sampling_points
                    ps_samples = model.flow(flow_cond_vector).sample((N_samples,))
                    for j in range(10):
                        fig, ax = plt.subplots(figsize=(7,6), dpi=100)
                        h = ax.hist2d(ps_target_scaled[:,j].tile(N_samples,1,1).flatten().cpu().numpy(),
                                        ps_samples[:,:,j].flatten().cpu().numpy(),
                                        bins=50, range=((-1, 1),(-1, 1)),cmin=1)
                        fig.colorbar(h[3], ax=ax)
                        exp.log_figure(f"ps_sampled_2D_{j}", fig, step=e)

                        fig, ax = plt.subplots(figsize=(7,6), dpi=100)
                        h = ax.hist2d(ps_target_scaled[:,j].tile(N_samples,1,1).flatten().cpu().numpy(),
                                        ps_samples[:,:,j].flatten().cpu().numpy(),
                                        bins=50, range=((-1, 1),(-1, 1)), norm=LogNorm() )
                        fig.colorbar(h[3], ax=ax)
                        exp.log_figure(f"ps_sampled_2D_{j}_log", fig, step=e)

                        fig, ax = plt.subplots(figsize=(7,6), dpi=100)
                        h = ax.hist(
                            (ps_target_scaled[:,j].tile(N_samples,1,1) - ps_samples[:,:,j]).flatten().cpu().numpy(),
                            range=(-1,1),   bins=50)
                        exp.log_figure(f"ps_1D_{j}", fig, step=e)

                        # Correlation coefficiency for each sampled PS
                        # Just take the first sample
                        corr= torch.corrcoef(torch.stack((ps_samples[0,:,j], ps_target_scaled[:,j])))[0,1]
                        exp.log_metric(f"val_ps_sampled_corrcoef_{j}", corr, epoch=e)

                    

        if exp is not None and device==0 or world_size is None:
            if sampling_training:
                exp.log_metric("loss_tot_val_sampling", valid_loss_final/N_valid, epoch=e )
            else:
                exp.log_metric("loss_tot_val_density", valid_loss_final/N_valid, epoch=e )

            
            exp.log_metric("flow_ninf_val", valid_ninf_flow, epoch=e)
            exp.log_metric("flow_nproblems", valid_nproblems, epoch=e)
            exp.log_metric("loss_mmd_val", valid_loss_mmd/N_valid,epoch=e)
            exp.log_metric('loss_huber_val', valid_loss_huber/N_valid,epoch= e)
            exp.log_metric('loss_huber_val_H', valid_lossH/N_valid,epoch= e)
            exp.log_metric('loss_huber_val_Tlep', valid_lossTlep/N_valid,epoch= e)
            exp.log_metric('loss_huber_val_Thad', valid_lossThad/N_valid,epoch= e)
            exp.log_metric('loss_huber_val_Gluon', valid_lossGluon/N_valid,epoch= e)
            exp.log_metric('loss_huber_val_boost', valid_lossBoost/N_valid,epoch= e)
            # exp.log_metric('loss_huber_val_PS_cm', valid_huber_PS_cm/N_valid, epoch=e)
            exp.log_metric("loss_mmd_val_H", valid_mmd_H/N_valid,epoch= e)
            exp.log_metric("loss_mmd_val_thad", valid_mmd_thad/N_valid,epoch= e)
            exp.log_metric("loss_mmd_val_tlep", valid_mmd_tlep/N_valid,epoch= e)
            exp.log_metric("loss_mmd_val_gluon", valid_mmd_gluon/N_valid,epoch= e)
            # exp.log_metric('loss_mmd_val_PS_cm', valid_mmd_PS_cm/N_valid, epoch=e)
            exp.log_metric("loss_mmd_val_boost", valid_mmd_boost/N_valid,epoch= e)
            exp.log_metric("loss_mmd_val_all", valid_mmd_all/N_valid,epoch= e)
            exp.log_metric('val_PSregr_avgdiff', (logit_ps_regr - ps_target_scaled).abs().mean(), step=ii)
            exp.log_metric('val_PSregr_stddiff', (logit_ps_regr - ps_target_scaled).abs().std(), step=ii)
            exp.log_metric('val_PSregr_mmd', MMD(logit_ps_regr, ps_target_scaled, config.training_params.mmd_kernel, device, dtype), step=ii)

        if device == 0 or world_size is None:
            if early_stopper.early_stop(valid_loss_final/N_valid,
                                    model.state_dict(), optimizer.state_dict(), modelName, exp):
                print(f"Model converges at epoch {e} !!!")         
                break

        # Step the scheduler at the end of the val
        if scheduler_type == "reduce_on_plateau":
            # Step the scheduler at the end of the val
            scheduler.step(valid_loss_final/N_valid)

        

    # writer.close()
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
    name_dir = f'{outputDir}/flow_{conf.name}_{conf.version}_{conf.unfolding_flow.base}_NoTransf{conf.unfolding_flow.ntransforms}_NoBins{conf.unfolding_flow.bins}_DNN{conf.unfolding_flow.hiddenMLP_NoLayers}_{conf.unfolding_flow.hiddenMLP_LayerDim}'
    

    os.makedirs(name_dir, exist_ok=True)
    
    with open(f"{name_dir}/config_{conf.name}_{conf.version}.yaml", "w") as fo:
        fo.write(OmegaConf.to_yaml(conf)) 

    if conf.training_params.dtype == "float32":
        dtype = torch.float32
    elif conf.training_params.dtype == "float64":
        dtype = torch.float64
        
    
    if len(actual_devices) >1 and args.distributed:
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
    
    print(f"Flow training finished succesfully! Version: {conf.version}")
    
    
    
    
    
    
    
    
    
