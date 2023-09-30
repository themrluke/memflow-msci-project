from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

import torch
from memflow.read_data.dataset_all import DatasetCombined

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

    # rescale to pi
    # overflow_delta = (inp[inp>PI]- PI).mean()+(inp[inp<-PI]+PI).mean()
    inp[inp>PI] = inp[inp>PI]-2*PI
    inp[inp<-PI] = inp[inp<-PI] + 2*PI
    
    deltaPhi = target - inp
    deltaPhi = torch.where(deltaPhi > PI, deltaPhi - 2*PI, deltaPhi)
    deltaPhi = torch.where(deltaPhi <= -PI, deltaPhi + 2*PI, deltaPhi)
    
    return loss_fn(deltaPhi, torch.zeros(deltaPhi.shape, device=device))# + overflow_delta*0.5


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


def compute_mmd_loss(mmd_input, mmd_target, kernel, device, dtype, total=False, split=False):
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

    

def train( device, name_dir, config,  outputDir, dtype,
           world_size=None, device_ids=None):
    # device is device when not distributed and rank when distributed
    print("START OF RANK:", device)
    if world_size is not None:
        ddp_setup(device, world_size, config.ddp_port)

    device_id = device_ids[device] if device_ids is not None else device

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
        exp.log_parameters(config.MDMM) 
    else:
        exp = None

    model = ConditioningTransformerLayer(no_jets = config.input_shape.number_jets,
                                    no_lept = config.input_shape.number_lept,
                                    input_features=config.input_shape.input_features, 
                                    hidden_features=config.conditioning_transformer.hidden_features,
                                    dim_feedforward_transformer= config.conditioning_transformer.dim_feedforward_transformer,
                                    out_features=config.conditioning_transformer.out_features,
                                    nhead_encoder=config.conditioning_transformer.nhead_encoder,
                                    no_layers_encoder=config.conditioning_transformer.no_layers_encoder,
                                    nhead_decoder=config.conditioning_transformer.nhead_decoder,
                                    no_layers_decoder=config.conditioning_transformer.no_layers_decoder,
                                    no_decoders=config.conditioning_transformer.no_decoders,
                                    aggregate=config.conditioning_transformer.aggregate,
                                    use_latent=config.conditioning_transformer.use_latent,
                                    out_features_latent=config.conditioning_transformer.out_features_latent,
                                    no_layers_decoder_latent=config.conditioning_transformer.no_layers_decoder_latent,     
                                         dtype=dtype) 

    model = model.to(device)
    model.disable_latent_training()

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

    #print("Loading datasets")
    train_dataset = DatasetCombined(config.input_dataset_train,dev=device,
                           dtype=dtype, boost_CM=False,
                           reco_list=['scaledLogRecoParticles', 'mask_lepton', 
                                      'mask_jets','mask_met',
                                      'mask_boost', 'scaledLogBoost'],
                           parton_list=['logScaled_data_higgs_t_tbar_ISR',
                                        'logScaled_data_boost',
                                        'mean_log_data_higgs_t_tbar_ISR',
                                        'std_log_data_higgs_t_tbar_ISR',
                                        'mean_log_data_boost',
                                        'std_log_data_boost'])

    val_dataset = DatasetCombined(config.input_dataset_validation,dev=device,
                           dtype=dtype, boost_CM=False,
                           reco_list=['scaledLogRecoParticles', 'mask_lepton', 
                                      'mask_jets','mask_met',
                                      'mask_boost', 'scaledLogBoost'],
                           parton_list=['logScaled_data_higgs_t_tbar_ISR',
                                        'logScaled_data_boost',
                                        'mean_log_data_higgs_t_tbar_ISR',
                                        'std_log_data_higgs_t_tbar_ISR',
                                        'mean_log_data_boost',
                                        'std_log_data_boost'])

    log_mean_parton = train_dataset.parton_data.mean_log_data_higgs_t_tbar_ISR
    log_std_parton = train_dataset.parton_data.std_log_data_higgs_t_tbar_ISR
    log_mean_boost = train_dataset.parton_data.mean_log_data_boost
    log_std_boost = train_dataset.parton_data.std_log_data_boost

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
        
    constraint = mdmm.MaxConstraint(
                    compute_mmd_loss,
                    config.MDMM.max_mmd, # to be modified based on the regression
                    scale=config.MDMM.scale_mmd,
                    damping=config.MDMM.dumping_mmd,
                    )

    # Create the optimizer
    if dtype == torch.float32:
        MDMM_module = mdmm.MDMM([constraint]).float() # support many constraints TODO: use a constraint for every particle
    else:
        MDMM_module = mdmm.MDMM([constraint])

    loss_fn = torch.nn.HuberLoss(delta=config.training_params.huber_delta)

    # optimizer = optim.Adam(list(model.parameters()) , lr=config.training_params.lr)
    optimizer = MDMM_module.make_optimizer(model.parameters(), lr=config.training_params.lr)
    # optimizer = optim.Rprop(list(model.parameters()) , lr=config.training_params.lr)
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=config.training_params.cosine_scheduler.Tmax,
                                  eta_min=config.training_params.cosine_scheduler.eta_min)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                       factor=config.training_params.reduce_on_plateau.factor,
    #                                                       patience=config.training_params.reduce_on_plateau.patience,
    #                                                       threshold=config.training_params.reduce_on_plateau.threshold, verbose=True)
    
    
    early_stopper = EarlyStopper(patience=config.training_params.nEpochsPatience, min_delta=0.0001)

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
            
            (logScaled_partons, logScaled_boost,
            logScaled_reco, mask_lepton_reco, 
            mask_jets, mask_met, 
            mask_boost_reco, data_boost_reco) = data_batch
            
            mask_recoParticles = torch.cat((mask_jets, mask_lepton_reco, mask_met), dim=1)
            # remove prov
            if (config.noProv):
                logScaled_reco = logScaled_reco[:,:,:-1]

            out = ddp_model(logScaled_reco, data_boost_reco, mask_recoParticles, mask_boost_reco)

            higgs = out[0] 
            thad = out[1]
            tlep = out[2]
            
            data_regressed, boost_regressed = Compute_ParticlesTensor.get_HttISR_fromlab_numpy(out, log_mean_parton,
                                  log_std_parton,
                                  log_mean_boost, log_std_boost,
                                  device, cartesian=False, eps=0.0)

            boost_notscaled = boost_regressed[:, [0,-1]]
            boost = boost_notscaled.clone()
            boost[:,0] = torch.log(boost_notscaled[:,0] + 1)
            boost = (boost - log_mean_boost)/log_std_boost

            gluon_toscale = data_regressed[:,3] #pt, eta, phi
            gluon = gluon_toscale.clone()
            gluon[:,0] = torch.log(gluon_toscale[:,0] + 1) # No need for abs and sign
            gluon = (gluon - log_mean_parton) / log_std_parton
                        
            MMD_input = [higgs, thad, tlep, gluon, boost]
            MMD_target = [logScaled_partons[:,0],
                          logScaled_partons[:,1],
                          logScaled_partons[:,2],
                          logScaled_partons[:,3],
                          logScaled_boost]

            lossH, lossThad, lossTlep, lossGluon, lossBoost = compute_regr_losses(logScaled_partons,
                                                                                  logScaled_boost,
                                                                                  higgs, thad,
                                                                                  tlep, gluon, boost,
                                                                                  config.cartesian, loss_fn,
                                                                                  scaling_phi=[log_mean_parton[2], log_std_parton[2]], # Added scaling for phi
                                                                                  split=True,
                                                                                  device=device)
            regr_loss =  (lossH + lossThad + lossTlep+ lossGluon+ lossBoost)/13
                    
            mdmm_return = MDMM_module(regr_loss, [(MMD_input, MMD_target, config.training_params.mmd_kernel, device, dtype, True, False)])

            loss_final = mdmm_return.value
            #print(f"MMD loss: {mdmm_return.fn_values}, huber loss {regr_loss.item()}, loss tot{loss_final.item()}")

            loss_final.backward()
            optimizer.step()

            with torch.no_grad():

                (mmd_loss_H, mmd_loss_thad,
                 mmd_loss_tlep,
                 mmd_loss_gluon,
                 mmd_loss_boost,
                 mmd_loss_all) = compute_mmd_loss(MMD_input, MMD_target,
                                                  kernel=config.training_params.mmd_kernel,
                                                  device=device,total=True, dtype=dtype, split=True)
                
                if exp is not None and device==0 or world_size is None:
                    if i % 10 == 0:
                        exp.log_metric("loss_mmd_H", mmd_loss_H.item(),step=ii)
                        exp.log_metric('loss_mmd_thad', mmd_loss_thad.item(),step=ii)
                        exp.log_metric('loss_mmd_tlep', mmd_loss_tlep.item(),step=ii)
                        exp.log_metric('loss_mmd_gluon', mmd_loss_gluon.item(),step=ii)
                        exp.log_metric('loss_mmd_boost', mmd_loss_boost.item(),step=ii)
                        exp.log_metric('loss_mmd_all', mmd_loss_all.item(),step=ii)
                        exp.log_metric('loss_mmd_tot', (mmd_loss_H+mmd_loss_thad + mmd_loss_tlep + mmd_loss_boost + mmd_loss_gluon +  mmd_loss_all).item()/6,step=ii)
                        exp.log_metric('loss_huber_H', lossH.item()/3,step=ii)
                        exp.log_metric('loss_huber_Thad', lossThad.item()/3,step=ii)
                        exp.log_metric('loss_huber_Tlep', lossTlep.item()/3,step=ii)
                        exp.log_metric('loss_huber_gluon', lossGluon.item()/3,step=ii)
                        exp.log_metric('loss_huber_boost', lossBoost.item(),step=ii)
                        exp.log_metric('loss_huber_tot', regr_loss.item(),step=ii)
                        exp.log_metric('loss_tot_train', loss_final.item(),step=ii)
                
                sum_loss += loss_final.item()

        ### END of training 
        if exp is not None and device==0 or world_size is None:
            exp.log_metric("loss_epoch_total_train", sum_loss/N_train, epoch=e, step=ii)
            exp.log_metric("learning_rate", optimizer.param_groups[0]['lr'], epoch=e, step=ii)

        
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
                mask_boost_reco, data_boost_reco) = data_batch
                
                mask_recoParticles = torch.cat((mask_jets, mask_lepton_reco, mask_met), dim=1)

                # remove prov
                if (config.noProv):
                    logScaled_reco = logScaled_reco[:,:,:-1]

                out = ddp_model(logScaled_reco, data_boost_reco, mask_recoParticles, mask_boost_reco)
            
                higgs = out[0]
                thad = out[1]
                tlep = out[2]

                data_regressed, boost_regressed = Compute_ParticlesTensor.get_HttISR_fromlab_numpy(out, log_mean_parton,
                                  log_std_parton,
                                  log_mean_boost, log_std_boost,
                                  device, cartesian=False, eps=0.0)

                boost = boost_regressed[:, [0,-1]]
                # boost = (torch.sign(boost)*torch.log(boost.abs()+1) - log_mean_boost)/log_std_boost
                boost_notscaled = boost_regressed[:, [0,-1]]
                boost = boost_notscaled.clone()
                boost[:,0] = torch.log(boost_notscaled[:,0] + 1)
                boost = (boost - log_mean_boost)/log_std_boost

                gluon_toscale = data_regressed[:,3] #pt, eta, phi
                gluon = gluon_toscale.clone()
                gluon[:,0] = torch.log(gluon_toscale[:,0] +1)
                gluon = (gluon - log_mean_parton) / log_std_parton
                
            
                MMD_input = [higgs, thad, tlep, gluon, boost]
                MMD_target = [logScaled_partons[:,0],
                          logScaled_partons[:,1],
                          logScaled_partons[:,2],
                          logScaled_partons[:,3],
                          logScaled_boost]

                lossH, lossThad, lossTlep, lossGluon, lossBoost = compute_regr_losses(logScaled_partons,
                                                                                      logScaled_boost,
                                                                                      higgs, thad, tlep,gluon, boost,
                                                                                      config.cartesian, loss_fn,
                                                                                      scaling_phi=[log_mean_parton[2], log_std_parton[2]], # Added scaling for phi
                                                                                      split=True,
                                                                                      device=device)

                regr_loss =  (lossH + lossThad + lossTlep+lossGluon + lossBoost)/13

                (mmd_loss_H,
                 mmd_loss_thad,
                 mmd_loss_tlep,
                 mmd_loss_gluon,
                 mmd_loss_boost,
                 mmd_loss_all)= compute_mmd_loss(MMD_input, MMD_target,
                            kernel=config.training_params.mmd_kernel,
                                                   device=device,
                                                   total=True,
                                                   dtype=dtype,
                                                   split=True)
                
                mmd_loss = (mmd_loss_H+ mmd_loss_thad + mmd_loss_tlep + mmd_loss_gluon +  mmd_loss_boost + mmd_loss_all)/6
                                
                mdmm_return = MDMM_module(regr_loss, [(MMD_input, MMD_target, config.training_params.mmd_kernel, device, dtype, True, False)])

                valid_loss_huber += regr_loss.item()
                valid_lossH += lossH.item()/3
                valid_lossTlep += lossTlep.item()/3
                valid_lossThad += lossThad.item()/3
                valid_lossBoost += lossBoost.item()
                valid_lossGluon += lossGluon.item()/3
                valid_loss_mmd += mmd_loss.item()
                valid_mmd_H += mmd_loss_H.item()
                valid_mmd_thad += mmd_loss_thad.item()
                valid_mmd_tlep += mmd_loss_tlep.item()
                valid_mmd_gluon += mmd_loss_gluon.item()
                valid_mmd_boost += mmd_loss_boost.item()
                valid_mmd_all += mmd_loss_all.item()
                valid_loss_final += regr_loss.item()   # using only the main loss, not MDMM

                particle_list = [higgs, thad, tlep, gluon]
                
                if i == 0 and ( exp is not None and device==0 or world_size is None):
                    for particle in range(len(particle_list)): # 4 or 3 particles: higgs/thad/tlep/gluonISR
                        
                        # 4 or 3 features
                        for feature in range(3):  
                            fig, ax = plt.subplots(figsize=(7,6), dpi=100)
                            h = ax.hist2d(logScaled_partons[:,particle,feature].detach().cpu().numpy(),
                                          particle_list[particle][:,feature].cpu().detach().numpy().flatten(),
                                          bins=40, range=((-3, 3),(-3, 3)), cmin=1)
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
                                      bins=40, range=(ranges_boost[feature],ranges_boost[feature] ), cmin=1)
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


        if exp is not None and device==0 or world_size is None:
            exp.log_metric("loss_total_val", valid_loss_final/N_valid, epoch=e )
            exp.log_metric("loss_mmd_val", valid_loss_mmd/N_valid,epoch= e)
            exp.log_metric('loss_huber_val', valid_loss_huber/N_valid,epoch= e)
            exp.log_metric('loss_huber_val_H', valid_lossH/N_valid,epoch= e)
            exp.log_metric('loss_huber_val_Tlep', valid_lossTlep/N_valid,epoch= e)
            exp.log_metric('loss_huber_val_Thad', valid_lossThad/N_valid,epoch= e)
            exp.log_metric('loss_huber_val_Gluon', valid_lossGluon/N_valid,epoch= e)
            exp.log_metric('loss_huber_val_boost', valid_lossBoost/N_valid,epoch= e)
            exp.log_metric("loss_mmd_val_H", valid_mmd_H/N_valid,epoch= e)
            exp.log_metric("loss_mmd_val_thad", valid_mmd_thad/N_valid,epoch= e)
            exp.log_metric("loss_mmd_val_tlep", valid_mmd_tlep/N_valid,epoch= e)
            exp.log_metric("loss_mmd_val_gluon", valid_mmd_gluon/N_valid,epoch= e)
            exp.log_metric("loss_mmd_val_boost", valid_mmd_boost/N_valid,epoch= e)
            exp.log_metric("loss_mmd_val_all", valid_mmd_all/N_valid,epoch= e)

        if device == 0 or world_size is None:
            if early_stopper.early_stop(valid_loss_final/N_valid,
                                    model.state_dict(), optimizer.state_dict(), modelName, exp):
                print(f"Model converges at epoch {e} !!!")         
                break

        # Step the scheduler at the end of the val
        scheduler.step()

        # scheduler.step(valid_loss_final/N_valid) # reduce lr if the model is not improving anymore
        

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
    latentSpace = conf.conditioning_transformer.use_latent
    name_dir = f'{outputDir}/preTraining_{conf.name}_{conf.version}_hiddenFeatures:{conf.conditioning_transformer.hidden_features}_dimFeedForward:{conf.conditioning_transformer.dim_feedforward_transformer}_nheadEnc:{conf.conditioning_transformer.nhead_encoder}_LayersEnc:{conf.conditioning_transformer.no_layers_encoder}_nheadDec:{conf.conditioning_transformer.nhead_decoder}_LayersDec:{conf.conditioning_transformer.no_layers_decoder}'

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
                    world_size, dev_dct),
            nprocs=world_size,
            # join=True
        )
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        train(device,name_dir, conf,  outputDir, dtype)
    
    print(f"preTraining finished succesfully! Version: {conf.version}")
    
    
    
    
    
    
    
    
    
