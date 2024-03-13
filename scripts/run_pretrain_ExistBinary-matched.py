from comet_ml import Experiment
#from comet_ml.integration.pytorch import log_model

import torch
from memflow.read_data.dataset_all import DatasetCombined
from memflow.pretrain_exist.pretrain_exist_binary import Classify_ExistJet
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

from torcheval.metrics.functional import multiclass_f1_score

from random import randint
PI = torch.pi

# TODO: right now logScaled_reco_sortedByProv has null_token inside
def sample_existToken(model, logScaled_reco_sortedByProv, logScaled_partons, mask_reco, device, dtype, encode_position=True):
    null_token = torch.ones((logScaled_reco_sortedByProv.shape[0], 1, 5), device=device, dtype=dtype) * -1
    null_token[:,0,0] = 0 # exist flag = 0 not -1

    # mask for the null token = True
    null_token_mask = torch.ones((mask_reco.shape[0], 1), device=device, dtype=torch.bool)

    # attach null token and update the mask for the scaling_reco_lab
    scaling_reco_lab_withNullToken = torch.cat((null_token, logScaled_reco_sortedByProv), dim=1)
    mask_reco_withNullToken = torch.cat((null_token_mask, mask_reco), dim=1)
    
    scaledLogParton_afterLin = model.gelu(model.linearDNN_parton(logScaled_partons))
    scaledLogReco_afterLin = model.gelu(model.linearDNN_reco(scaling_reco_lab_withNullToken) * mask_reco_withNullToken[..., None])
        
    output_decoder = model.transformer_model(scaledLogParton_afterLin, scaledLogReco_afterLin, tgt_mask=model.tgt_mask)

    if encode_position:
        hot_encoded = [i for i in range(logScaled_reco_sortedByProv.shape[1] + 1)]
        hot_encoded = torch.tensor(hot_encoded, device=device, dtype=dtype).unsqueeze(dim=1)
        hot_encoded = hot_encoded.expand(output_decoder.shape[0], -1, -1)
        output_decoder = torch.cat((output_decoder, hot_encoded), dim=2)
    
    prob_each_jet = model.model(output_decoder).squeeze(dim=2)
    prob_each_jet = prob_each_jet[:,:model.no_max_objects]

    jetExist_sampled = torch.where(prob_each_jet < 0.5, 0, 1)

    return jetExist_sampled

def existQuality_print(experiment, sampledEvent, logScaled_reco_target, plotJets, epoch):
    # check exist flag
    target_exist = logScaled_reco_target[:,plotJets,0]
    sampled_exist = sampledEvent[:,plotJets]

    # check overlapping values
    mask_same_exist = target_exist == sampled_exist
    fraction_same_exist = (torch.count_nonzero(mask_same_exist)/torch.numel(mask_same_exist)).cpu().numpy()

    # keep only exist = 0
    mask_exist_0 = target_exist == 0
    mask_same_exist_0 = target_exist[mask_exist_0] == sampled_exist[mask_exist_0]
    fraction_same_exist_0 = (torch.count_nonzero(mask_same_exist_0)/torch.numel(mask_same_exist_0)).cpu().numpy()

    # keep only exist = 1
    mask_exist_1 = target_exist == 1
    mask_same_exist_1 = target_exist[mask_exist_1] == sampled_exist[mask_exist_1]
    fraction_same_exist_1 = (torch.count_nonzero(mask_same_exist_1)/torch.numel(mask_same_exist_1)).cpu().numpy()

    # plot quality of `exist` sampling
    fig, ax = plt.subplots(figsize=(7,6), dpi=100)
    ax.bar(["all Jets", "Jets With Exist=0", "Jets With Exist=1"], [fraction_same_exist, fraction_same_exist_0, fraction_same_exist_1], color ='maroon', width = 0.4)
    ax.set_ylabel(f'Fraction of correct assignments from total values')
    experiment.log_figure(f"Quality_flow_exist jets:{plotJets}", fig, step=epoch)

    experiment.log_metric("fraction_same_exist", fraction_same_exist, epoch=epoch)
    experiment.log_metric("fraction_same_exist_0", fraction_same_exist_0, epoch=epoch)
    experiment.log_metric("fraction_same_exist_1", fraction_same_exist_1, epoch=epoch)

def unscale_pt(logScaled_reco, mask_recoParticles, log_mean_reco, log_std_reco, no_max_objects):
    # pt is on the 2nd position, exist flag is on the first position (only for logScaled_reco)
    unscaled_pt = torch.exp(logScaled_reco[:,:no_max_objects,1]*log_std_reco[0] + log_mean_reco[0]) - 1
    unscaled_pt = unscaled_pt*mask_recoParticles[:,:no_max_objects] # set masked objects to 0
    return unscaled_pt
    
def compute_loss_per_pt(loss_per_pt, flow_pr, scaledLogReco, maskedReco, log_mean_reco, log_std_reco, no_max_objects,
                        pt_bins=[5, 50, 75, 100, 150, 200, 300, 1500, 3000]):
    unscaled_pt = unscale_pt(scaledLogReco, maskedReco, log_mean_reco, log_std_reco, no_max_objects)

    for i in range(len(pt_bins) - 1):
        mask_pt_greater = unscaled_pt > pt_bins[i]
        mask_pt_lower = unscaled_pt < pt_bins[i+1]
        mask_pt = torch.logical_and(mask_pt_greater, mask_pt_lower)
        #print(torch.count_nonzero(mask_pt))
        if torch.count_nonzero(mask_pt) == 0:
            loss_per_pt[i] =  0
        else:
            loss_per_pt[i] =  flow_pr[mask_pt].mean()

    return loss_per_pt

def attach_position(input_tensor, position):
    position = position.expand(input_tensor.shape[0], -1, -1)
    input_tensor = torch.cat((input_tensor, position), dim=2)
    
    return input_tensor

def check_calibration_batch(prediction, target_jets, calibration_bins):

    confidence = []
    accuracy = []
    noElems_bins = []
    
    for i in range(len(calibration_bins) - 1):
        mask_jets = (prediction > calibration_bins[i]) & (prediction <= calibration_bins[i+1])
        prediction_currentBin = prediction[mask_jets]

        jets_currentBin = target_jets[mask_jets]

        confidence_currentBin = torch.sum(prediction_currentBin.flatten())
        accuracy_currentBin = torch.sum(jets_currentBin.flatten())

        confidence.append(confidence_currentBin)
        accuracy.append(accuracy_currentBin)
        noElems_bins.append(torch.numel(prediction_currentBin.flatten()))
        

    return confidence, accuracy, noElems_bins

def compute_ECE(confidence, accuracy, noElems_bins, N_valid):

    ECE_perBin = [noElems_bins[i]*abs(accuracy[i] - confidence[i])/N_valid for i in range(len(confidence))]
    total_ECE = sum(ECE_perBin)
    
    return total_ECE

def reliability_diagram(experiment, accuracy, confidence, calibration_bins, ECE, epoch):
    width = calibration_bins[1]-calibration_bins[0]
    positions = torch.Tensor([width/2 + x for x in calibration_bins[:-1]])
    
    fig, ax = plt.subplots(figsize=(4,4), constrained_layout=True)    

    accuracy_plot = ax.bar(positions.cpu().numpy(), accuracy.cpu().numpy(), width=width, color='b', alpha=0.6,
                 linewidth=1, label="accuracy")

    conf_plot = ax.bar(positions.cpu().numpy(), confidence.cpu().numpy(), width=width, color='r', alpha=0.25,
                     linewidth=1, label="confidence avg")
    
    ax.plot([0,1], [0,1], linestyle = "--", color="gray")
    
    ax.legend(fontsize=8)
        
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    ax.set_xlabel('confidence', fontsize=12)
    ax.set_ylabel('accuracy', fontsize=12)
    
    ax.set_xticks(calibration_bins)
    ax.set_xticks([], minor=True)
    
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    
    ax.annotate(f'ECE = {ECE}', (0.6,0.8), xytext=(0.6, 0.15), fontsize=10)

    experiment.log_figure(f"reliability_diagram", fig, step=epoch)

    experiment.log_metric("ECE", ECE, epoch=epoch)

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
           world_size=None, device_ids=None, easy_version=False):
    # device is device when not distributed and rank when distributed
    print("START OF RANK:", device)
    if world_size is not None:
        ddp_setup(device, world_size, config.ddp_port)

    device_id = device_ids[device] if device_ids is not None else device
 
    modelName = f"{name_dir}/model_Classifier_ExistBinary_matched_{config.name}_{config.version}.pt"

    #print("Loading datasets")
    train_dataset = DatasetCombined(config.input_dataset_train, dev=device,
                                    dtype=dtype, datasets=['partons_lab', 'reco_lab'],
                           reco_list_lab=['scaledLogReco_sortedByProv',
                                          'mask_scaledLogReco_sortedByProv',
                                          'mask_boost', 'scaledLogBoost'],
                           parton_list_lab=['logScaled_data_higgs_t_tbar_ISR'])

    val_dataset = DatasetCombined(config.input_dataset_validation,dev=device,
                                  dtype=dtype, datasets=['partons_lab', 'reco_lab'],
                           reco_list_lab=['scaledLogReco_sortedByProv',
                                          'mask_scaledLogReco_sortedByProv',
                                          'mask_boost', 'scaledLogBoost'],
                           parton_list_lab=['logScaled_data_higgs_t_tbar_ISR'])

    log_mean_reco = train_dataset.reco_lab.meanRecoParticles
    log_std_reco = train_dataset.reco_lab.stdRecoParticles
    log_mean_parton = train_dataset.partons_lab.mean_log_data_higgs_t_tbar_ISR
    log_std_parton = train_dataset.partons_lab.std_log_data_higgs_t_tbar_ISR
    no_recoObjects = train_dataset.reco_lab.scaledLogReco_sortedByProv.shape[1]

    if easy_version:
        print('Load easy version: only events with 6/7/8 jets')
        scaledRecoList = [train_dataset.reco_lab.scaledLogReco_sortedByProv, val_dataset.reco_lab.scaledLogReco_sortedByProv]

        for i in range(2):
            scaledReco = scaledRecoList[i]
            no_objs = torch.sum(scaledReco[:,:,0], dim=1)
            mask_min6Jets_max8Jets = (no_objs >= 8) & (no_objs < 11)
            indices = mask_min6Jets_max8Jets.nonzero().squeeze(dim=1)
            if i == 0:
                train_dataset = torch.utils.data.Subset(train_dataset, indices)
            else:
                 val_dataset = torch.utils.data.Subset(val_dataset, indices)

    if device == torch.device('cuda'):
        log_mean_reco = log_mean_reco.cuda()
        log_std_reco = log_std_reco.cuda()
        log_mean_parton = log_mean_parton.cuda()
        log_std_parton = log_std_parton.cuda()


    pt_bins=[5, 50, 75, 100, 150, 200, 300, 1500]

    # Initialize model
    model = Classify_ExistJet(no_recoVars=5, # exist + 3-mom + position
                no_partonVars=4, # 3-mom + position
                no_recoObjects=no_recoObjects,

                no_transformers=config.transformerConditioning.no_transformers,
                transformer_input_features=config.transformerConditioning.input_features,
                transformer_nhead=config.transformerConditioning.nhead,
                transformer_num_encoder_layers=config.transformerConditioning.no_encoder_layers,
                transformer_num_decoder_layers=config.transformerConditioning.no_decoder_layers,
                transformer_dim_feedforward=config.transformerConditioning.dim_feedforward,
                transformer_activation=nn.GELU(),
                 
                DNN_nodes=config.DNN.nodes,
                DNN_layers=config.DNN.layers,
                no_max_objects=config.transferFlow.no_max_objects,
                 
                device=device,
                dtype=dtype,
                eps=1e-4)

    # Experiment logging
    if device == 0 or world_size is None:
        # Loading comet_ai logging
        exp = Experiment(
            api_key=config.comet_token,
            project_name="pretrain_flow_exist",
            workspace="antoniopetre",
            auto_output_logging = "simple",
            # disabled=True
        )
        exp.add_tags([config.name, config.version, 'sortProv', 'hot-encoded+inTransformer', 'Classifier_notFlow', f'min6Jets_max8={easy_version}', 'only_exist_pt_eta_phi', 'jetsSortedbyProv', 'HiggsAssignment', 'null_token_only_in_transformer', f'scheduler={config.training_params.scheduler}'])
        exp.log_parameters(config)
        exp.log_parameters({"model_param_tot":count_parameters(model)})
        exp.log_parameters({"model_param_transformer":count_parameters(model.transformer_model)})
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

    optimizer = optim.RAdam(list(model.parameters()) , lr=config.training_params.lr)
    # optimizer = optim.Rprop(list(model.parameters()) , lr=config.training_params.lr)
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

    loss = nn.BCELoss()
    loss_eachObj = nn.BCELoss(reduction='none')

    ii = 0
    ii_valid = 0

    # attach one-hot encoded position for jets
    pos_jets_lepton_MET = [pos for pos in range(8)] # 6 jets + lepton + MET
    pos_other_jets = [8 for pos in range(no_recoObjects - 8)]
    
    pos_jets_lepton_MET = torch.tensor(pos_jets_lepton_MET, device=device, dtype=dtype)
    pos_other_jets = torch.tensor(pos_other_jets, device=device, dtype=dtype)
    pos_logScaledReco = torch.cat((pos_jets_lepton_MET, pos_other_jets), dim=0).unsqueeze(dim=1)

    # attach one-hot encoded position for partons
    pos_partons = torch.tensor([0,1,2,3], device=device, dtype=dtype).unsqueeze(dim=1) # higgs, t1, t2, ISR

    # check calibration
    calibration_bins = [i/10 for i in range(10)]

    for e in range(config.training_params.nepochs):
        
        N_train = 0
        N_valid = 0
        if world_size is not None:
            print(
                f"[GPU{device_id}] | Rank {device} | Epoch {e} | Batchsize: {config.training_params.batch_size_training*len(device_ids)} | Steps: {len(trainingLoader)}"
            )
            trainingLoader.sampler.set_epoch(e)
            
        sum_loss = 0.
        loss_total_each_object = torch.zeros(config.transferFlow.no_max_objects, device=device)
        loss_per_pt = torch.zeros(len(pt_bins) - 1, device=device)
        total_loss_per_pt = torch.zeros(len(pt_bins) - 1, device=device)
    
        # training loop    
        print("Before training loop")
        ddp_model.train()

        for i, data_batch in enumerate(trainingLoader):
            N_train += 1
            ii+=1

            optimizer.zero_grad()
            
            (logScaled_partons,
             logScaled_reco_sortedByProv, mask_recoParticles,
             mask_boost, data_boost_reco) = data_batch
                            
            # exist + 3-mom
            logScaled_reco_sortedByProv = logScaled_reco_sortedByProv[:,:,:4]
            # The provenance is remove in the model

            # attach 1 hot-encoded position
            logScaled_reco_sortedByProv = attach_position(logScaled_reco_sortedByProv, pos_logScaledReco)
            logScaled_partons = attach_position(logScaled_partons, pos_partons)

            avg_flow_prob_exist, flow_prob_exist_batch, flow_prob_exist = ddp_model(logScaled_reco_sortedByProv,
                                                                                logScaled_partons,
                                                                                data_boost_reco,
                                                                                mask_recoParticles,
                                                                                mask_boost)


            loss_main = loss(flow_prob_exist, logScaled_reco_sortedByProv[:,:config.transferFlow.no_max_objects,0])

            mask_recoParticles = mask_recoParticles[:,:config.transferFlow.no_max_objects]
            logScaled_reco_sortedByProv = logScaled_reco_sortedByProv[:,:config.transferFlow.no_max_objects]

            loss_main_perObj = loss_eachObj(flow_prob_exist, logScaled_reco_sortedByProv[:,:config.transferFlow.no_max_objects, 0])
            loss_Sum_each_object = torch.sum(loss_main_perObj*mask_recoParticles, dim=0)
            number_MaskedObjects = torch.sum(mask_recoParticles, dim=0)
            loss_mean_each_object = torch.div(loss_Sum_each_object, number_MaskedObjects)

            # for cases when jets no. 12 doesn't exist in any of the events -> nan because 0/0 -> replace with 0
            loss_mean_each_object = torch.nan_to_num(loss_mean_each_object, nan=0.0)
            loss_total_each_object = torch.add(loss_total_each_object, loss_mean_each_object)

            loss_per_pt = compute_loss_per_pt(loss_per_pt, loss_main_perObj, logScaled_reco_sortedByProv, mask_recoParticles, log_mean_reco, log_std_reco, config.transferFlow.no_max_objects, pt_bins=pt_bins)
            total_loss_per_pt = torch.add(total_loss_per_pt, loss_per_pt)

            if torch.isnan(loss_total_each_object).any() or torch.isinf(loss_total_each_object).any():
                print(f'ii= {ii} Training_Total: nans = {torch.count_nonzero(torch.isnan(loss_total_each_object))}       infs = {torch.count_nonzero(torch.isinf(loss_total_each_object))}')

            if torch.isnan(total_loss_per_pt).any() or torch.isinf(total_loss_per_pt).any():
                print(f'ii= {ii} Training_pt: nans = {torch.count_nonzero(torch.isnan(total_loss_per_pt))}       infs = {torch.count_nonzero(torch.isinf(total_loss_per_pt))}')
                    
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
                        for j in range(config.transferFlow.no_max_objects):
                            exp.log_metric(f'loss_object_{j}', loss_mean_each_object[j], step=ii)
                        for j in range(len(pt_bins) - 1):
                            exp.log_metric(f'loss_per_pt_{pt_bins[j]}_{pt_bins[j+1]}', loss_per_pt[j], step=ii)

        ### END of training 
        if exp is not None and device==0 or world_size is None:
            exp.log_metric("loss_train_epoch", sum_loss/N_train, epoch=e)
            for j in range(config.transferFlow.no_max_objects):
                exp.log_metric(f'loss_epoch_total_object_{j}', loss_total_each_object[j]/N_train, epoch=e)
            for j in range(len(pt_bins) - 1):
                exp.log_metric(f'loss_epoch_total_per_pt_{pt_bins[j]}_{pt_bins[j+1]}', total_loss_per_pt[j]/N_train, epoch=e)
            # if scheduler_type == "cyclic_lr":
            #     exp.log_metric("learning_rate", scheduler.get_last_lr(), epoch=e, step=ii)

        total_valid_loss = 0.
        loss_Valid_total_each_object = torch.zeros(config.transferFlow.no_max_objects, device=device)
        valid_total_loss_per_pt = torch.zeros(len(pt_bins) - 1, device=device)
        f1_valid_total = 0.
        f1_valid_total_eachObject = torch.zeros(config.transferFlow.no_max_objects, device=device)

        total_confidence = [0 for i in range(len(calibration_bins) - 1)]
        total_accuracy = [0 for i in range(len(calibration_bins) - 1)]
        noElems_bins = [0 for i in range(len(calibration_bins) - 1)]
        
        # validation loop (don't update weights and gradients)
        print("Before validation loop")
        ddp_model.eval()
        
        for i, data_batch in enumerate(validLoader):
            N_valid += 1
            ii_valid += 1
            # Move data to device
            with torch.no_grad():

                (logScaled_partons,
                logScaled_reco_sortedByProv, mask_recoParticles,
                mask_boost, data_boost_reco) = data_batch
                
                # exist + 3-mom
                logScaled_reco_sortedByProv = logScaled_reco_sortedByProv[:,:,:4]

                # attach 1 hot-encoded position
                logScaled_reco_sortedByProv = attach_position(logScaled_reco_sortedByProv, pos_logScaledReco)
                logScaled_partons = attach_position(logScaled_partons, pos_partons)

                # The provenance is remove in the model
                avg_flow_prob_exist, flow_prob_exist_batch, flow_prob_exist = ddp_model(logScaled_reco_sortedByProv,
                                                                                    logScaled_partons,
                                                                                    data_boost_reco,
                                                                                    mask_recoParticles,
                                                                                    mask_boost)


                loss_main = loss(flow_prob_exist, logScaled_reco_sortedByProv[:,:config.transferFlow.no_max_objects,0])
                
                total_valid_loss += loss_main.item() # using only the main loss

                loss_main_perObj = loss_eachObj(flow_prob_exist, logScaled_reco_sortedByProv[:,:config.transferFlow.no_max_objects, 0])
                loss_Sum_each_object = torch.sum(loss_main_perObj*mask_recoParticles[:,:config.transferFlow.no_max_objects], dim=0)
                number_MaskedObjects = torch.sum(mask_recoParticles[:,:config.transferFlow.no_max_objects], dim=0)
                loss_mean_each_object = torch.div(loss_Sum_each_object, number_MaskedObjects)

                loss_mean_each_object = torch.nan_to_num(loss_mean_each_object, nan=0.0)
                loss_Valid_total_each_object = torch.add(loss_Valid_total_each_object, loss_mean_each_object)

                loss_per_pt = compute_loss_per_pt(loss_per_pt, loss_main_perObj, logScaled_reco_sortedByProv, mask_recoParticles, log_mean_reco, log_std_reco, config.transferFlow.no_max_objects,
                        pt_bins=pt_bins)

                valid_total_loss_per_pt = torch.add(valid_total_loss_per_pt, loss_per_pt)

                if torch.isnan(loss_Valid_total_each_object).any() or torch.isinf(loss_Valid_total_each_object).any():
                    print(f'ii= {ii} VALID_total: nans = {torch.count_nonzero(torch.isnan(loss_Valid_total_each_object))}       infs = {torch.count_nonzero(torch.isinf(loss_Valid_total_each_object))}')

                if torch.isnan(valid_total_loss_per_pt).any() or torch.isinf(valid_total_loss_per_pt).any():
                    print(f'ii= {ii} VALID_pt: nans = {torch.count_nonzero(torch.isnan(valid_total_loss_per_pt))}       infs = {torch.count_nonzero(torch.isinf(valid_total_loss_per_pt))}')

                prediction = torch.where(flow_prob_exist < 0.5, 0, 1)
                f1_score = multiclass_f1_score(prediction.flatten(), logScaled_reco_sortedByProv[:,:config.transferFlow.no_max_objects,0].flatten(), num_classes=2)

                f1_valid_total += f1_score
                for jet in range(config.transferFlow.no_max_objects):
                    f1_valid_total_eachObject[jet] += multiclass_f1_score(prediction[:,jet], logScaled_reco_sortedByProv[:,jet,0], num_classes=2)

                if i == 0:
                    # print sampled partons
                    fullGeneratedEvent = sample_existToken(model, logScaled_reco_sortedByProv, logScaled_partons, mask_recoParticles, device, dtype)

                    allJets = [i for i in range(config.transferFlow.no_max_objects)]
                    existQuality_print(exp, fullGeneratedEvent, logScaled_reco_sortedByProv, allJets, e)

                    for jet in range(config.transferFlow.no_max_objects):
                        existQuality_print(exp, fullGeneratedEvent, logScaled_reco_sortedByProv, jet, e)

                confidence_batch, accuracy_batch, noElems_bins_batch = check_calibration_batch(flow_prob_exist, logScaled_reco_sortedByProv[:,:config.transferFlow.no_max_objects,0], calibration_bins)

                total_confidence = [x + y for x, y in zip(total_confidence, confidence_batch)]
                total_accuracy = [x + y for x, y in zip(total_accuracy, accuracy_batch)]
                noElems_bins = [x + y for x, y in zip(noElems_bins, noElems_bins_batch)]

                # check calibration
                if i == len(validLoader) - 1:
                    total_confidence = [x/noElems_bins[i] for i, x in enumerate(total_confidence)]
                    total_accuracy = [x/noElems_bins[i] for i, x in enumerate(total_accuracy)]

                    ECE = compute_ECE(total_confidence, total_accuracy, noElems_bins, N_valid)

                    reliability_diagram(exp, torch.tensor(total_accuracy), torch.tensor(total_confidence),
                                        torch.tensor(calibration_bins), ECE, e)

                    
                                                    

        if exp is not None and device==0 or world_size is None:
            exp.log_metric("total_valid_loss", total_valid_loss/N_valid, epoch=e)
            exp.log_metric("f1_valid_total", f1_valid_total/N_valid, epoch=e)
            for j in range(config.transferFlow.no_max_objects):
                exp.log_metric(f'loss_Valid_epoch_total_object_{j}', loss_Valid_total_each_object[j]/N_valid, epoch=e)
                exp.log_metric(f'f1_valid_total_eachObject_{j}', f1_valid_total_eachObject[j]/N_valid, epoch=e)
            for j in range(len(pt_bins) - 1):
                exp.log_metric(f'loss_Valid_epoch_total_per_pt_{pt_bins[j]}_{pt_bins[j+1]}', valid_total_loss_per_pt[j]/N_valid, epoch=e)
            
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
    print('Flow training finished!!')
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-config', type=str, required=True, help='path to config.yaml File')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--on-GPU', action="store_true",  help='run on GPU boolean')
    parser.add_argument('--easy-version', action="store_true",  help='6 Jets')
    parser.add_argument('--distributed', action="store_true")
    args = parser.parse_args()
    
    path_to_conf = args.path_config
    on_GPU = args.on_GPU # by default run on CPU
    outputDir = args.output_dir
    easy_version = args.easy_version

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
    name_dir = f'{outputDir}/Classifier_ExistBinary_matched_{conf.name}_{conf.version}_{conf.transferFlow.base}_NoTransf{conf.transferFlow.ntransforms}_NoBins{conf.transferFlow.bins}_DNN:{conf.transferFlow.hiddenMLP_NoLayers}_{conf.transferFlow.hiddenMLP_LayerDim}'
    

    os.makedirs(name_dir, exist_ok=True)
    
    with open(f"{name_dir}/config_{conf.name}_{conf.version}.yaml", "w") as fo:
        fo.write(OmegaConf.to_yaml(conf)) 

    if conf.training_params.dtype == "float32":
        dtype = torch.float32
    elif conf.training_params.dtype == "float64":
        dtype = torch.float64
    else:
        dtype = None
        
    
    if len(actual_devices) > 1 and args.distributed:
        # make a dictionary with k: rank, v: actual device
        dev_dct = {i: actual_devices[i] for i in range(world_size)}
        print(f"Devices dict: {dev_dct}")
        mp.spawn(
            train,
            args=(name_dir, conf,  outputDir, dtype,
                    world_size, dev_dct, easy_version),
            nprocs=world_size,
            # join=True
        )
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        train(device, name_dir, conf,  outputDir, dtype, None, None, easy_version)
    
    print(f"Flow training finished succesfully! Version: {conf.version}")
    
    
    
    
    
    
    
    
    
