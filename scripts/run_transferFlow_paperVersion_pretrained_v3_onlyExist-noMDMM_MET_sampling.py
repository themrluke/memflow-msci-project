from comet_ml import Experiment
#from comet_ml.integration.pytorch import log_model

import torch
from memflow.read_data.dataset_all import DatasetCombined
from memflow.transfer_flow.transfer_flow_paper_pretrained_v3_onlyExist_met import TransferFlow_Paper_pretrained_v3_onlyExist_MET
from memflow.unfolding_flow.utils import *
from utils_transferFlow_paper import sample_fullRecoEvent_classifier_v3_leptonMET
from utils_transferFlow_paper import existQuality_print
from utils_transferFlow_paper import sampling_print
from utils_transferFlow_paper import validation_print
from utils_transferFlow_paper import unscale_pt
from utils_transferFlow_paper import compute_loss_per_pt
import mdmm

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

def attach_position(input_tensor, position):
    position = position.expand(input_tensor.shape[0], -1, -1)
    input_tensor = torch.cat((input_tensor, position), dim=2)
    
    return input_tensor

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

def train(device, path_to_dir, path_to_model, config, dtype, old_order, validation,
           world_size=None, device_ids=None):
    # device is device when not distributed and rank when distributed
    print("START OF RANK:", device)
    if world_size is not None:
        ddp_setup(device, world_size, config.ddp_port)

    device_id = device_ids[device] if device_ids is not None else device

    dataset = config.input_dataset_test
    if validation:
        print('use validation dataset')
        dataset = config.input_dataset_validation
 
    test_dataset = DatasetCombined(dataset, dev=device,
                                    dtype=dtype, datasets=['partons_lab', 'reco_lab'],
                           reco_list_lab=['scaledLogReco_sortedBySpanet',
                                          'mask_scaledLogReco_sortedBySpanet',
                                          'mask_boost', 'scaledLogBoost'],
                           parton_list_lab=['logScaled_data_higgs_t_tbar_ISR'])

    no_recoObjs = test_dataset.reco_lab.scaledLogReco_sortedBySpanet.shape[1]

    log_mean_reco = test_dataset.reco_lab.meanRecoParticles
    log_std_reco = test_dataset.reco_lab.stdRecoParticles
    log_mean_parton = test_dataset.partons_lab.mean_log_data_higgs_t_tbar_ISR
    log_std_parton = test_dataset.partons_lab.std_log_data_higgs_t_tbar_ISR

    if device == torch.device('cuda'):
        log_mean_reco = log_mean_reco.cuda()
        log_std_reco = log_std_reco.cuda()
        log_mean_parton = log_mean_parton.cuda()
        log_std_parton = log_std_parton.cuda()

    # Initialize model
    model = TransferFlow_Paper_pretrained_v3_onlyExist_MET(no_recoVars=5, # exist + 3-mom + encoded_position
                no_partonVars=4,
                no_recoObjects=no_recoObjs,

                no_transformers=config.transformerConditioning.no_transformers,
                transformer_input_features=config.transformerConditioning.input_features,
                transformer_nhead=config.transformerConditioning.nhead,
                transformer_num_encoder_layers=config.transformerConditioning.no_encoder_layers,
                transformer_num_decoder_layers=config.transformerConditioning.no_decoder_layers,
                transformer_dim_feedforward=config.transformerConditioning.dim_feedforward,
                transformer_activation=nn.GELU(),
                 
                flow_nfeatures=3, #pt/eta/phi
                flow_ntransforms=config.transferFlow.ntransforms,
                flow_hiddenMLP_NoLayers=config.transferFlow.hiddenMLP_NoLayers,
                flow_hiddenMLP_LayerDim=config.transferFlow.hiddenMLP_LayerDim,
                flow_bins=config.transferFlow.bins,
                flow_autoregressive=config.transferFlow.autoregressive,
                flow_base=config.transferFlow.base,
                flow_base_first_arg=config.transferFlow.base_first_arg,
                flow_base_second_arg=config.transferFlow.base_second_arg,
                flow_bound=config.transferFlow.bound,
                randPerm=config.transferFlow.randPerm,
                no_max_objects=config.transferFlow.no_max_objects,

                flow_lepton_ntransforms=config.transferFlow_lepton.ntransforms,
                flow_lepton_hiddenMLP_NoLayers=config.transferFlow_lepton.hiddenMLP_NoLayers,
                flow_lepton_hiddenMLP_LayerDim=config.transferFlow_lepton.hiddenMLP_LayerDim,
                flow_lepton_bins=config.transferFlow_lepton.bins,

                DNN_nodes=config.DNN.nodes, DNN_layers=config.DNN.layers,
                pretrained_classifier=config.DNN.path_pretraining,
                load_classifier=False,
                encode_position=True,
                 
                device=device,
                dtype=dtype,
                eps=1e-4)

    # Setting up DDP
    model = model.to(device)

    state_dict = torch.load(path_to_model, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])

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

    batch_size = config.training_params.batch_size_training
    No_samples = 30
    fileName = 'sample_many_events.pt'
    No_eventsSampled = test_dataset.reco_lab.scaledLogReco_sortedBySpanet.shape[0] / batch_size
    
    if few_events:
        batch_size = 2
        No_samples = 20000
        fileName = 'sample_few_events.pt'
        No_eventsSampled = 50

    if validation:
        fileName = 'valid.pt'
        
    # Datasets
    testLoader = DataLoader(
        test_dataset,
        batch_size= batch_size,
        shuffle=False if world_size is not None else True,
        sampler=DistributedSampler(test_dataset) if world_size is not None else None,
        drop_last=True
    )
    
    # attach one-hot encoded position for jets
    pos_jets_lepton_MET = [pos for pos in range(8)] # 6 jets + lepton + MET
    pos_other_jets = [8 for pos in range(no_recoObjs - 8)]
    
    pos_jets_lepton_MET = torch.tensor(pos_jets_lepton_MET, device=device, dtype=dtype)
    pos_other_jets = torch.tensor(pos_other_jets, device=device, dtype=dtype)
    pos_logScaledReco = torch.cat((pos_jets_lepton_MET, pos_other_jets), dim=0).unsqueeze(dim=1)

    # attach one-hot encoded position for partons
    pos_partons = torch.tensor([0,1,2,3], device=device, dtype=dtype).unsqueeze(dim=1) # higgs, t1, t2, ISR

    # new order: lepton MET tlep thad Higgs etc
    new_order_list = [6,7,5,2,3,4,0,1]
    if old_order:
        new_order_list = [6,7,0,1,2,3,4,5] # first order used in this model
        
    lastElems = [i+8 for i in range(no_recoObjs - 8)]
    new_order_list = new_order_list + lastElems
    new_order = torch.LongTensor(new_order_list)

    # No_samples + 1 because the first elem is the target
    unscaled_sampledTensor = torch.empty((0, No_samples + 1, config.transferFlow.no_max_objects, 3), device=device, dtype=dtype)
               
    print("Before sampling loop")
    ddp_model.eval()
    
    for i, data_batch in enumerate(testLoader):
        # Move data to device
        with torch.no_grad():

            if (few_events and batch_size*i > 50):
                break

            if (i % 50 == 0):
                print(f'batch {i} from {len(testLoader)}')

            (logScaled_partons,
            logScaled_reco_sortedBySpanet, mask_recoParticles,
            mask_boost, data_boost_reco) = data_batch
            
            # exist + 3-mom
            logScaled_reco_sortedBySpanet = logScaled_reco_sortedBySpanet[:,:,:4] 

            # put the lepton first:
            logScaled_reco_sortedBySpanet = logScaled_reco_sortedBySpanet[:,new_order,:]
            mask_recoParticles = mask_recoParticles[:,new_order]

            # attach 1 hot-encoded position
            logScaled_reco_sortedBySpanet = attach_position(logScaled_reco_sortedBySpanet, pos_logScaledReco)
            logScaled_partons = attach_position(logScaled_partons, pos_partons)
                    
            # print sampled partons
            fullGeneratedEvent, mask_reco = sample_fullRecoEvent_classifier_v3_leptonMET(ddp_model, logScaled_partons, logScaled_reco_sortedBySpanet.shape[0], device, dtype, No_samples=No_samples)

            fullGeneratedEvent = torch.reshape(fullGeneratedEvent, (No_samples, batch_size, config.transferFlow.no_max_objects, 5))
            fullGeneratedEvent = torch.transpose(fullGeneratedEvent, 0, 1)

            mask_nonExist = (fullGeneratedEvent[...,0] == 0).unsqueeze(dim=3)

            fullGeneratedEvent[...,1:4] = (fullGeneratedEvent[...,1:4]*log_std_reco + log_mean_reco)
            fullGeneratedEvent[...,1] = (torch.exp(fullGeneratedEvent[...,1]) - 1) # unscale pt
            fullGeneratedEvent[...,1:4] = torch.where(mask_nonExist, -5, fullGeneratedEvent[...,1:4])

            maskTarget_nonExist = (logScaled_reco_sortedBySpanet[...,0] == 0).unsqueeze(dim=2)
            
            unscaledTarget = (logScaled_reco_sortedBySpanet[...,1:4]*log_std_reco + log_mean_reco)
            unscaledTarget[...,0] = (torch.exp(unscaledTarget[...,0]) - 1) # unscale pt
            unscaledTarget = torch.where(maskTarget_nonExist, -5, unscaledTarget).unsqueeze(dim=1)

            fullGeneratedEvent_withFirstTarget = torch.cat((unscaledTarget[:,:,:config.transferFlow.no_max_objects],
                                                           fullGeneratedEvent[...,1:4]),
                                                           dim=1)
            
            unscaled_sampledTensor = torch.cat((unscaled_sampledTensor, fullGeneratedEvent_withFirstTarget), dim=0)
            
            
            
    torch.save(unscaled_sampledTensor, f'{path_to_dir}/{fileName}')
    # exp_log.end()
    destroy_process_group()
    print('Sampling finished!!')
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-model', type=str, required=True, help='path to model directory')
    parser.add_argument('--few-events',action="store_true",  help='sample 20k reco events for only 50 gen-level events')
    parser.add_argument('--old-order',action="store_true",  help='use first ordering (lep/MET/Higgs/thad/tlep)')
    parser.add_argument('--on-GPU', action="store_true",  help='run on GPU boolean')
    parser.add_argument('--validation', action="store_true",  help='use validation dataset')
    parser.add_argument('--distributed', action="store_true")
    args = parser.parse_args()
    
    path_to_dir = args.path_model
    few_events = args.few_events
    old_order = args.old_order
    on_GPU = args.on_GPU # by default run on CPU
    validation = args.validation

    path_to_conf = path_to_dir + '/config_transfer_flow_2nd_2nd.yaml'
    path_to_model = path_to_dir + '/model_transfer_flow_2nd_2nd.pt'

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
            args=(device, path_to_dir, path_to_model, conf, dtype, old_order, validation,
                    world_size, dev_dct),
            nprocs=world_size,
            # join=True
        )
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        train(device, path_to_dir, path_to_model, conf, dtype, old_order, validation, None, None)
    
    print(f"Sampled succesfully! Version: {conf.version}")
    
    
    
    
    
    
    
    
    
