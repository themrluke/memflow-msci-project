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
import vector

from utils import constrain_energy
from utils import total_mom

from earlystop import EarlyStopper

PI = torch.pi

def loss_fn_periodic(input, target, loss_fn, device):

    deltaPhi = target - input
    deltaPhi = torch.where(deltaPhi > PI, deltaPhi - 2*PI, deltaPhi)
    deltaPhi = torch.where(deltaPhi <= -PI, deltaPhi + 2*PI, deltaPhi)
    
    return loss_fn(deltaPhi, torch.zeros(deltaPhi.shape, device=device))

def compute_losses(logScaled_partons, higgs, thad, tlep, cartesian, loss_fn, device):
    lossH = 0
    lossThad = 0
    lossTlep = 0

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
            # case when feature is equal to phi (phi is periodic variable)
            else:
                lossH += loss_fn_periodic(higgs[:,feature], logScaled_partons[:,0,feature], loss_fn, device)
                lossThad +=  loss_fn_periodic(thad[:,feature], logScaled_partons[:,1,feature], loss_fn, device)
                lossTlep +=  loss_fn_periodic(tlep[:,feature], logScaled_partons[:,2,feature], loss_fn, device)

        lossH = lossH/higgs.shape[1]
        lossThad = lossThad/thad.shape[1]
        lossTlep = lossTlep/tlep.shape[1]
    
    return lossH, lossThad, lossTlep

def TrainingAndValidLoop(config, device, model, trainingLoader, validLoader, outputDir, HuberLoss):
    
    if HuberLoss:
        loss_fn = torch.nn.HuberLoss(delta=1.0)
    else:
        loss_fn = torch.nn.MSELoss() 
    
    optimizer = optim.Adam(list(model.parameters()) , lr=config.training_params.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    
    outputDir = os.path.abspath(outputDir)
    latentSpace = config.conditioning_transformer.use_latent
    name_dir = f'{outputDir}/preTraining_noProv:{config.noProv}_cartesian:{config.cartesian}_HuberLoss:{HuberLoss}_latent_space:{latentSpace}_hiddenFeatures:{config.conditioning_transformer.hidden_features}_dimFeedForward:{config.conditioning_transformer.dim_feedforward_transformer}_nheadEnc:{config.conditioning_transformer.nhead_encoder}_LayersEnc:{config.conditioning_transformer.no_layers_encoder}_nheadDec:{config.conditioning_transformer.nhead_decoder}_LayersDec:{config.conditioning_transformer.no_layers_decoder}'
    
    writer = SummaryWriter(name_dir)

    with open(f"{name_dir}/config_{config.name}_{config.version}.yaml", "w") as fo:
        fo.write(OmegaConf.to_yaml(config)) 
    
    N_train = len(trainingLoader)
    N_valid = len(validLoader)
    
    ii = 0
    early_stopper = EarlyStopper(patience=config.training_params.nEpochsPatience, min_delta=0.001)

    modelName = f"{name_dir}/model_{config.name}_{config.version}.pt"

    log_mean = torch.tensor(config.scaling_params.log_mean, device=device)
    log_std = torch.tensor(config.scaling_params.log_std, device=device)

    for e in range(config.training_params.nepochs):
        
        sum_loss = 0.
    
        # training loop    
        print("Before training loop")
        model.train()
        for i, data in enumerate(trainingLoader):
            
            ii += 1

            if (i % 100 == 0):
                print(i)

            optimizer.zero_grad()
            
            (logScaled_partons, 
            logScaled_reco, mask_lepton_reco, 
            mask_jets, mask_met, 
            mask_boost_reco, data_boost_reco) = data
            
            mask_recoParticles = torch.cat((mask_jets, mask_lepton_reco, mask_met), dim=1)

            # remove prov
            if (config.noProv):
                logScaled_reco = logScaled_reco[:,:,:-1]

            out = model(logScaled_reco, data_boost_reco, mask_recoParticles, mask_boost_reco)
            
            higgs = out[0]
            thad = out[1]
            tlep = out[2]

            # check mass for debubgging
            # check_mass(ISR, log_mean, log_std)
        
            lossH, lossThad, lossTlep = compute_losses(logScaled_partons, higgs, thad, tlep,
                                                        config.cartesian, loss_fn, device)

            writer.add_scalar('loss_H', lossH.item(), ii)
            writer.add_scalar('loss_Thad', lossThad.item(), ii)
            writer.add_scalar('loss_Tlep', lossTlep.item(), ii)

            loss = lossH + lossThad + lossTlep
            
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()

            writer.add_scalar("Loss_step_train", loss.item(), ii)
        
        writer.add_scalar('Loss_epoch_train', sum_loss/N_train, e)
        valid_loss = 0
        valid_lossH = 0
        valid_lossTlep = 0
        valid_lossThad = 0
 
        # validation loop (don't update weights and gradients)
        print("Before validation loop")
        model.eval()
        for i, data in enumerate(validLoader):
            
            with torch.no_grad():

                (logScaled_partons, 
                logScaled_reco, mask_lepton_reco, 
                mask_jets, mask_met, 
                mask_boost_reco, data_boost_reco) = data
                
                mask_recoParticles = torch.cat((mask_jets, mask_lepton_reco, mask_met), dim=1)

                # remove prov
                if (config.noProv):
                    logScaled_reco = logScaled_reco[:,:,:-1]

                out = model(logScaled_reco, data_boost_reco, mask_recoParticles, mask_boost_reco)
            
                higgs = out[0]
                thad = out[1]
                tlep = out[2]

                lossH, lossThad, lossTlep = compute_losses(logScaled_partons, higgs, thad, tlep,
                                                        config.cartesian, loss_fn, device)

                loss = lossH + lossThad + lossTlep

                valid_loss += loss.item()
                valid_lossH += lossH.item()
                valid_lossTlep += lossTlep.item()
                valid_lossThad += lossThad.item()

                particle_list = [higgs, thad, tlep]
                
                if i == 0:
                    for particle in range(len(particle_list)): # 4 or 3 particles: higgs/thad/tlep/gluonISR
                        
                        # 4 or 3 features
                        for feature in range(config.conditioning_transformer.out_features):  

                            fig, ax = plt.subplots()
                            h = ax.hist2d(particle_list[particle][:,feature].cpu().detach().numpy().flatten(),
                                          logScaled_partons[:,particle,feature].detach().cpu().numpy(),
                                          bins=100, range=((-3, 3),(-3, 3)))
                            fig.colorbar(h[3], ax=ax)
                            writer.add_figure(f"Validation_particleNo_{particle}_Feature_{feature}", fig, e)

                            fig, ax = plt.subplots()
                            h = ax.hist((particle_list[particle][:,feature].cpu().detach().numpy().flatten() - \
                                         logScaled_partons[:,particle,feature].detach().cpu().numpy()), bins=100)
                            writer.add_figure(f"Validation_particleNo_{particle}__Feature_{feature}_Diff", fig, e)
         

        writer.add_scalar('Loss_epoch_val', valid_loss/N_valid, e)
        writer.add_scalar('Loss_epoch_val_H', valid_lossH/N_valid, e)
        writer.add_scalar('Loss_epoch_val_Tlep', valid_lossTlep/N_valid, e)
        writer.add_scalar('Loss_epoch_val_Thad', valid_lossThad/N_valid, e)

        if early_stopper.early_stop(valid_loss, model.state_dict(), optimizer.state_dict(), modelName):
            print(f"Model converges at epoch {e} !!!")         
            break

        scheduler.step() # reduce lr if the model is not improving anymore

    writer.close()

    print('preTraining finished!!')
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-config', type=str, required=True, help='path to config.yaml File')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--on-GPU', action="store_true",  help='run on GPU boolean')
    parser.add_argument('--huberLoss', action="store_true",  help='use Huber loss')
    args = parser.parse_args()
    
    path_to_conf = args.path_config
    on_GPU = args.on_GPU # by default run on CPU
    outputDir = args.output_dir
    use_huberLoss = args.huberLoss

    # Read config file in 'conf'
    with open(path_to_conf) as f:
        conf = OmegaConf.load(path_to_conf)
    
    print("Training with cfg: \n", OmegaConf.to_yaml(conf))
    
    if on_GPU:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
    
    # READ data
    if (conf.cartesian):
        data = DatasetCombined(conf.input_dataset, dev=device, dtype=torch.float64,
                                reco_list=['scaledLogRecoParticlesCartesian', 'mask_lepton', 
                                            'mask_jets','mask_met',
                                            'mask_boost', 'data_boost'],
                                parton_list=['logScaled_data_higgs_t_tbar_ISR_cartesian'])
    else:
        data = DatasetCombined(conf.input_dataset, dev=device, dtype=torch.float64,
                                reco_list=['scaledLogRecoParticles', 'mask_lepton', 
                                            'mask_jets','mask_met',
                                            'mask_boost', 'data_boost'],
                                parton_list=['logScaled_data_higgs_t_tbar_ISR'])

    # split data for training sample and validation sample
    train_subset, val_subset = torch.utils.data.random_split(
            data, [conf.training_params.training_sample, conf.training_params.validation_sample],
            generator=torch.Generator().manual_seed(1))
    
    train_loader = DataLoader(dataset=train_subset, shuffle=True, batch_size=conf.training_params.batch_size_training)
    val_loader = DataLoader(dataset=val_subset, shuffle=True, batch_size=conf.training_params.batch_size_validation)

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

    # Copy model on GPU memory
    if (device == torch.device('cuda')):
        model = model.cuda()

    print(f"parameters total:{count_parameters(model)}\n")

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
        TrainingAndValidLoop(conf, device, model, train_loader, val_loader, outputDir, use_huberLoss)
        #else:
        #    TrainingAndValidLoop(conf, device, model, train_loader, val_loader, outputDir, use_huberLoss)
    else:
        TrainingAndValidLoop(conf, device, model, train_loader, val_loader, outputDir, use_huberLoss)
        
    
    print(f"Normal version: preTraining finished succesfully! Version: {conf.version}")
    
    
    
    
    
    
    
    
    
    
    
    

