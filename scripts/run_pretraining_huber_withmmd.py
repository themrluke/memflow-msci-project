from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
from memflow.read_data.dataset_all import DatasetCombined

from memflow.unfolding_network.conditional_transformer import ConditioningTransformerLayer
from memflow.unfolding_flow.utils import *
from memflow.unfolding_flow.mmd_loss import MMD
from memflow.unfolding_flow.utils import Compute_ParticlesTensor

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.functional import normalize
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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

PI = torch.pi




def loss_fn_periodic(inp, target, loss_fn, device):

    # rescale to pi
    inp[inp>PI] = inp[inp>PI]-2*PI
    inp[inp<-PI] = inp[inp<-PI] + 2*PI
    
    deltaPhi = target - inp
    deltaPhi = torch.where(deltaPhi > PI, deltaPhi - 2*PI, deltaPhi)
    deltaPhi = torch.where(deltaPhi <= -PI, deltaPhi + 2*PI, deltaPhi)
    
    return loss_fn(deltaPhi, torch.zeros(deltaPhi.shape, device=device))

def compute_regr_losses(logScaled_partons, higgs, thad, tlep, cartesian, loss_fn,
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
                # lossGluon +=  loss_fn(logScaled_partons[:,3,feature], gluon[:,feature])
            # case when feature is equal to phi (phi is periodic variable)
            else:
                lossH += loss_fn_periodic(higgs[:,feature]*scaling_phi[1] + scaling_phi[0],
                                          logScaled_partons[:,0,feature]*scaling_phi[1] + scaling_phi[0], loss_fn, device)
                lossThad +=  loss_fn_periodic(thad[:,feature]*scaling_phi[1] + scaling_phi[0],
                                              logScaled_partons[:,1,feature]*scaling_phi[1] + scaling_phi[0], loss_fn, device)
                lossTlep +=  loss_fn_periodic(tlep[:,feature]*scaling_phi[1] + scaling_phi[0],
                                              logScaled_partons[:,2,feature]*scaling_phi[1] + scaling_phi[0], loss_fn, device)
                # lossGluon +=  loss_fn_periodic(gluon[:,feature]*scaling_phi[1] + scaling_phi[0],
                                              # logScaled_partons[:,3,feature]*scaling_phi[1] + scaling_phi[0], loss_fn, device)

    if split:
        return lossH, lossThad, lossTlep#, lossGluon
    else:
        return  (lossH + lossThad + lossTlep)/9


def compute_mmd_loss(mmd_input, mmd_target, kernel, device, split=False):
    mmds = []
    for particle in range(mmd_input.shape[1]):
        mmds.append(MMD(mmd_input[:,particle], mmd_target[:,particle], kernel, device ))
    if split:
        return mmds
    else:
        return sum(mmds)/3

def TrainingAndValidLoop(config, device, model, trainingLoader, validLoader, outputDir, HuberLoss):


    constraint = mdmm.MaxConstraint(
                    compute_mmd_loss,
                    config.MDMM.max_mmd, # to be modified based on the regression
                    scale=config.MDMM.scale_mmd,
                    damping=config.MDMM.dumping_mmd,
                    )

    # Create the optimizer
    MDMM_module = mdmm.MDMM([constraint]) # support many constraints TODO: use a constraint for every particle
    
    if HuberLoss:
        loss_fn = torch.nn.HuberLoss(delta=1.0)
    else:
        loss_fn = torch.nn.MSELoss() 
    
    # optimizer = optim.Adam(list(model.parameters()) , lr=config.training_params.lr)
    optimizer = MDMM_module.make_optimizer(model.parameters(), lr=config.training_params.lr)
    # optimizer = optim.Rprop(list(model.parameters()) , lr=config.training_params.lr)
    # scheduler = CosineAnnealingLR(optimizer, T_max=10)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           factor=config.training_params.reduce_on_plateau.factor,
                                                           patience=config.training_params.reduce_on_plateau.patience,
                                                           threshold=config.training_params.reduce_on_plateau.threshold)
    
    
    outputDir = os.path.abspath(outputDir)
    latentSpace = config.conditioning_transformer.use_latent
    name_dir = f'{outputDir}/preTraining_{config.name}_{config.version}_hiddenFeatures:{config.conditioning_transformer.hidden_features}_dimFeedForward:{config.conditioning_transformer.dim_feedforward_transformer}_nheadEnc:{config.conditioning_transformer.nhead_encoder}_LayersEnc:{config.conditioning_transformer.no_layers_encoder}_nheadDec:{config.conditioning_transformer.nhead_decoder}_LayersDec:{config.conditioning_transformer.no_layers_decoder}'

    os.makedirs(name_dir, exist_ok=True)
    # writer = SummaryWriter(name_dir)
    # Loading comet_ai logging
    exp = Experiment(
        api_key=config.comet_token,
        project_name="memflow",
        workspace="valsdav"
    )

    exp.add_tags([config.name,config.version])
    exp.log_parameters(config.training_params)
    exp.log_parameters(config.conditioning_transformer)
    exp.log_parameters(config.MDMM)

    with open(f"{name_dir}/config_{config.name}_{config.version}.yaml", "w") as fo:
        fo.write(OmegaConf.to_yaml(config)) 
    
    N_train = len(trainingLoader)
    N_valid = len(validLoader)
    
    early_stopper = EarlyStopper(patience=config.training_params.nEpochsPatience, min_delta=0.0001)

    modelName = f"{name_dir}/model_{config.name}_{config.version}.pt"

    log_mean = torch.tensor(config.scaling_params.log_mean, device=device)
    log_std = torch.tensor(config.scaling_params.log_std, device=device)

    ii = 0
    for e in range(config.training_params.nepochs):
        
        sum_loss = 0.
    
        # training loop    
        print("Before training loop")
        model.train()
        for i, data in enumerate(trainingLoader):
            ii+=1
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
            MMD_input = torch.cat((higgs.unsqueeze(1),
                                   thad.unsqueeze(1),
                                   tlep.unsqueeze(1)), dim=1)
            # compute gluon
            #HttISR_regressed, boost_regressed = Compute_ParticlesTensor.get_HttISR_numpy(out, log_mean,
                                                                                         # log_std, device,
                                                                                         # cartesian=False,
                                                                                         # eps=1e-5)
            #gluon = HttISR_regressed[:,-3:]
            MMD_target = logScaled_partons[:,0:3]

            lossH, lossThad, lossTlep = compute_regr_losses(logScaled_partons, higgs, thad, tlep,
                                                config.cartesian, loss_fn,
                                            scaling_phi=[log_mean[2], log_std[2]], # Added scaling for phi
                                                split=True,
                                                device=device)
            regr_loss =  (lossH + lossThad + lossTlep)/9
            
                    
            mdmm_return = MDMM_module(regr_loss, [(MMD_input, MMD_target, config.training_params.mmd_kernel, device, False)])

            loss_final = mdmm_return.value
            #print(f"MMD loss: {mdmm_return.fn_values}, huber loss {regr_loss.item()}, loss tot{loss_final.item()}")

            loss_final.backward()
            optimizer.step()

            with torch.no_grad():

                mmd_loss_H, mmd_loss_thad, mmd_loss_tlep = compute_mmd_loss(MMD_input, MMD_target, kernel=config.training_params.mmd_kernel, device=device, split=True)
                # lossH, lossThad, lossTlep = compute_regr_losses(logScaled_partons, higgs, thad, tlep,
                #                                 config.cartesian, loss_fn,
                #                             scaling_phi=[log_mean[2], log_std[2]], # Added scaling for phi
                #                                 split=True,
                #                                 device=device)

                exp.log_metric("loss_mmd_H", mmd_loss_H.item(),step=ii)
                exp.log_metric('loss_mmd_thad', mmd_loss_thad.item(),step=ii)
                exp.log_metric('loss_mmd_tlep', mmd_loss_tlep.item(),step=ii)
                exp.log_metric('loss_mmd_tot', (mmd_loss_H+mmd_loss_thad + mmd_loss_tlep).item()/3,step=ii)
                exp.log_metric('loss_huber_H', lossH.item()/3,step=ii)
                exp.log_metric('loss_huber_Thad', lossThad.item()/3,step=ii)
                exp.log_metric('loss_huber_Tlep', lossTlep.item()/3,step=ii)
                exp.log_metric('loss_huber_tot', regr_loss.item(),step=ii)
                exp.log_metric('loss_tot_train', loss_final.item(),step=ii)

            sum_loss += loss_final.item()
        
        exp.log_metric("loss_epoch_total_train", sum_loss/N_train, epoch=e, step=ii)
        
        valid_loss_huber = 0.
        valid_loss_mmd = 0.
        valid_lossH = 0.
        valid_lossTlep = 0.
        valid_lossThad = 0.
        valid_lossGluon = 0.
        valid_loss_final = 0.
        valid_mmd_H = 0.
        valid_mmd_thad = 0.
        valid_mmd_tlep = 0.
 
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
                MMD_input = torch.cat((higgs.unsqueeze(1),
                                       thad.unsqueeze(1),
                                       tlep.unsqueeze(1)), dim=1)
                # compute gluon
                # HttISR_regressed, boost_regressed = Compute_ParticlesTensor.get_HttISR_numpy(out, log_mean,
                #                                                                          log_std, device,
                #                                                                          cartesian=False,
                #                                                                          eps=1e-5)
                # gluon = HttISR_regressed[:,-3:]
                
                MMD_target = logScaled_partons[:,0:3]

                lossH, lossThad, lossTlep, = compute_regr_losses(logScaled_partons, higgs, thad, tlep,
                                                                      config.cartesian, loss_fn,
                                                    scaling_phi=[log_mean[2], log_std[2]], # Added scaling for phi
                                                           split=True,
                                                           device=device)

                regr_loss =  (lossH + lossThad + lossTlep)/9
                
                mmd_loss_H, mmd_loss_thad, mmd_loss_tlep = compute_mmd_loss(MMD_input, MMD_target, kernel=config.training_params.mmd_kernel,
                                            device=device, split=True)
                mmd_loss = (mmd_loss_H+ mmd_loss_thad + mmd_loss_tlep)/3
                                
                mdmm_return = MDMM_module(regr_loss, [(MMD_input, MMD_target, config.training_params.mmd_kernel, device, False)])

                valid_loss_huber += regr_loss.item()
                valid_lossH += lossH.item()/3
                valid_lossTlep += lossTlep.item()/3
                valid_lossThad += lossThad.item()/3
                # valid_lossGluon += lossGluon.item()/3
                valid_loss_mmd += mmd_loss.item()
                valid_mmd_H += mmd_loss_H.item()
                valid_mmd_thad += mmd_loss_thad.item()
                valid_mmd_tlep += mmd_loss_tlep.item()
            
                valid_loss_final += mdmm_return.value.item()

                particle_list = [higgs, thad, tlep]
                
                if i == 0:
                    for particle in range(len(particle_list)): # 4 or 3 particles: higgs/thad/tlep/gluonISR
                        
                        # 4 or 3 features
                        for feature in range(config.conditioning_transformer.out_features):  
                            exp.log_histogram_3d(particle_list[particle][:,feature].flatten().cpu().numpy(),
                                                 name = f"particle_regressed_{particle}_{feature}",
                                                 epoch=e, step=e)

                            exp.log_histogram_3d(logScaled_partons[:,particle,feature].cpu().numpy(),
                                                 name = f"particle_truth_{particle}_{feature}",
                                                 epoch=e, step=e)

                            exp.log_histogram_3d(particle_list[particle][:,feature].flatten().cpu().numpy()-\
                                                 logScaled_partons[:,particle,feature].cpu().numpy(),
                                                 name = f"diff_particle_{particle}_{feature}",
                                                 epoch=e, step=e)

                            fig, ax = plt.subplots(figsize=(7,6), dpi=100)
                            h = ax.hist2d(logScaled_partons[:,particle,feature].detach().cpu().numpy(),
                                          particle_list[particle][:,feature].cpu().detach().numpy().flatten(),
                                          bins=40, range=((-3, 3),(-3, 3)), cmin=1)
                            fig.colorbar(h[3], ax=ax)
                            exp.log_figure(f"particle_2D_{particle}_{feature}", fig)

                    
                            fig, ax = plt.subplots(figsize=(7,6), dpi=100)
                            ax.hist(logScaled_partons[:,particle,feature].detach().cpu().numpy(),
                                          bins=30, range=(-2, 2), label="truth", histtype="step")
                            ax.hist(particle_list[particle][:,feature].cpu().detach().numpy().flatten(),
                                          bins=30, range=(-2, 2), label="regressed",histtype="step")
                            ax.legend()
                            ax.set_xlabel(f"particle {particle} feature {feature}")
                            exp.log_figure(f"particle_1D_{particle}_{feature}", fig)
                            
         
       
        exp.log_metric("loss_total_val", valid_loss_final/N_valid, epoch=e )
        exp.log_metric("loss_mmd_val", valid_loss_mmd/N_valid,epoch= e)
        exp.log_metric('loss_huber_val', valid_loss_huber/N_valid,epoch= e)
        exp.log_metric('loss_huber_val_H', valid_lossH/N_valid,epoch= e)
        exp.log_metric('loss_huber_val_Tlep', valid_lossTlep/N_valid,epoch= e)
        exp.log_metric('loss_huber_val_Thad', valid_lossThad/N_valid,epoch= e)
        exp.log_metric("loss_mmd_val_H", valid_mmd_H/N_valid,epoch= e)
        exp.log_metric("loss_mmd_val_thad", valid_mmd_thad/N_valid,epoch= e)
        exp.log_metric("loss_mmd_val_tlep", valid_mmd_tlep/N_valid,epoch= e)

        if early_stopper.early_stop(valid_loss_final/N_valid,
                                    model.state_dict(), optimizer.state_dict(), modelName, exp):
            print(f"Model converges at epoch {e} !!!")         
            break

        scheduler.step(valid_loss_final/N_valid) # reduce lr if the model is not improving anymore

    # writer.close()
    # exp_log.end()

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
        env_var = os.environ.get("CUDA_VISIBLE_DEVICES")
        if env_var:
            actual_devices = env_var.split(",")
        else:
            actual_devices = list(range(torch.cuda.device_count()))
        print("Actual devices: ", actual_devices)
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
                                    out_features_latent=conf.conditioning_transformer.out_features_latent,
                                    no_layers_decoder_latent=conf.conditioning_transformer.no_layers_decoder_latent,     
                                    dtype=torch.float64)


    print(f"parameters total:{count_parameters(model)}\n")

    if (device == torch.device('cuda')):
        if len(actual_devices)>1:
            model = torch.nn.DataParallel(model)

        print("Moving the model to device")
        model = model.to(device)
    
        TrainingAndValidLoop(conf, device, model, train_loader, val_loader, outputDir, use_huberLoss)
    else:
        TrainingAndValidLoop(conf, device, model, train_loader, val_loader, outputDir, use_huberLoss)
        
    
    print(f"Normal version: preTraining finished succesfully! Version: {conf.version}")
    
    
    
    
    
    
    
    
    
    
