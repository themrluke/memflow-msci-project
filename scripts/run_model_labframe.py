from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
from memflow.unfolding_flow.unfolding_flow import UnfoldingFlow
from memflow.read_data.dataset_all import DatasetCombined
from memflow.unfolding_network.conditional_transformer import ConditioningTransformerLayer
from torch.optim.lr_scheduler import CosineAnnealingLR
from earlystop import EarlyStopper
import yaml
from yaml.loader import SafeLoader
import torch
import mdmm
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

PI = torch.pi

def BiasLoss(PS_Target, Flow_Sample):

    DiffSamplingTarget = PS_Target - Flow_Sample # [samples, events, Rambo]
    DiffSamplingTarget_transposed = torch.transpose(DiffSamplingTarget, 0, 1) # [events, samples, Rambo]

    mean_BiasOverSamples = torch.abs(torch.mean(DiffSamplingTarget_transposed, dim=1))  # [events, Rambo]

    mean_BiasOverEvents = torch.mean(mean_BiasOverSamples, dim=0) # [Rambo]
    std_BiasOverEvents = torch.std(mean_BiasOverSamples, dim=0) # [Rambo]

    mean_BiasOverRambo = torch.mean(mean_BiasOverEvents, dim=0) # [1]
    std_BiasOverRambo = torch.mean(std_BiasOverEvents, dim=0) # [1]

    return mean_BiasOverRambo, std_BiasOverRambo

def BiasLoss_Std(std_BiasOverRambo):

    return std_BiasOverRambo


def loss_fn_periodic(inp, target, loss_fn, device):

    # rescale to pi
    inp[inp>PI] = inp[inp>PI]-2*PI
    inp[inp<-PI] = inp[inp<-PI] + 2*PI
    
    deltaPhi = target - inp
    deltaPhi = torch.where(deltaPhi > PI, deltaPhi - 2*PI, deltaPhi)
    deltaPhi = torch.where(deltaPhi <= -PI, deltaPhi + 2*PI, deltaPhi)
    
    return loss_fn(deltaPhi, torch.zeros(deltaPhi.shape, device=device))


def compute_regr_losses(logScaled_partons, logScaled_boost, higgs, thad, tlep, boost, cartesian, loss_fn,
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

        lossBoost = loss_fn(logScaled_boost, boost)

    if split:
        return lossH, lossThad, lossTlep, lossBoost#, lossGluon
    else:
        return  (lossH + lossThad + lossTlep + lossBoost)/11


def compute_mmd_regr_loss(mmd_input, mmd_target, kernel, device, split=False):
    mmds = []
    for particle in range(len(mmd_input)):
        mmds.append(MMD(mmd_input[particle], mmd_target[particle], kernel, device))
    if split:
        return mmds
    else:
        return sum(mmds)/len(mmd_input)

    


def TrainingAndValidLoop(config, model, trainingLoader, validLoader, outputDir, alternativeTr, disableGradTransformer, HuberLoss):

    N_train = len(trainingLoader)
    N_valid = len(validLoader)

    # Define the constraint
    constraint_regr_huber = mdmm.MaxConstraint(
                    compute_regr_losses,
                    max=condig.MDMM.huber_max, # to be modified based on the regression
                    scale=condig.MDMM.huber_scale,
                    damping=condig.MDMM.huber_damping,
                    )
    constraint_regr_mmd = mdmm.MaxConstraint(
                    compute_mmd_regr_loss,
                    max=condig.MDMM.mmd_regr_max, # to be modified based on the regression
                    scale=condig.MDMM.mmd_regr_scale,
                    damping=condig.MDMM.mmd_regr_damping,
                    )

    # constraint_bias = mdmm.MaxConstraint(
    #                 BiasLoss_Std,
    #                 max=0.01, # to be modified based on the regression
    #                 scale=config.MDMM.eps_stdMean,
    #                 damping=5,
    #                 )

     if HuberLoss:
        loss_fn = torch.nn.HuberLoss(delta=config.training_params.huber_delta)
    else:
        loss_fn = torch.nn.MSELoss()
        
    # Create the optimizer
    MDMM_module = mdmm.MDMM([constraint_regr_huber,
                             constraint_regr_mmd])

    optimizer = MDMM_module.make_optimizer(model.parameters(), lr=config.training_params.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           factor=config.training_params.reduce_on_plateau.factor,
                                                           patience=config.training_params.reduce_on_plateau.patience,
                                                           threshold=config.training_params.reduce_on_plateau.threshold, verbose=True)
    
    

    outputDir = os.path.abspath(outputDir)
    name_dir = f'{outputDir}/flow_{config.name}_{config.version}_{config.unfolding_flow.base}_FirstArg:{config.unfolding_flow.base_first_arg}_NoTransf:{config.unfolding_flow.ntransforms}_NoBins:{config.unfolding_flow.bins}_DNN:{config.unfolding_flow.hiddenMLP_NoLayers}:{config.unfolding_flow.hiddenMLP_LayerDim}_epsMDMM:{config.MDMM.eps_regression}'
    modelName = f"{name_dir}/model_flow.pt"

    os.makedirs(name_dir, exist_ok=True)

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

    early_stopper = EarlyStopper(patience=config.training_params.nEpochsPatience, min_delta=config.training_params.early_stop_delta)

    # For debugging it slows down critically the jobs
    # torch.autograd.set_detect_anomaly(True)

    sampling_Forward = config.training_params.sampling_Forward
    N_samplesLoss = config.training_params.sampling_points_loss
    no_iterations = 1

    eps_logit = 5e-5
    
    for e in range(config.training_params.nepochs):
        
        sum_loss = 0.
        bias_sum_loss = 0.
        std_sum_loss = 0.

        if alternativeTr:
            if e % 2 == 0:
                sampling_Forward = False
                no_iterations = 1
            else:
                sampling_Forward = True
                no_iterations = config.training_params.subsplit
    
        # training loop    
        print("Before training loop")
        model.train()
        for i, data in enumerate(trainingLoader):
            
            if (i % 100 == 0):
                print(i)

            if (sampling_Forward and i > N_train*config.training_params.percentage_sampling_epoch):
                break

            optimizer.zero_grad()

            (logScaled_partons, logScales_boost,
            PS_target, PS_rambo_detjacobian,
            logScaled_reco, mask_lepton_reco, 
            mask_jets, mask_met, 
            mask_boost_reco, data_boost_reco) = data

            if (config.unfolding_flow.logit):
                mask0 = PS_target < eps_logit
                mask1 = PS_target > 1-eps_logit
                PS_target[mask0] = eps_logit
                PS_target[mask1] = 1-eps_logit
                PS_target = torch.logit(PS_target)
            else:
                mask0 = PS_target < 0
                mask1 = PS_target > 1
                if (mask0.any() or mask1.any()):
                    print('PS target < 0 or > 1')
                    exit(0)

            # here i start to subsplit -> do this because too much memory is required to sample
            # if the batch size is kept fixed (like in NormalizingDirection training))
            for ii in range(no_iterations):

                NoElems = config.training_params.batch_size_validation//no_iterations
                firstElem = ii*NoElems
                lastElem = (ii+1)*NoElems

                if firstElem > PS_target.size(0):
                    break
                elif lastElem > PS_target.size(0):
                    lastElem = PS_target.size(0)

                # Runs the forward pass with autocasting.
                # with autocast(device_type='cuda', dtype=torch.float16):
                #if True:
                    cond_X, PS_regressed = model(mask_jets=mask_jets[firstElem:lastElem],
                                             mask_lepton_reco=mask_lepton_reco[firstElem:lastElem],
                                             mask_met=mask_met[firstElem:lastElem],
                                             mask_boost_reco=mask_boost_reco[firstElem:lastElem],
                                             logScaled_reco=logScaled_reco[firstElem:lastElem],
                                             data_boost_reco=data_boost_reco[firstElem:lastElem], 
                                             device=device,
                                             noProv=config.noProv,
                                             eps=config.training_params.eps,
                                             order=config.training_params.order,
                                             disableGradTransformer=disableGradTransformer)

                if (config.unfolding_flow.logit):
                    mask0 = PS_regressed < eps_logit
                    mask1 = PS_regressed > 1-eps_logit
                    PS_regressed[mask0] = eps_logit
                    PS_regressed[mask1] = 1-eps_logit
                    PS_regressed = torch.logit(PS_regressed)

                detjac = PS_rambo_detjacobian[firstElem:lastElem].log()

                if sampling_Forward:

                    # rsample for sampling with grads
                    flow_sample = model.flow(PS_regressed).rsample((N_samplesLoss,)) # size [100,1024,10]
                    PS_target_masked = PS_target[firstElem:lastElem] # size [1,1024,10]

                    # MASK TO REMOVE EVENTS IF SAMPLING IS OUTSIDE 0-1
                    #sample_mask_all = (flow_sample>=0) & (flow_sample<=1)
                    #sample_mask_all = torch.transpose(sample_mask_all, 0, 1) #[events,samples]
                    #sample_mask = torch.all(sample_mask_all, dim=2) # reduce the last dimension
                    #sample_mask_events = torch.all(sample_mask, dim=1) # mask for the events with good samples
                    #flow_sample = flow_sample[:, sample_mask_events] # remove the events with bad samples
                    #PS_target_masked = PS_target_masked[sample_mask_events] # remove the events with bad samples

                    mean_BiasOverRambo, std_BiasOverRambo = BiasLoss(PS_target_masked, flow_sample)

                    mdmm_return = MDMM_module_bias(mean_BiasOverRambo, [(std_BiasOverRambo)])

                    flow_loss = mdmm_return.value
                    bias_sum_loss += mean_BiasOverRambo.item()
                    std_sum_loss += std_BiasOverRambo.item()

                else:
                    flow_prob = model.flow(PS_regressed).log_prob(PS_target[firstElem:lastElem])
                    flow_loss = -flow_prob

                inf_mask = torch.isinf(flow_loss)
                nonzeros = torch.count_nonzero(inf_mask)
                #writer.add_scalar(f"Number_INF_flow_{e}", nonzeros.item(), i)

                detjac_mean = -detjac.nanmean()

                if sampling_Forward:
                    loss = flow_loss
                else:
                    loss = flow_loss[torch.logical_not(inf_mask)].mean() # dont take infinities into consideration

                higgs = cond_X[0]
                thad = cond_X[1]
                tlep = cond_X[2]

                mdmm_return = MDMM_module(loss, [(logScaled_partons[firstElem:lastElem], higgs, thad, tlep,
                                                    config.cartesian, loss_fn, device)])

                mdmm_return.value.backward()
                optimizer.step()

                sum_loss += mdmm_return.value.item()

                writer.add_scalar(f"detJacOnly_epoch_step", detjac_mean.item(), i)
                if sampling_Forward:
                    writer.add_scalar(f"Loss_step_train_epoch_step_SamplingDir_MDMMLoss", loss.item(), i)
                    writer.add_scalar(f"Loss_step_train_epoch_step_SamplingDir_BiasMeanLoss", mean_BiasOverRambo.item(), i)
                    writer.add_scalar(f"Loss_step_train_epoch_step_SamplingDir_StdMeanLoss", std_BiasOverRambo.item(), i)
                else:
                    writer.add_scalar(f"Loss_step_train_epoch_step_Normalizing_dir", loss.item(), i)


        if sampling_Forward:
            writer.add_scalar(f"Loss_epoch_train_SamplingDir_MDMMLoss", sum_loss/i/no_iterations, e)
            writer.add_scalar(f"Loss_epoch_train_SamplingDir_BiasMeanLoss", bias_sum_loss/i/no_iterations, e)
            writer.add_scalar(f"Loss_epoch_train_SamplingDir_StdMeanLoss", std_sum_loss/i/no_iterations, e)
        else:
            writer.add_scalar(f"Loss_epoch_train_NormalizingDir", sum_loss/i/no_iterations, e)

        valid_loss = 0.
        bias_sum_valid = 0.
        std_sum_valid = 0.

        # validation loop (don't update weights and gradients)
        print("Before validation loop")
        model.eval()
        for i, data in enumerate(validLoader):

            (logScaled_partons,
            PS_target,
            PS_rambo_detjacobian,
            logScaled_reco, mask_lepton_reco, 
            mask_jets, mask_met, 
            mask_boost_reco, data_boost_reco) = data

            if (config.unfolding_flow.logit):
                mask0 = PS_target < eps_logit
                mask1 = PS_target > 1-eps_logit
                PS_target[mask0] = eps_logit
                PS_target[mask1] = 1-eps_logit
                PS_target = torch.logit(PS_target)
            else:
                mask0 = PS_target < 0
                mask1 = PS_target > 1
                if (mask0.any() or mask1.any()):
                    print('PS target < 0 or > 1')
                    exit(0)

            with torch.no_grad():

                for ii in range(no_iterations):

                    NoElems = config.training_params.batch_size_validation//no_iterations
                    firstElem = ii*NoElems
                    lastElem = (ii+1)*NoElems

                    if firstElem > PS_target.size(0):
                        break
                    elif lastElem > PS_target.size(0):
                        lastElem = PS_target.size(0)

                    cond_X, PS_regressed = model(mask_jets=mask_jets[firstElem:lastElem], mask_lepton_reco=mask_lepton_reco[firstElem:lastElem],
                                                mask_met=mask_met[firstElem:lastElem], mask_boost_reco=mask_boost_reco[firstElem:lastElem],
                                                logScaled_reco=logScaled_reco[firstElem:lastElem], data_boost_reco=data_boost_reco[firstElem:lastElem], 
                                                device=device, noProv=config.noProv, eps=config.training_params.eps,
                                                order=config.training_params.order, disableGradTransformer=disableGradTransformer)

                    if (config.unfolding_flow.logit):
                        mask0 = PS_regressed < eps_logit
                        mask1 = PS_regressed > 1-eps_logit
                        PS_regressed[mask0] = eps_logit
                        PS_regressed[mask1] = 1-eps_logit
                        PS_regressed = torch.logit(PS_regressed)

                    detjac = PS_rambo_detjacobian[firstElem:lastElem].log()

                    if sampling_Forward:

                        flow_sample = model.flow(PS_regressed).sample((N_samplesLoss,)) # size [100,1024,10]
                        PS_target_masked = PS_target[firstElem:lastElem] # size [1,1024,10]

                        #sample_mask_all = (flow_sample>=0) & (flow_sample<=1)
                        #sample_mask_all = torch.transpose(sample_mask_all, 0, 1) #[events,samples]
                        #sample_mask = torch.all(sample_mask_all, dim=2) # reduce the last dimension
                        #sample_mask_events = torch.all(sample_mask, dim=1) # mask for the events with good samples
                        #flow_sample = flow_sample[:, sample_mask_events] # remove the events with bad samples
                        #PS_target_masked = PS_target_expanded[sample_mask_events] # remove the events with bad samples

                        mean_BiasOverRambo, std_BiasOverRambo = BiasLoss(PS_target_masked, flow_sample)
                        
                        mdmm_return = MDMM_module_bias(mean_BiasOverRambo, [(std_BiasOverRambo)])

                        flow_loss = mdmm_return.value
                        bias_sum_loss += mean_BiasOverRambo.item()
                        std_sum_loss += std_BiasOverRambo.item()

                    else:
                        flow_prob = model.flow(PS_regressed).log_prob(PS_target[firstElem:lastElem])
                        flow_loss = -flow_prob

                    inf_mask = torch.isinf(flow_loss)
                    nonzeros = torch.count_nonzero(inf_mask)
                    detjac_mean = -detjac.nanmean()
                    
                    if sampling_Forward:
                        loss = flow_loss
                    else:
                        loss = flow_loss[torch.logical_not(inf_mask)].mean() # dont take infinities into consideration
                    
                    higgs = cond_X[0]
                    thad = cond_X[1]
                    tlep = cond_X[2]

                    mdmm_return = MDMM_module(loss, [(logScaled_partons[firstElem:lastElem], higgs, thad, tlep,
                                                    config.cartesian, loss_fn, device)])
                    
                    valid_loss += mdmm_return.value.item()

                if i == 0:

                    N_samples = 100 # 100 x 1024 x 10
                    ps_new = model.flow(PS_regressed).sample((N_samples,))

                    data_ps_cpu = PS_target[firstElem:lastElem].detach().cpu()
                    ps_new_cpu = ps_new.detach().cpu()

                    mean_BiasOverRambo, std_BiasOverRambo = BiasLoss(data_ps_cpu, ps_new_cpu)

                    writer.add_scalar(f"Validation_Bias_Samples_Var", mean_BiasOverRambo.item(), e)
                    writer.add_scalar(f"Validation_Std_Samples_Var", std_BiasOverRambo.item(), e)

                    for x in range(data_ps_cpu.size(1)):
                        fig, ax = plt.subplots()
                        h = ax.hist2d(data_ps_cpu[:,x].tile(N_samples,1,1).flatten().numpy(),
                                        ps_new_cpu[:,:,x].flatten().numpy(),
                                        bins=50, range=((-12, 12),(-12, 12)))
                        fig.colorbar(h[3], ax=ax)
                        writer.add_figure(f"CheckElemOutside_Plot_{x}", fig, e)

                        #fig, ax = plt.subplots()
                        #h = ax.hist2d(data_ps_cpu[:,x].tile(N_samples,1,1).flatten().numpy(),
                        #            ps_new_cpu[:,:,x].flatten().numpy(),
                        #            bins=50, range=((-0.1, 1.1),(-0.1, 1.1)))
                        #fig.colorbar(h[3], ax=ax)
                        #writer.add_figure(f"Validation_ramboentry_Plot_{x}", fig, e)

                        fig, ax = plt.subplots()
                        h = ax.hist(
                            (data_ps_cpu[:,x].tile(N_samples,1,1) - ps_new_cpu[:,:,x]).flatten().numpy(),
                            bins=100)
                        writer.add_figure(f"Validation_ramboentry_Diff_{x}", fig, e)

        valid_loss = valid_loss/i/no_iterations

        if sampling_Forward:
            bias_sum_valid = bias_sum_valid/i/no_iterations
            std_sum_valid = std_sum_valid/i/no_iterations
            writer.add_scalar(f"Loss_epoch_val_SamplingDir_MDMMLoss", valid_loss, e)
            writer.add_scalar(f"Loss_epoch_val_SamplingDir_BiasLoss", bias_sum_valid, e)
            writer.add_scalar(f"Loss_epoch_val_SamplingDir_StdMeanLoss", std_sum_valid, e)
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
    parser.add_argument('--huberLoss', action="store_true",  help='use Huber loss')
    args = parser.parse_args()
    
    path_to_dir = args.model_dir
    output_dir = f'{args.model_dir}/{args.output_dir}'
    on_GPU = args.on_GPU # by default run on CPU
    alternativeTr = args.alternativeTr # by default do the training in a specific direction
    disableGradTransf = False
    use_huberLoss = args.huberLoss

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
                                parton_list=['logScaled_data_higgs_t_tbar_ISR_cartesian',
                                            'phasespace_intermediateParticles_onShell',
                                            'phasespace_rambo_detjacobian_onShell'])
    else:
        data = DatasetCombined(conf.input_dataset, dev=device, dtype=torch.float64,
                                reco_list=['scaledLogRecoParticles', 'mask_lepton', 
                                            'mask_jets','mask_met',
                                            'mask_boost', 'data_boost'],
                                parton_list=['logScaled_data_higgs_t_tbar_ISR',
                                            'phasespace_intermediateParticles_onShell',
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
                    affine_param_input1=conf.unfolding_flow.affine_param_input1,
                    affine_param_input2=conf.unfolding_flow.affine_param_input2,
                    affine_param_output1=conf.unfolding_flow.affine_param_output1,
                    affine_param_output2=conf.unfolding_flow.affine_param_output2,
                    randPerm=conf.unfolding_flow.randPerm,
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
        TrainingAndValidLoop(conf, model, train_loader, val_loader, output_dir, alternativeTr, disableGradTransf, use_huberLoss)
        #else:
        #    TrainingAndValidLoop(conf, model, train_loader, val_loader, output_dir, alternativeTr, disableGradTransf)
    else:
        TrainingAndValidLoop(conf, model, train_loader, val_loader, output_dir, alternativeTr, disableGradTransf, use_huberLoss)
        
    print("Training finished succesfully!")
    
    
    
    
    
    
    
    
    
    
    
    
    
