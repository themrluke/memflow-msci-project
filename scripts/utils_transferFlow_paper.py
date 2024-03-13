from hist import Hist
import torch
import hist
import awkward as ak
import numpy as np
import os
import mplhep
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib as mpl

prov = {
  "higgs": 1,
  "thad": 2,
  "tlep": 3
}

spanet_columns = {
  "higgs": 2,
  "thad": 0,
  "tlep": 1
}

def sample_next_token(model, logScaled_reco_sortedBySpanet, logScaled_partons, mask_reco):
    # create null token and its mask
    null_token = torch.ones((logScaled_reco_sortedBySpanet.shape[0], 1, 4), device=model.device, dtype=model.dtype) * -1
    null_token[:,0,0] = 0 # exist flag = 0 not -1
    # mask for the null token = True
    null_token_mask = torch.ones((mask_reco.shape[0], 1), device=model.device, dtype=torch.bool)

    # attach null token and update the mask for the scaling_reco_lab
    scaling_reco_lab_withNullToken = torch.cat((null_token, logScaled_reco_sortedBySpanet), dim=1)
    mask_reco_withNullToken = torch.cat((null_token_mask, mask_reco), dim=1)
    
    scaledLogReco_afterLin = model.gelu(model.linearDNN_reco(scaling_reco_lab_withNullToken) * mask_reco_withNullToken[..., None])
    scaledLogParton_afterLin = model.gelu(model.linearDNN_parton(logScaled_partons))  
        
    tgt_mask = model.transformer_model.generate_square_subsequent_mask(scaledLogReco_afterLin.size(1), device=model.device)

    if model.dtype == torch.float32:
        tgt_mask = tgt_mask.float()
    elif dtype == torch.float64:
        tgt_mask = tgt_mask.double()
        
    output_decoder = scaledLogReco_afterLin

    for transfermer in model.transformer_list:
        output_decoder = transfermer(scaledLogParton_afterLin, output_decoder, tgt_mask=tgt_mask)

    # take the last conditioning (the one on jet_0 ... jet_-1) 
    conditioning_exist = output_decoder[:,-1:]
    jetExist_sampled = model.flow_exist(conditioning_exist).rsample((1,))
    jetExist_sampled = jetExist_sampled.squeeze(dim=0)
    # remake the `exist` flag discrete
    jetExist_sampled = torch.where(jetExist_sampled < 0.5, 0, 1)

    # take the last conditioning (the one on jet_0 ... jet_-1) 
    conditioning_pt = output_decoder[:,-1:]
    jetsPt_sampled = model.flow_pt(conditioning_pt).rsample((1,))
    jetsPt_sampled = jetsPt_sampled.squeeze(dim=0)
    # if sampled_exist == 0 => pt_sampled = -1
    jetsPt_sampled = torch.where(jetExist_sampled == 0, -1, jetsPt_sampled)

    # take the last conditioning (the one on jet_0 ... jet_-1) + sampled_pt
    conditioning_eta = torch.cat((output_decoder[:,-1:], jetsPt_sampled), dim=2)
    jetsEta_sampled = model.flow_eta(conditioning_eta).rsample((1,))
    jetsEta_sampled = jetsEta_sampled.squeeze(dim=0)
    # if sampled_exist == 0 => eta_sampled = -1
    jetsEta_sampled = torch.where(jetExist_sampled == 0, -1, jetsEta_sampled)

    # take the last conditioning (the one on jet_0 ... jet_-1)  + sampled_pt + sampled_eta
    conditioning_phi = torch.cat((output_decoder[:,-1:], jetsPt_sampled, jetsEta_sampled), dim=2)
    jetsPhi_sampled = model.flow_phi(conditioning_phi).rsample((1,))
    jetsPhi_sampled = jetsPhi_sampled.squeeze(dim=0)
    # if sampled_exist == 0 => phi_sampled = -1
    jetsPhi_sampled = torch.where(jetExist_sampled == 0, -1, jetsPhi_sampled)

    generated_jet = torch.cat((jetExist_sampled, jetsPt_sampled, jetsEta_sampled, jetsPhi_sampled), dim=2)

    # return generated_jet including its position
    return generated_jet


def sample_fullRecoEvent(model, logScaled_partons, no_events, device, dtype, No_samples=1):

    fullGeneratedEvent = torch.empty((no_events, 0, 4), device=device, dtype=dtype)
    
    for j in range(model.no_max_objects):

        next_jet = sample_next_token(model, fullGeneratedEvent, logScaled_partons, mask_reco)
    
        fullGeneratedEvent = torch.cat((fullGeneratedEvent, next_jet), dim=1)

    return fullGeneratedEvent

def sample_next_token_classifier(model, logScaled_reco_sortedBySpanet, logScaled_partons, mask_reco):
    # create null token and its mask
    null_token = torch.ones((logScaled_reco_sortedBySpanet.shape[0], 1, 5), device=model.device, dtype=model.dtype) * -1
    null_token[:,0,0] = 0 # exist flag = 0 not -1
    # mask for the null token = True
    null_token_mask = torch.ones((mask_reco.shape[0], 1), device=model.device, dtype=torch.bool)

    # attach null token and update the mask for the scaling_reco_lab
    scaling_reco_lab_withNullToken = torch.cat((null_token, logScaled_reco_sortedBySpanet), dim=1)
    mask_reco_withNullToken = torch.cat((null_token_mask, mask_reco), dim=1)
    
    scaledLogReco_afterLin = model.gelu(model.linearDNN_reco(scaling_reco_lab_withNullToken) * mask_reco_withNullToken[..., None])
    scaledLogParton_afterLin = model.gelu(model.linearDNN_parton(logScaled_partons))  
        
    tgt_mask = model.transformer_model.generate_square_subsequent_mask(scaledLogReco_afterLin.size(1), device=model.device)

    if model.dtype == torch.float32:
        tgt_mask = tgt_mask.float()
    elif dtype == torch.float64:
        tgt_mask = tgt_mask.double()

    # classifier part
    output_decoder_classifier = model.classifier_exist.transformer_model(scaledLogParton_afterLin, scaledLogReco_afterLin, tgt_mask=tgt_mask)

    if model.encode_position:
        hot_encoded = model.classifier_exist.hot_encoded.expand(output_decoder_classifier.shape[0], -1, -1)
        output_decoder_classifier = torch.cat((output_decoder_classifier, hot_encoded[:,:output_decoder_classifier.shape[1]]), dim=2)

    # take the last jetExist_sampled[:,-1:] -> for the last jet
    prob_each_jet = model.classifier_exist.model(output_decoder_classifier[:,-1:]).squeeze(dim=2)
    jetExist_sampled = torch.where(prob_each_jet < 0.5, 0, 1).unsqueeze(dim=2) # match dimension with 'jetsPt_sampled'

    # flow part
    output_decoder = model.transformer_model(scaledLogParton_afterLin, scaledLogReco_afterLin, tgt_mask=tgt_mask)

    if model.encode_position:
        hot_encoded = model.classifier_exist.hot_encoded.expand(output_decoder.shape[0], -1, -1)
        output_decoder = torch.cat((output_decoder, hot_encoded[:,:output_decoder.shape[1]]), dim=2)

    # take the last conditioning (the one on jet_0 ... jet_-1) 
    conditioning_pt = output_decoder[:,-1:]
    jetsPt_sampled = model.flow_pt(conditioning_pt).rsample((1,))
    jetsPt_sampled = jetsPt_sampled.squeeze(dim=0)
    # if sampled_exist == 0 => pt_sampled = -1
    jetsPt_sampled = torch.where(jetExist_sampled == 0, -1, jetsPt_sampled) 

    # take the last conditioning (the one on jet_0 ... jet_-1) + sampled_pt
    conditioning_eta = torch.cat((output_decoder[:,-1:], jetsPt_sampled), dim=2)
    jetsEta_sampled = model.flow_eta(conditioning_eta).rsample((1,))
    jetsEta_sampled = jetsEta_sampled.squeeze(dim=0)
    # if sampled_exist == 0 => eta_sampled = -1
    jetsEta_sampled = torch.where(jetExist_sampled == 0, -1, jetsEta_sampled)

    # take the last conditioning (the one on jet_0 ... jet_-1)  + sampled_pt + sampled_eta
    conditioning_phi = torch.cat((output_decoder[:,-1:], jetsPt_sampled, jetsEta_sampled), dim=2)
    jetsPhi_sampled = model.flow_phi(conditioning_phi).rsample((1,))
    jetsPhi_sampled = jetsPhi_sampled.squeeze(dim=0)
    # if sampled_exist == 0 => phi_sampled = -1
    jetsPhi_sampled = torch.where(jetExist_sampled == 0, -1, jetsPhi_sampled)

    # get the first dimension of 'logScaled_reco_sortedBySpanet'
    # [1:2] to save it as a tensor
    position_jet = torch.tensor(list(logScaled_reco_sortedBySpanet.shape[1:2]), device=model.device)
    position_jet = position_jet.expand(output_decoder.shape[0], 1, 1)
    if position_jet[0,0,0] > 8:
        position_jet[:,0,0] = 8

    generated_jet = torch.cat((jetExist_sampled, jetsPt_sampled, jetsEta_sampled, jetsPhi_sampled, position_jet), dim=2)

    return generated_jet


def sample_fullRecoEvent_classifier(model, logScaled_partons, no_events, device, dtype, No_samples=1):
    
    fullGeneratedEvent = torch.empty((no_events, 0, 5), device=device, dtype=dtype)
    mask_reco = torch.empty((no_events, 0), device=device, dtype=dtype)
    mask_one = torch.ones((no_events, 1), device=device, dtype=dtype)
    
    for j in range(model.no_max_objects):

        next_jet = sample_next_token_classifier(model, fullGeneratedEvent, logScaled_partons, mask_reco)
    
        fullGeneratedEvent = torch.cat((fullGeneratedEvent, next_jet), dim=1)

        # update the mask
        mask_reco = torch.cat((mask_reco, mask_one), dim=1)
        # if I pass the MET position and the existance == False => the next jets are padding jets
        if j > 7:
            mask_reco[:,j] = torch.where(fullGeneratedEvent[:,j,0] == 1, 1, 0)        

    return fullGeneratedEvent

def existQuality_print(experiment, sampledEvent, logScaled_reco_target, plotJets, epoch):
    # check exist flag
    target_exist = logScaled_reco_target[:,plotJets,0]
    sampled_exist = sampledEvent[:,plotJets,0]

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

def sampling_print(experiment, sampledEvent, logScaled_reco_target, mask_recoParticles, plotJets, epoch, onlyExistElem=False):

    # plot [pt,eta,phi]
    var_name = ['pt', 'eta', 'phi']
    
    partialMaskReco = mask_recoParticles[:,plotJets]
    partialMaskReco = partialMaskReco.bool()

    if onlyExistElem:
        check_target_events_withExist = logScaled_reco_target[:,plotJets,0] == 1
        partialMaskReco = torch.logical_and(partialMaskReco, check_target_events_withExist)

    # keep objects starting from pt=1
    fullGeneratedEvent_fromPt = sampledEvent[:,plotJets,1:]
    maskedGeneratedEvent = fullGeneratedEvent_fromPt[partialMaskReco]

    # keep objects starting from pt=1
    partial_logScaled_reco_sortedBySpanet = logScaled_reco_target[:,plotJets,1:]
    maskedTargetEvent = partial_logScaled_reco_sortedBySpanet[partialMaskReco]

    # check pt,eta,phi distrib
    for plot_var in range(3):

        fig, ax = plt.subplots(figsize=(7,6), dpi=100)
        diff_generatedAndTarget = (maskedGeneratedEvent[:,plot_var] - maskedTargetEvent[:,plot_var])
        ax.hist(diff_generatedAndTarget.detach().cpu().numpy(), range=(-5,5), bins=20, histtype='step', color='b', stacked=False, fill=False)
        ax.set_xlabel(f'{var_name[plot_var]}_generated - {var_name[plot_var]}_target')
        experiment.log_figure(f"Diff_generated_{var_name[plot_var]} for jets{plotJets}", fig, step=epoch)

        fig, ax = plt.subplots(figsize=(7,6), dpi=100)
        h = ax.hist2d(maskedGeneratedEvent[:,plot_var].detach().cpu().numpy(),
                      maskedTargetEvent[:,plot_var].detach().cpu().numpy(),
                      bins=30, range=[(-5,5),(-5,5)], cmin=1)
        fig.colorbar(h[3], ax=ax)
        ax.set_xlabel(f'sampled {var_name[plot_var]}')
        ax.set_ylabel(f'target {var_name[plot_var]}')
        experiment.log_figure(f"2D_correlation_{var_name[plot_var]} for jets{plotJets}", fig,step=epoch)

def validation_print(experiment, flow_pr, wrong_pt_batch_flow_pr, wrong_ptAndEta_batch_flow_pr, epoch, range_x=(-60,60), no_bins=100,
                    label1='diff: pt_0 10%', label2='diff: pt_0 10% and eta', particles='jets'):
    # Valid 1             
    fig, ax = plt.subplots(figsize=(7,6), dpi=100)
    ax.hist(flow_pr.detach().cpu().numpy(), range=range_x, bins=no_bins, histtype='step', label='target', color='b', stacked=False, fill=False)
    ax.hist(wrong_pt_batch_flow_pr.detach().cpu().numpy(), range=range_x, bins=75, histtype='step', label=label1, color='r', stacked=False, fill=False)
    plt.legend()
    ax.set_xlabel('+ logprob')
    experiment.log_figure(f"validation_figure_1 {particles}", fig, step=epoch)
                    
    # Valid 2
    fig, ax = plt.subplots(figsize=(7,6), dpi=100)
    ax.hist(flow_pr.detach().cpu().numpy(), range=range_x, bins=75, histtype='step', label=f'target', color='r', stacked=False, fill=False)
    ax.hist(wrong_ptAndEta_batch_flow_pr.detach().cpu().numpy(), range=range_x, bins=no_bins, histtype='step', label=label2, color='g', stacked=False, fill=False)
    plt.legend()
    ax.set_xlabel('+ logprob')
    experiment.log_figure(f"validation_figure_2 {particles}", fig, step=epoch)

    # Diff valid 1      
    fig, ax = plt.subplots(figsize=(7,6), dpi=100)
    ax.hist((flow_pr - wrong_pt_batch_flow_pr).detach().cpu().numpy(), range=(-5,5), bins=20, histtype='step', color='b', stacked=False, fill=False)
    ax.set_xlabel('target - pt_altered (logprob)')
    experiment.log_figure(f"Diff_log_prob_1 {particles}", fig, step=epoch)
    

    # Diff valid 2          
    fig, ax = plt.subplots(figsize=(7,6), dpi=100)
    ax.hist((flow_pr - wrong_ptAndEta_batch_flow_pr).detach().cpu().numpy(), range=(-5,5), bins=20, histtype='step', color='b', stacked=False, fill=False)
    ax.set_xlabel('pt_altered - ptAndEta_altered (logprob)')
    experiment.log_figure(f"Diff_log_prob_2 {particles}", fig, step=epoch)

    # Correct vs wrong 1
    correct_model_1 = flow_pr > wrong_pt_batch_flow_pr
    no_correct_1 = torch.count_nonzero(correct_model_1).cpu().numpy()
    no_wrong_1 = len(flow_pr) - no_correct_1
        
    fig, ax = plt.subplots(figsize=(7,6), dpi=100)
    ax.bar(["correct", "wrong"], [no_correct_1, no_wrong_1], color ='maroon', width = 0.4)
    experiment.log_figure(f"Correct_wrong_1 {particles}", fig, step=epoch)

    # Correct vs wrong 2
    correct_model_2 = flow_pr > wrong_ptAndEta_batch_flow_pr
    no_correct_2 = torch.count_nonzero(correct_model_2).cpu().numpy()
    no_wrong_2 = len(flow_pr) - no_correct_2

    fig, ax = plt.subplots(figsize=(7,6), dpi=100)
    ax.bar(["correct", "wrong"], [no_correct_2, no_wrong_2], color ='maroon', width = 0.4)
    experiment.log_figure(f"Correct_wrong_2 {particles}", fig, step=epoch)

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
            loss_per_pt[i] =  -1*flow_pr[mask_pt].mean()

    return loss_per_pt
