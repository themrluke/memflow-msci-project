import math
import torch
from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import Callback

from abc import ABCMeta, abstractmethod

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Nimbus Roman']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Nimbus Roman'
plt.rcParams['mathtext.it'] = 'Nimbus Roman:italic'
plt.rcParams['mathtext.bf'] = 'Nimbus Roman:bold'
plt.rcParams["text.usetex"] = False

EPS = 1e-12

class BaseCallback(Callback,metaclass=ABCMeta):
    def __init__(self,dataset,selection,features_per_type,preprocessing,suffix=None,frequency=1,raw=False,bins=None,log_scale=False,device=None,batch_size=1024,N_batch=np.inf,label_names={}):
        super().__init__()

        # Attributes #
        self.dataset = dataset
        self.selection = selection
        self.features_per_type = features_per_type
        self.preprocessing = preprocessing
        self.suffix = suffix
        self.loader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
        self.N_batch = N_batch
        self.frequency = frequency
        self.raw = raw
        self.bins = bins
        self.log_scale = log_scale
        self.label_names = label_names
        self.device = device

    def on_validation_epoch_end(self,trainer,pl_module):
        if trainer.sanity_checking:  # optional skip
            return
        if trainer.current_epoch == 0:
            return
        if trainer.current_epoch % self.frequency != 0:
           return

        # Get figures #
        figs = self.make_plots(pl_module,disable_tqdm=True,show=False)

        # Log them #
        for figure_name,figure in figs.items():
            trainer.logger.experiment.log_figure(
                figure_name = figure_name,
                figure = figure,
                overwrite = True,
                step = trainer.current_epoch,
            )
            plt.close(figure)

    def predict(self,model,disable_tqdm=False):
        # Select device #
        if self.device is None:
            device = model.device
        else:
            device = self.device
        model = model.to(device)
        model.eval()

        # Loop over batch #
        inputs = []
        masks = []
        targets = []
        preds = []
        for batch_idx, batch in tqdm(enumerate(self.loader),desc='Predict',disable=disable_tqdm,leave=True,total=min(self.N_batch,len(self.loader)),position=0):
            if batch_idx >= self.N_batch:
                break

            inputs.append(batch['data'])
            targets.append(batch['target'])
            masks.append(batch['mask'])

            # Predict #
            batch = model.transfer_batch_to_device(batch, device, 0)
            with torch.no_grad():
                preds.append(model(batch).cpu())
                y = model(batch)

        # Concat #
        inputs = [
            torch.cat(
                [
                    inp[i]
                    for inp in inputs
                ],
                dim=0,
            )
            for i in range(len(inputs[0]))
        ]
        masks = [
            torch.cat(
                [
                    mask[i]
                    for mask in masks
                ],
                dim=0,
            )
            for i in range(len(masks[0]))
        ]
        targets = torch.cat(targets,dim=0)
        preds = torch.cat(preds,dim=0)

        return inputs, masks, targets, preds

    def make_plots(self,model,show=True,disable_tqdm=False):
        inputs, masks, targets, preds = self.predict(model,disable_tqdm)

        figs = {}

        # target vs preds plots #
        if hasattr(self,'make_prediction_plots') and callable(self.make_prediction_plots):
            figs.update(self.make_prediction_plots(targets,preds,self.suffix))

        # make per particle plots #
        figs.update(self.make_particle_plots(inputs,masks,targets,preds))

        if show:
            plt.show()

        return figs

    def make_particle_plots(self,inputs,masks,targets,preds):
        if not hasattr(self,'plot_particle') or not callable(self.plot_particle):
            return {}

        # Loop over particle types #
        figs = {}
        for k in range(len(inputs)):
            # Get type objects #
            inputs_type = inputs[k]
            masks_type = masks[k]
            name = self.selection[k]
            features = self.features_per_type[k]
            # Preprocessing #
            if self.raw:
                inputs_type, features = self.preprocessing.inverse(
                    name = name,
                    x = inputs_type,
                    mask = masks_type,
                    fields = features,
                )
            # Loop over particles #
            for j in range(inputs_type.shape[1]):
                # Make plot #
                title = f'{self.suffix} : ' if self.suffix is not None else ''
                title += f'{name} #{j}'
                fig = self.plot_particle(
                    inputs = inputs_type[:,j,:],
                    masks = masks_type[:,j],
                    targets = targets,
                    preds = preds,
                    features = features,
                    title = title,
                )
                figure_name = f'{self.suffix}_' if self.suffix is not None else ''
                figure_name += f'{name}_{j}'
                if fig is not None:
                    figs[figure_name] = fig

        return figs


    @abstractmethod
    def plot_particle(self,inputs,masks,targets,preds,features,title):
        pass




class AcceptanceCallback(BaseCallback):
    def __init__(self,min_selected_events_per_bin=None,**kwargs):
        self.min_selected_events_per_bin = min_selected_events_per_bin
        super().__init__(**kwargs)

    def make_binning(self,values,thresh=None):
        bin_content,bin_edges = np.histogram(values,self.bins)
        # No threshold, just use bins #
        if thresh is None:
            return bin_edges
        assert isinstance(thresh,(float,int))
        # Accumulate bins to have them always above threshold #
        acc_content = 0.
        final_bin_edges = []
        idx_last = 0
        for i in range(len(bin_content)):
            if len(final_bin_edges) == 0:
                # device where to put first bin
                if bin_content[i] > 0:
                    final_bin_edges.append(bin_edges[i])
            # Check if total of accumaulated bin is above threshold
            acc_content += bin_content[i]
            if acc_content > thresh:
                final_bin_edges.append(bin_edges[i+1])
                acc_content = 0.
            if bin_content[i] > 0:
                idx_last = i+1
        # If still some content in accumulation, add latest bin that had non-zero content
        if acc_content > 0:
            final_bin_edges.append(bin_edges[idx_last])
        return np.array(final_bin_edges)

    def plot_particle(self,inputs,masks,targets,preds,features,title):
        # Apply mask #
        if masks.sum() == 0:
            return None
        inputs  = inputs[masks]
        preds   = preds[masks]
        targets = targets[masks]

        # filter out mass and pdgid for poster plots
        filtered_features = []
        filtered_inputs = []
        for i, feature in enumerate(features):
            if feature.lower() not in ['mass', 'pdgid']:
                filtered_features.append(feature)
                filtered_inputs.append(inputs[:, i])
        
        if not filtered_features:
            return None
        
        inputs = np.column_stack(filtered_inputs)
        features = filtered_features

        # make figure #
        N = len(features)
        fig,axs = plt.subplots(ncols=N,nrows=2,figsize=(6*N,5),height_ratios=[1.0,0.2], dpi=400)
        plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.0)
        plt.suptitle(title,fontsize=16)

        # Make mask of selected particles #
        mask_sel = (targets > 0).ravel()

        # Loop over features #
        for i in range(N):
            # Determine the binning from selected events #
            if self.min_selected_events_per_bin is None:
                thresh = None
            elif isinstance(self.min_selected_events_per_bin,(float,int)):
                thresh = self.min_selected_events_per_bin
            elif isinstance(self.min_selected_events_per_bin,dict):
                if not features[i] in self.min_selected_events_per_bin.keys():
                    raise RuntimeError(f'Feature {features[i]} not in min_selected_events_per_bin choices {self.min_selected_events_per_bin.keys()}')
                thresh = self.min_selected_events_per_bin[features[i]]
            else:
                raise RuntimeError(f'Type {type(self.min_selected_events_per_bin)} of min_selected_events_per_bin not understood')
            binning = self.make_binning(inputs[mask_sel,i],thresh)
            # Make histogram of selected events only #
            content_selected, binning = np.histogram(inputs[mask_sel,i],bins=binning)
            # Make histogram of all hard events #
            content_feat, _ = np.histogram(inputs[:,i],bins=binning)
            # Make histogram of prediction #
            content_pred_weighted, _ = np.histogram(inputs[:,i],bins=binning,weights=preds.ravel())
            content_pred_tot, _ = np.histogram(inputs[:,i],bins=binning)
            # Make histogram of acceptance (true) #
            content_acc = content_selected / (content_feat + EPS)
            # Make uncertainty bands #
            var_acc = np.sqrt(content_selected) / (content_feat + EPS)
            content_acc_up = content_acc + var_acc
            content_acc_down = content_acc - var_acc
            # Make histogram of acceptance (pred) #
            content_pred_avg = content_pred_weighted / (content_pred_tot + EPS)
            # Make ratio #
            ratio_nom = content_pred_avg / (content_acc + EPS)
            ratio_var = var_acc / (content_acc + EPS)
            ratio_up = 1+abs(ratio_var)
            ratio_down = 1-abs(ratio_var)
            # Plot #
            axs[0,i].stairs(
                values = content_acc,
                edges = binning,
                color = 'royalblue',
                label = 'Truth',
                linewidth = 2,
            )
            axs[0,i].fill_between(
                x = binning,
                y1 = np.r_[content_acc_down,content_acc_down[-1]],
                y2 = np.r_[content_acc_up,content_acc_up[-1]],
                color = 'royalblue',
                alpha = 0.3,
                step = 'post',
            )
            axs[0,i].stairs(
                values = content_pred_avg,
                edges = binning,
                color = 'orange',
                label = 'Classifier',
                linewidth = 2,
            )
            # Ratio #
            axs[1,i].step(
                x = binning,
                y = np.r_[ratio_nom[0],ratio_nom],
                linewidth = 2,
                color = 'orange',
            )
            axs[1,i].fill_between(
                x = binning,
                y1 = np.r_[ratio_down,ratio_down[-1]],
                y2 = np.r_[ratio_up,ratio_up[-1]],
                color = 'royalblue',
                alpha = 0.3,
                step = 'post',
            )
            # Esthetic #
            axs[0,i].set_ylabel('Acceptance',fontsize=16,labelpad=12)
            axs[0,i].set_xticklabels([])
            axs[0,i].legend(loc='upper right',fontsize=15)

            axs[0,i].tick_params(labelbottom=False, bottom=True)
            #yticks = axs[0,i].get_yticks()
            #yticklabels = ['' if tick == 0 else f'{tick:.2f}' for tick in yticks]
            #axs[0,i].set_yticklabels(yticklabels)
            #axs[0,i].set_yticks(np.linspace(0.10, 0.14, 5))
            axs[0,i].sharex(axs[1,i])

            if features[i] == 'pt':
                axs[1,i].set_xlabel(f"{self.label_names[features[i]]} [GeV]",fontsize=16)
            elif features[i] == 'phi':
                axs[1,i].set_xlabel(f"{self.label_names[features[i]]} [rad]",fontsize=16)
                
            else:
                axs[1,i].set_xlabel(self.label_names[features[i]],fontsize=16)
            axs[1,i].set_ylabel(r'$\frac{\text{Model}}{\text{Truth}}$',fontsize=16)
            axs[1,i].set_ylim(0.8,1.2)  # changed limits from 0.5 to 1.5 to be able to see features better
            axs[1,i].grid(visible=True,which='major',axis='y')

            feature_name = features[i].lower()
            if feature_name == 'pt':
                non_zero_bins = np.where(content_feat > 0)[0]
                if len(non_zero_bins) > 0:
                    last_non_zero_bin = non_zero_bins[-1]
                    max_pt = binning[last_non_zero_bin+1]
                    axs[0,i].set_xlim(0, max_pt)
                axs[0,i].set_ylim(0.001, max(content_acc.max(), content_pred_avg.max()) * 1.4)
            elif feature_name == 'eta':
                axs[0,i].set_xlim(-5,5)
                axs[0,i].set_ylim(0.01, max(content_acc.max(), content_pred_avg.max()) * 1.4)
            elif feature_name == 'phi':
                axs[0,i].set_xlim(-math.pi, math.pi)
                axs[0,i].set_ylim(0.095, 0.14)
                axs[1,i].set_ylim(0.9,1.1)
                
            else:
                print("Invalid feature name")
        return fig

    def make_prediction_plots(self,targets,preds,title):
        figs = {}

        fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(6,4))
        plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.1)
        bins = np.linspace(0,1,self.bins)
        ht = ax.hist(
            preds[targets==0],
            bins = bins,
            color = 'firebrick',
            label = 'P(selected|x non reco)',
            histtype = 'step',
            density = True,
        )
        hp = ax.hist(
            preds[targets==1],
            bins = bins,
            color = 'forestgreen',
            label = 'P(selected|x reco)',
            histtype = 'step',
            density = True,
        )
        ax.text(
            x = 0.7,
            y = 0.7,
            s = f'True eff = {(targets*1.).mean()*100:5.2f}%',
            horizontalalignment='center',
            verticalalignment='center',
            transform = ax.transAxes,
            fontsize = 14,
        )
        ax.text(
            x = 0.7,
            y = 0.6,
            s = f'Pref eff = {preds.mean()*100:5.2f}%',
            horizontalalignment='center',
            verticalalignment='center',
            transform = ax.transAxes,
            fontsize = 14,
        )


        # Esthetic #
        if self.log_scale:
            ax.set_yscale('log')
            ax.set_ylim(1e-1,None)
        else:
            ax.set_ylim(0,None)
        ax.set_title(title)
        ax.legend(loc='upper right',fontsize=14)
        ax.set_xlabel('Acceptance probability',fontsize=14)

        figname = 'score'
        if title is not None:
            figname += f'_{title}'
        figs[figname] = fig


        return figs



class MultiplicityCallback(BaseCallback):
    def __init__(self,N_min,N_max,**kwargs):
        self.N_min = N_min
        self.N_max = N_max
        super().__init__(**kwargs)

    def plot_particle(self,inputs,masks,targets,preds,features,title):
        # Apply mask #
        if masks.sum() == 0:
            return None
        inputs  = inputs[masks]
        preds   = preds[masks]
        targets = targets[masks]

        # make figure #
        N = len(features)
        fig,axs = plt.subplots(ncols=N,nrows=2,figsize=(6*N,8))
        plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.2,hspace=0.2)
        plt.suptitle(title,fontsize=16)

        preds = F.softmax(preds,dim=-1)

        bins_mult = torch.arange(self.N_min,self.N_max+2,1)

        plt.suptitle(title)
        for i in range(N):
            # Make 2D binning #
            feat_min = inputs[:,i].min()
            feat_max = inputs[:,i].max()
            bins_feat = torch.linspace(
                feat_min * (1 - feat_min.sign() * 0.01),
                feat_max * (1 + feat_max.sign() * 0.01),
                # above trick to extend bin range slightly to contain all values (truncation)
                # works whether the value is positive or negative
                self.bins,
            )
            X,Y = np.meshgrid(bins_feat,bins_mult)
            # Make true array #
            true_arr = np.histogram2d(
                inputs[:,i],
                targets.argmax(dim=-1)+self.N_min,
                bins = (bins_feat,bins_mult),
            )[0]
            # Make pred array #
            """pred_arr = np.zeros((self.bins-1,self.N_max-1))"""
            pred_arr = np.zeros((self.bins-1,self.N_max-self.N_min+1))
            idx = np.digitize(inputs[:,i],bins_feat,right=False) - 1
            """line below is new"""
            preds_np = preds.cpu().numpy()
            """for j,ibin in enumerate(idx):
                pred_arr[ibin,:] += preds[j].numpy()"""
            for j,ibin in enumerate(idx):
                if 0 <= ibin < self.bins-1:
                    pred_arr[ibin,:] += preds_np[j]

            # Plot #
            """line below new"""
            valid_pred = pred_arr[pred_arr>0]
            """vmin = pred_arr[pred_arr>0].min()
            vmax = max(pred_arr.max(),true_arr.max())"""
            vmin = valid_pred.min() if len(valid_pred) > 0 else pred_arr.min()
            vmax = max(pred_arr.max(),true_arr.max())
            pc1 = axs[0,i].pcolormesh(
                X,Y,true_arr.T,
                norm = matplotlib.colors.LogNorm(vmin,vmax) if self.log_scale else None,
                vmin = None if self.log_scale else 0.,
                vmax = None if self.log_scale else vmax,
            )
            plt.colorbar(pc1,ax=axs[0,i])
            pc2 = axs[1,i].pcolormesh(
                X,Y,pred_arr.T,
                norm = matplotlib.colors.LogNorm(vmin,vmax) if self.log_scale else None,
                vmin = None if self.log_scale else 0.,
                vmax = None if self.log_scale else vmax,
            )
            plt.colorbar(pc2,ax=axs[1,i])

            # Esthetics #
            if features[i] in self.label_names.keys():
                axs[0,i].set_xlabel(self.label_names[features[i]],fontsize=16)
                axs[1,i].set_xlabel(self.label_names[features[i]],fontsize=16)
            else:
                axs[0,i].set_xlabel(features[i],fontsize=16)
                axs[1,i].set_xlabel(features[i],fontsize=16)
            axs[0,i].set_ylabel('Multiplicity',fontsize=16)
            axs[1,i].set_ylabel('Multiplicity',fontsize=16)

        return fig



    def make_prediction_plots(self,targets,preds,title):
        figs = {}

        fig,axs = plt.subplots(ncols=1,nrows=2,figsize=(8,5),height_ratios=[1.0,0.2], sharex=True, dpi=400)
        plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.1,hspace=0.0)
        plt.suptitle(title)

        preds = F.softmax(preds,dim=-1)

        bins = torch.arange(self.N_min,self.N_max+2,1)

        target_sum = targets.sum(dim=0).numpy()
        preds_sum = preds.sum(dim=0).numpy()

        # Normalize to sum to 1 (density=True)
        target_sum = target_sum / target_sum.sum()
        preds_sum = preds_sum / preds_sum.sum()

        # Compute errors for normalized histogram
        target_var = np.sqrt(targets.sum(dim=0).numpy()) / targets.sum().item()  # Normalize errors
        target_down = target_sum - target_var
        target_up = target_sum + target_var

        # Ratio
        ratio = preds_sum / target_sum
        ratio_down = target_down / target_sum
        ratio_up = target_up / target_sum

        axs[0].stairs(
            values = target_sum,
            edges = bins,
            color = 'royalblue',
            label = 'Truth',
            linewidth = 2,
        )
        axs[0].fill_between(
            x = bins,
            y1 = np.r_[target_down,target_down[-1]],
            y2 = np.r_[target_up,target_up[-1]],
            color = 'royalblue',
            alpha = 0.3,
            step = 'post',
        )
        axs[0].stairs(
            values = preds_sum,
            edges = bins,
            color = 'orange',
            label = 'Classifier',
            linewidth = 2,
        )
        axs[1].step(
            x = bins,
            y = np.r_[ratio[0],ratio],
            color = 'orange',
        )
        axs[1].fill_between(
            x = bins,
            y1 = np.r_[ratio_down,ratio_down[-1]],
            y2 = np.r_[ratio_up,ratio_up[-1]],
            color = 'royalblue',
            alpha = 0.3,
            step = 'post',
        )

        # Esthetic #
        if self.log_scale:
            axs[0].set_yscale('log')
        axs[0].set_xticklabels([])
        axs[0].set_xlim(5, 13)
        axs[0].set_ylim(0.002)
        axs[0].legend(fontsize=20, frameon=False)
        axs[1].set_xlabel('Multiplicity',fontsize=20)
        ratio_maxvar = max(
            [
                abs(1-ratio).max(),
                abs(1-ratio_down).max(),
                abs(1-ratio_up).max(),
            ]
        ) * 1.1

        axs[1].set_ylabel(r'$\frac{\text{Model}}{\text{Truth}}$',fontsize=20)
        axs[1].set_ylim(0.8, 1.2)
        y_min, y_max = axs[1].get_ylim()
        axs[1].axhline(y_min + 0.15, color='black', linestyle='dotted', linewidth=1, alpha=0.5)
        axs[1].axhline(y_max - 0.15, color='black', linestyle='dotted', linewidth=1, alpha=0.5)
        axs[1].set_yticks([y_min + 0.15, 1.0, y_max - 0.15])

        # Set ticks to include all bin edges
        axs[1].tick_params(axis='x', which='both', bottom=True, labelbottom=True)
        axs[1].set_xticks(bins)  # Use all bin edges including the last one
        axs[1].set_xticklabels([int(b) for b in bins])  # All labels as integers

        axs[0].set_ylabel('Density', fontsize=20)

        axs[0].tick_params(axis='y', which='major', labelsize=14)
        axs[1].tick_params(axis='both', which='major', labelsize=14)

        figname = 'multiplicity'
        if title is not None:
            figname += f'_{title}'
        figs[figname] = fig

        return figs

