import math
import torch
from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import Callback

from abc import ABCMeta, abstractmethod

EPS = 1e-12

class BaseCallback(Callback,metaclass=ABCMeta):
    def __init__(self,dataset,frequency=1,raw=False,bins=None,log_scale=False,device=None,batch_size=1024,N_batch=np.inf,label_names={}):
        super().__init__()

        # Attributes #
        self.dataset = dataset
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

        # Loop over batch #
        inputs = []
        targets = []
        preds = []
        for batch_idx, batch in tqdm(enumerate(self.loader),desc='Predict',disable=disable_tqdm,leave=True,total=min(self.N_batch,len(self.loader)),position=0):
            if batch_idx >= self.N_batch:
                break

            # Predict #
            x = batch[0]
            t = batch[1]
            x = x.to(device)
            with torch.no_grad():
                y = model(x)

            # Record #
            inputs.append(x.cpu())
            targets.append(t.cpu())
            preds.append(y.cpu())

        # Concat #
        inputs = torch.cat(inputs,dim=0)
        targets = torch.cat(targets,dim=0)
        preds = torch.cat(preds,dim=0)

        return inputs, targets, preds

    def make_plots(self,model,show=True,disable_tqdm=False):
        inputs, targets, preds = self.predict(model,disable_tqdm)

        figs = {}

        # target vs preds plots #
        if hasattr(self,'make_prediction_plots') and callable(self.make_prediction_plots):
            figs.update(self.make_prediction_plots(targets,preds))

        # make per particle plots #
        figs.update(self.make_particle_plots(inputs,targets,preds))

        if show:
            plt.show()

        return figs

    def make_particle_plots(self,inputs,targets,preds):
        if not hasattr(self,'plot_particle') or not callable(self.plot_particle):
            return {}

        # Make figure plots #
        figs = {}
        n_hards = self.dataset.hard_dataset.number_particles_per_type
        features_per_type = self.dataset.hard_dataset.input_features
        particle_names = self.dataset.hard_dataset.selection

        # Loop over particle types #
        idxs = np.r_[0,np.cumsum(n_hards)]
        for j,(idx_i,idx_f) in enumerate(zip(idxs[:-1],idxs[1:])):
            features = features_per_type[j]
            name = particle_names[j]
            inputs_type = inputs[:,idx_i:idx_f,:]

            # Preprocessing #
            if self.raw:
                preprocessing = self.dataset.hard_dataset.preprocessing
                inputs_type, features = preprocessing.inverse(
                    name = name,
                    x = inputs_type,
                    mask = torch.full((inputs_type.shape[0],inputs_type.shape[1]),fill_value=True),
                    fields = features,
                )

            # Loop over particles within type #
            for i in range(idx_f-idx_i):
                fig = self.plot_particle(
                    inputs = inputs_type[:,i,:],
                    targets = targets,
                    preds = preds,
                    features = features,
                    title = f'{name} #{i}'
                )
                figure_name = f'{name}_{i}'
                figs[figure_name] = fig

        return figs

    @abstractmethod
    def plot_particle(self,inputs,targets,preds,features,title):
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

    def plot_particle(self,inputs,targets,preds,features,title):
        N = len(features)
        fig,axs = plt.subplots(ncols=N,nrows=2,figsize=(6*N,5),height_ratios=[1.0,0.2])
        plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.1)
        mask_sel = (targets > 0).ravel()

        plt.suptitle(title,fontsize=16)
        for i in range(N):
            # Determine the binning from selected events #
            if isinstance(self.min_selected_events_per_bin,(float,int)):
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
                y = np.r_[ratio_nom,ratio_nom[-1]],
                linewidth = 2,
                color = 'orange',
                where = 'post',
            )
            axs[1,i].fill_between(
                x = binning,
                y1 = np.r_[ratio_down,ratio_down[-1]],
                y2 = np.r_[ratio_up,ratio_up[-1]],
                color = 'royalblue',
                alpha = 0.3,
                step = 'mid',
            )
            # Esthetic #
            axs[0,i].set_ylabel('Acceptance',fontsize=16,labelpad=12)
            axs[0,i].set_xticklabels([])
            axs[0,i].set_ylim(0,max(content_acc.max(),content_pred_avg.max())*1.4)
            axs[0,i].legend(loc='upper right',fontsize=15)
            if features[i] in self.label_names.keys():
                axs[1,i].set_xlabel(self.label_names[features[i]],fontsize=16)
            else:
                axs[1,i].set_xlabel(features[i],fontsize=16)
            axs[1,i].set_ylabel(r'$\frac{\text{Model}}{\text{Truth}}$',fontsize=16)
            axs[1,i].set_ylim(0.75,1.25)
            axs[1,i].grid(visible=True,which='major',axis='y')

        return fig



class MultiplicityCallback(BaseCallback):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def plot_particle(self,inputs,targets,preds,features,title):
        N = len(features)
        fig,axs = plt.subplots(ncols=N,nrows=2,figsize=(5*N,8))
        plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.3,hspace=0.3)

        mult_true = torch.where(targets)[1]
        mult_pred = torch.argmax(preds,dim=1)
        N_max = targets.shape[1]
        bins_mult = torch.arange(0,N_max+1,1)

        plt.suptitle(title)
        for i in range(N):
            h1 = axs[0,i].hist2d(
                inputs[:,i],
                mult_true,
                bins = (self.bins,bins_mult),
                norm = matplotlib.colors.LogNorm() if self.log_scale else None,
            )
            plt.colorbar(h1[3],ax=axs[0,i])
            h2 = axs[1,i].hist2d(
                inputs[:,i],
                mult_pred,
                bins = (self.bins,bins_mult),
                norm = matplotlib.colors.LogNorm() if self.log_scale else None,
            )
            plt.colorbar(h2[3],ax=axs[1,i])

            axs[0,i].set_xlabel(features[i])
            axs[1,i].set_xlabel(features[i])
            axs[0,i].set_ylabel('Multiplicity')
            axs[1,i].set_ylabel('Multiplicity')

        return fig



    def make_prediction_plots(self,targets,preds):
        fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(5,4))
        plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.3)

        mult_true = torch.where(targets)[1]
        mult_pred = torch.argmax(preds,dim=1)

        N_max = targets.shape[1]
        bins = torch.arange(0,N_max+1,1)

        ax.hist(
            mult_true,
            bins = bins,
            color = 'b',
            label = 'Truth',
            histtype= 'step',
        )
        ax.hist(
            mult_pred,
            bins = bins,
            color = 'orange',
            label = 'Classifier',
            histtype= 'step',
        )

        # Esthetic #
        if self.log_scale:
            ax.set_yscale('log')
            ax.set_ylim(1e-1,None)
        else:
            ax.set_ylim(0,None)
        ax.set_xlabel('Multiplicity')
        ax.legend()

        return {'multiplicity':fig}


