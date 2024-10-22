import math
import torch
from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import Callback

class SamplingCallback(Callback):
    def __init__(self,dataset,idx_to_monitor,N_sample,frequency=1,raw=False,bins=50,log_scale=False,device=None):
        super().__init__()

        # Attributes #
        self.dataset = dataset
        self.N_sample = N_sample
        self.frequency = frequency
        self.raw = raw
        self.bins = bins
        self.log_scale = log_scale
        self.device = device

        # Call batch getting #
        self.set_idx(idx_to_monitor)

    def set_idx(self,idx_to_monitor):
        self.idx_to_monitor = idx_to_monitor
        # Checks #
        if not torch.is_tensor(self.idx_to_monitor):
            self.idx_to_monitor = torch.tensor(self.idx_to_monitor)
        if self.idx_to_monitor.dim() < 1:
            self.idx_to_monitor = self.idx_to_monitor.reshape(-1)
        self.N_event = self.idx_to_monitor.shape[0]

        # Get batch of data #
        self.batch = self.dataset.batch_by_index(self.idx_to_monitor)

    @staticmethod
    def plot_particle(sample,reco,features,title,N_bins,log_scale=False):
        # sample (N,F)
        # reco (F)
        assert sample.shape[1] == reco.shape[0]
        assert reco.shape[0] == len(features)
        N = len(features)
        fig,axs = plt.subplots(N,N,figsize=(4*N,3*N))
        fig.suptitle(title)
        plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,hspace=0.3,wspace=0.3)

        def get_bins(feature,sample,reco):
            if feature in ['pt']:
                bins = np.linspace(
                    0.,
                    max(torch.quantile(sample,0.9999,interpolation='higher'),reco),
                    N_bins,
                )
            else:
                bins = np.linspace(
                    min(sample.min(),reco),
                    max(sample.max(),reco),
                    N_bins,
                )
            return bins


        for i in range(N):
            bins_x = get_bins(features[i],sample[:,i],reco[i])
            for j in range(N):
                bins_y = get_bins(features[j],sample[:,j],reco[j])
                if j > i:
                    axs[i,j].axis('off')
                elif j == i:
                    axs[i,j].hist(sample[:,i],bins=bins_x)
                    axs[i,j].axvline(reco[i],color='r')
                    axs[i,j].set_xlabel(features[i])
                    if log_scale:
                        axs[i,j].set_yscale('log')
                else:
                    h = axs[i,j].hist2d(sample[:,i],sample[:,j],bins=(bins_x,bins_y),norm=matplotlib.colors.LogNorm() if log_scale else None)
                    axs[i,j].scatter(reco[i],reco[j],marker='x',color='r',s=40)
                    axs[i,j].set_xlabel(features[i])
                    axs[i,j].set_ylabel(features[j])
                    plt.colorbar(h[3], ax=axs[i,j])
        return fig

    def make_sampling_plots(self,model,show=False):
        # Select device #
        if self.device is None:
            device = model.device
        else:
            device = self.device
        model = model.to(device)
        # Sample #
        hard_data = [data.to(device) for data in self.batch['hard']['data']]
        hard_mask_exist = [mask.to(device) for mask in self.batch['hard']['mask']]
        reco_data = [data.to(device) for data in self.batch['reco']['data']]
        reco_mask_exist = [mask.to(device) for mask in self.batch['reco']['mask']]

        # Sample #
        with torch.no_grad():
            samples = model.sample(hard_data,hard_mask_exist,reco_data,reco_mask_exist,self.N_sample)

        # Put all on cpu #
        samples = [sample.to('cpu') for sample in samples]
        hard_data = [data.to("cpu") for data in hard_data]
        hard_mask_exist = [mask.to("cpu") for mask in hard_mask_exist]
        reco_data = [data.to("cpu") for data in reco_data]
        reco_mask_exist = [mask.to("cpu") for mask in reco_mask_exist]

        # Inverse preprocessing #
        if self.raw:

            preprocessing = self.dataset.reco_dataset._preprocessing
            for i in range(len(reco_data)):
                name = self.dataset.reco_dataset.selection[i]
                fields = self.dataset.reco_dataset._fields[name]
                flow_fields = [fields[k] for k in model.flow_indices[i]]

                # Inverse for data #
                reco_data[i] = preprocessing.inverse(name,reco_data[i],reco_mask_exist[i],fields)

                # preprocessing expects :
                #   data = [events, particles, features]
                #   mask = [events, particles]
                # samples dims = [samples, events, particles, features]
                # will merge samples*event and unmerge later
                samples[i] = preprocessing.inverse(
                    name = name,
                    x = samples[i].reshape(self.N_sample*self.N_event,samples[i].shape[2],samples[i].shape[3]),
                    mask = reco_mask_exist[i].unsqueeze(0).repeat_interleave(self.N_sample,dim=0).reshape(self.N_sample*reco_mask_exist[i].shape[0],reco_mask_exist[i].shape[1]),
                    fields = flow_fields,
                ).reshape(self.N_sample,self.N_event,samples[i].shape[2],samples[i].shape[3])

        # Loop over events #
        figs = {}
        for event in range(self.N_event):
            # Loop over types #
            for i,(n,flow_features,flow_indices) in enumerate(zip(model.n_reco_particles_per_type,model.flow_input_features,model.flow_indices)):
                # Loop over particles #
                for j in range(n):
                    if reco_mask_exist[i][event,j]:
                        fig = self.plot_particle(
                            sample = samples[i][:,event,j,:],
                            reco = reco_data[i][event,j,flow_indices],
                            features = flow_features,
                            title = f'{self.dataset.reco_dataset.selection[i]} #{j} (event #{event})',
                            N_bins = self.bins,
                            log_scale = self.log_scale,
                        )
                        if show:
                            plt.show()
                        figure_name = f'event_{event}_obj_{self.dataset.reco_dataset.selection[i]}_{j}'
                        figs[figure_name] = fig
        return figs


    def on_validation_epoch_end(self,trainer,pl_module):
        if trainer.sanity_checking:  # optional skip
            return
        if trainer.current_epoch % self.frequency != 0:
           return

        # Get figures #
        figs = self.make_sampling_plots(pl_module)

        # Log them #
        for figure_name,figure in figs.items():
            trainer.logger.experiment.log_figure(
                figure_name = figure_name,
                figure = figure,
                overwrite = True,
                step = trainer.current_epoch,
            )
            plt.close(figure)

class BiasCallback(Callback):
    def __init__(self,dataset,N_sample=1,frequency=1,raw=False,bins=50,points=20,log_scale=False,device=None,N_batch=math.inf,batch_size=1024):
        super().__init__()

        # Attributes #
        self.dataset = dataset
        self.loader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
        self.N_sample = N_sample
        self.N_batch = N_batch
        self.frequency = frequency
        self.raw = raw
        self.bins = bins
        self.points = points
        self.log_scale = log_scale
        self.device = device


    def compute_coverage(self,truth,diff,bins,relative=False):
        centers = []
        quantiles = []
        for x_min,x_max in zip(bins[:-1],bins[1:]):
            mask = torch.logical_and(truth<=x_max,truth>=x_min)
            y = diff[mask]
            if relative:
                y /= truth[mask]
            if y.sum() == 0:
                continue
            quantiles.append(
                torch.quantile(
                    y,
                    q = torch.tensor([0.02275,0.1587,0.5,0.8413,0.97725]),
                ).unsqueeze(0)
            )
            centers.append((x_max+x_min)/2)
        return torch.tensor(centers),torch.cat(quantiles,dim=0)

    def compute_means(self,truth,diff,bins,relative=False):
        means = []
        for x_min,x_max in zip(bins[:-1],bins[1:]):
            mask = torch.logical_and(truth<=x_max,truth>=x_min)
            y = diff[mask]
            if relative:
                y /= truth[mask]
            if y.sum() == 0:
                continue
            means.append(y.mean())
        return torch.tensor(means)

    def compute_modes(self,truth,diff,bins,relative=False):
        modes = []
        for x_min,x_max in zip(bins[:-1],bins[1:]):
            mask = torch.logical_and(truth<=x_max,truth>=x_min)
            y = diff[mask]
            if relative:
                y /= truth[mask]
            if y.sum() == 0:
                continue
            y_binned,y_bins = torch.histogram(y,bins=21)
            # https://www.cuemath.com/data/mode-of-grouped-data/
            y_idxmax = y_binned.argmax()
            f0 = y_binned[max(0,y_idxmax-1)]
            f1 = y_binned[y_idxmax]
            f2 = y_binned[min(len(y_binned)-1,y_idxmax+1)]
            h = y_bins[1]-y_bins[0]
            L = y_bins[y_idxmax]
            modes.append(L + (f1-f0)/(2*f1-f0-f2) * h)
        return torch.tensor(modes)



    def plot_particle(self,truth,mask,samples,features,title,bins,points,log_scale=False):
        # truth shape [N,F]
        # mask shape [N]
        # samples shape [S,N,F]
        # S = samples size
        # N = number of events
        # F = number of features
        assert samples.shape[1] == truth.shape[0]
        assert truth.shape[1] == len(features)
        N = len(features)
        fig,axs = plt.subplots(ncols=N,nrows=3,figsize=(5*N,12))
        fig.suptitle(title)
        plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,hspace=0.3,wspace=0.3)

        if not mask.dtype == torch.bool:
            mask = mask > 0
        truth = truth[mask,:]
        samples = samples[:,mask,:]
        truth = truth.unsqueeze(0).repeat_interleave(repeats=samples.shape[0],dim=0)
        diff = samples-truth
        for j in range(N):
            # 1D plot #
            diff_max = abs(diff[...,j]).max()
            diff_bins = np.linspace(-diff_max,diff_max,bins)
            axs[0,j].hist(
                diff[...,j].ravel(),
                bins = diff_bins,
                histtype = 'step',
                color = 'b',
            )
            axs[0,j].set_xlabel(f'${features[j]}_{{sampled}} - {features[j]}_{{true}}$')
            if log_scale:
                axs[0,j].set_yscale('log')

            # 2D plot #
            if features[j] in ['pt']:
                scale_bins = np.linspace(
                    0,
                    max(
                        torch.quantile(truth[0,:,j].ravel(),q=0.9999,interpolation='higher'),
                        torch.quantile(samples[...,j].ravel(),q=0.9999,interpolation='higher'),
                    ),
                    bins,
                )
            else:
                scale_bins = np.linspace(
                    min(
                        truth[0,:,j].min(),
                        samples[...,j].min(),
                    ),
                    max(
                        truth[0,:,j].max(),
                        samples[...,j].max(),
                    ),
                    bins,
                )
            h = axs[1,j].hist2d(
                truth[...,j].ravel(),
                samples[...,j].ravel(),
                bins = (scale_bins,scale_bins),
                norm = matplotlib.colors.LogNorm() if log_scale else None,
            )
            axs[1,j].set_xlabel(f'${features[j]}_{{true}}$')
            axs[1,j].set_ylabel(f'${features[j]}_{{sampled}}$')
            plt.colorbar(h[3],ax=axs[1,j])

            # Bias plot #
            relative = features[j] in ['pt']
            quant_bins = torch.quantile(truth[0,:,j],q=torch.linspace(0.,1.,points+1))
            centers,coverages = self.compute_coverage(
                truth = truth[...,j].ravel(),
                diff = diff[...,j].ravel(),
                bins = quant_bins,
                relative = relative,
            )
            means = self.compute_means(
                truth = truth[...,j].ravel(),
                diff = diff[...,j].ravel(),
                bins = quant_bins,
                relative = relative,
            )
            #modes = self.compute_modes(
            #    truth = truth[...,j].ravel(),
            #    diff = diff[...,j].ravel(),
            #    bins = quant_bins,
            #    relative = relative,
            #)
            axs[2,j].plot(
                centers,
                means,
                linestyle='dashed',
                linewidth=2,
                marker='o',
                markersize = 2,
                color='k',
                label="mean",
            )
            #axs[2,j].plot(
            #    centers,
            #    modes,
            #    linestyle='dotted',
            #    linewidth=2,
            #    marker='o',
            #    markersize = 2,
            #    color='k',
            #    label="mode",
            #)
            axs[2,j].plot(
                centers,
                coverages[:,2],
                linestyle='-',
                marker='o',
                markersize = 2,
                color='k',
                label="median",
            )
            axs[2,j].fill_between(
                x = centers,
                y1 = coverages[:,1],
                y2 = coverages[:,3],
                color='r',
                alpha = 0.2,
                label="68% quantile",
            )
            axs[2,j].fill_between(
                x = centers,
                y1 = coverages[:,0],
                y2 = coverages[:,4],
                color='b',
                alpha = 0.2,
                label="95% quantile",
            )
            cov_max = abs(coverages).max()
            axs[2,j].set_ylim(-cov_max,cov_max)
            axs[2,j].legend(loc='upper right',facecolor='white',framealpha=1)
            axs[2,j].set_xlabel(f'${features[j]}_{{true}}$')
            if relative:
                axs[2,j].set_ylabel(fr'$\frac{{{features[j]}_{{sampled}} - {features[j]}_{{true}}}}{{{features[j]}_{{true}}}}$')
            else:
                axs[2,j].set_ylabel(fr'${features[j]}_{{sampled}} - {features[j]}_{{true}}$')

        return fig


    def plot_quantiles(self,truth,mask,samples,features,title):
        # Quantile-quantile plot #
        if not mask.dtype == torch.bool:
            mask = mask > 0

        truth = truth[mask,:]
        samples = samples[:,mask,:]
        samples = samples.reshape(samples.shape[0]*samples.shape[1],samples.shape[2])

        fig,ax = plt.subplots(1,1,figsize=(6,5))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(features)))
        for j in range(len(features)):
            true_quantiles = torch.linspace(0,1,21)
            true_cuts = torch.quantile(truth[:,j],true_quantiles)
            sampled_quantiles = torch.zeros_like(true_quantiles)
            for k,cut in enumerate(true_cuts):
                sampled_quantiles[k] = (samples[:,j].ravel() <= cut).sum() / samples.shape[0]
            ax.plot(
                true_quantiles,
                sampled_quantiles,
                marker = 'o',
                markersize = 3,
                color = colors[j],
                label = features[j],
            )
        ax.plot(
            [0,1],
            [0,1],
            linestyle = '--',
            color = 'k',
        )
        ax.set_xlabel('Quantile')
        ax.set_ylabel('Fraction sampled events')
        plt.suptitle(title)
        ax.legend(loc='lower right')

        return fig

    def make_bias_plots(self,model,show=False,disable_tqdm=False):
       # Get samples for whole dataset #
        if self.device is None:
            device = model.device
        else:
            device = self.device
        model = model.to(device)

        N_reco = len(self.dataset.reco_dataset.number_particles_per_type)
        samples = [[] for _ in range(N_reco)]
        truth   = [[] for _ in range(N_reco)]
        mask    = [[] for _ in range(N_reco)]
        for batch_idx, batch in tqdm(enumerate(self.loader,1),desc='Predict',disable=disable_tqdm,leave=True,total=min(self.N_batch,len(self.loader)),position=0):
            # Get parts #
            hard_data = [data.to(device) for data in batch['hard']['data']]
            hard_mask_exist = [mask.to(device) for mask in batch['hard']['mask']]
            reco_data = [data.to(device) for data in batch['reco']['data']]
            reco_mask_exist = [mask.to(device) for mask in batch['reco']['mask']]

            # Sample #
            with torch.no_grad():
                batch_samples = model.sample(hard_data,hard_mask_exist,reco_data,reco_mask_exist,self.N_sample)

            # Record #
            for i in range(N_reco):
                samples[i].append(batch_samples[i])
                truth[i].append(reco_data[i][...,model.flow_indices[i]])
                mask[i].append(reco_mask_exist[i])

            if batch_idx >= self.N_batch:
                break

        # Concat the whole samples list #
        samples = [torch.cat(sample,dim=1).cpu() for sample in samples]
        truth   = [torch.cat(t,dim=0).cpu() for t in truth]
        mask    = [torch.cat(m,dim=0).cpu() for m in mask]

        # Inverse preprocessing if raw #
        if self.raw:
            preprocessing = self.dataset.reco_dataset._preprocessing
            for i in range(len(truth)):
                name = self.dataset.reco_dataset.selection[i]
                fields = self.dataset.reco_dataset._fields[name]
                flow_fields = [fields[idx] for idx in model.flow_indices[i]]
                truth[i] = preprocessing.inverse(
                    name = name,
                    x = truth[i],
                    mask = mask[i],
                    fields = flow_fields,
                )
                # preprocessing expects :
                #   data = [events, particles, features]
                #   mask = [events, particles]
                # samples dims = [samples, events, particles, features]
                # will merge samples*event and unmerge later
                samples[i] = preprocessing.inverse(
                    name = name,
                    x = samples[i].reshape(self.N_sample*samples[i].shape[1],samples[i].shape[2],samples[i].shape[3]),
                    mask = mask[i].unsqueeze(0).repeat_interleave(self.N_sample,dim=0).reshape(self.N_sample*mask[i].shape[0],mask[i].shape[1]),
                    fields = flow_fields,
                ).reshape(self.N_sample,samples[i].shape[1],samples[i].shape[2],samples[i].shape[3])

        # Make figure plots #
        figs = {}
        for i,(truth_type,mask_type,samples_type) in enumerate(zip(truth,mask,samples)):
            for j in range(truth_type.shape[1]):
                fig = self.plot_particle(
                    truth = truth_type[:,j,:],
                    mask = mask_type[:,j],
                    samples = samples_type[:,:,j,:],
                    features = model.flow_input_features[i],
                    title = f'{self.dataset.reco_dataset.selection[i]} #{j}',
                    bins = self.bins,
                    points = self.points,
                    log_scale = self.log_scale,
                )
                if show:
                    plt.show()
                figure_name = f'{self.dataset.reco_dataset.selection[i]}_{j}_bias'
                figs[figure_name] = fig

                fig = self.plot_quantiles(
                    truth = truth_type[:,j,:],
                    mask = mask_type[:,j],
                    samples = samples_type[:,:,j,:],
                    features = model.flow_input_features[i],
                    title = f'{self.dataset.reco_dataset.selection[i]} #{j}',
                )
                figure_name = f'{self.dataset.reco_dataset.selection[i]}_{j}_quantile'

        return figs


    def on_validation_epoch_end(self,trainer,pl_module):
        if trainer.sanity_checking:  # optional skip
            return
        if trainer.current_epoch % self.frequency != 0:
           return

        # Get figures #
        figs = self.make_bias_plots(pl_module,disable_tqdm=True,show=False)

        # Log them #
        for figure_name,figure in figs.items():
            trainer.logger.experiment.log_figure(
                figure_name = figure_name,
                figure = figure,
                overwrite = True,
                step = trainer.current_epoch,
            )
            plt.close(figure)



