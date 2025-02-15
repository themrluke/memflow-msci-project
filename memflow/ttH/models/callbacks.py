# callbacks.py

import os
import math
from tqdm import tqdm
import numpy as np
import lightning as L
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from lightning.pytorch.callbacks import Callback
import torch
from torch.utils.data import DataLoader
from .utils import compare_distributions, plot_sampling_distributions



def angle_diff(delta_phi):
    """
    Maps any delta_phi into the interval (-pi, pi].
    delta_phi can be a PyTorch tensor.
    """
    return (delta_phi + math.pi) % (2 * math.pi) - math.pi


class CFMSamplingCallback(Callback):
    def __init__(self, dataset, freq=5, steps=10, show_sampling_distributions=False):
        """
        dataset: e.g. combined_dataset_valid
        freq   : how often (epochs) to do sampling
        steps  : bridging steps for Euler
        """
        super().__init__()
        self.dataset = dataset
        self.freq = freq
        self.steps = steps
        self.show_sampling_distributions = show_sampling_distributions

    def move_batch_to_device(batch, device):
        new_batch = {}
        for top_key, top_val in batch.items():
            new_top_dict = {}
            for key, val in top_val.items():
                if isinstance(val, list):
                    new_top_dict[key] = [item.to(device) for item in val]
                elif isinstance(val, torch.Tensor):
                    new_top_dict[key] = val.to(device)
                else:
                    new_top_dict[key] = val
            new_batch[top_key] = new_top_dict
        return new_batch

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.freq == 0:
            loader = DataLoader(self.dataset, batch_size=32, shuffle=False)
            batch = next(iter(loader))
            batch = self.move_batch_to_device(batch, pl_module.device)

            with torch.no_grad():
                 # Sample multiple times for distribution plots
                num_samples = 100  # Define how many times to sample for the distribution plot
                gen_data_samples = [pl_module.sample(batch, steps=self.steps) for _ in range(num_samples)]

                # Single-time sampling for comparison plots
                gen_data_list = pl_module.sample(batch, steps=self.steps)

             # Compare type=0 (partons->jets)
            real_data = batch["reco"]["data"][0]  # (B, n_recoJets, feats)
            gen_data = gen_data_list[0]          # (B, n_hardPartons, feats)

            # Call `compare_distributions`
            compare_distributions(real_data, gen_data, feat_idx=0, feat_name="pt")
            compare_distributions(real_data, gen_data, feat_idx=1, feat_name="eta")
            compare_distributions(real_data, gen_data, feat_idx=2, feat_name="phi")

            # Call `plot_sampling_distributions` if enabled
            if self.show_sampling_distributions:
                plot_sampling_distributions(
                    real_data=real_data,
                    gen_data_samples=[sample[0] for sample in gen_data_samples],  # Only type 0 (jets)
                    feat_names=["pt", "eta", "phi"],
                    event_idx=0  # Choose which event to plot
                )

            # Potentially log figure to Comet, etc.
            # trainer.logger.experiment.log_figure(...)



class SamplingCallback(Callback):
    def __init__(self,dataset,idx_to_monitor,preprocessing=None,
                 N_sample=1,steps=20,store_trajectories=False,
                 frequency=1,bins=50,log_scale=False,suffix='',
                 label_names={},device=None,pt_range=None):
        super().__init__()

        # Attributes #
        self.dataset = dataset
        self.N_sample = N_sample
        self.steps = steps
        self.store_trajectories = store_trajectories
        self.frequency = frequency
        self.preprocessing = preprocessing
        self.bins = bins
        self.log_scale = log_scale
        self.label_names = label_names
        self.device = device
        self.suffix = suffix
        self.label_names = label_names
        self.pt_range = pt_range

        # Collate partial batch
        self.set_idx(idx_to_monitor)


    def set_idx(self, idx_to_monitor):
        self.idx_to_monitor = idx_to_monitor
        # Checks #
        if not torch.is_tensor(self.idx_to_monitor):
            self.idx_to_monitor = torch.tensor(self.idx_to_monitor)
        if self.idx_to_monitor.dim() < 1:
            self.idx_to_monitor = self.idx_to_monitor.reshape(-1)
        self.N_event = self.idx_to_monitor.shape[0]

        # Build partial batch
        self.batch = {
            'hard': {'data': [], 'mask': []},
            'reco': {'data': [], 'mask': []},
        }
        for idx in self.idx_to_monitor:
            entry = self.dataset[idx]
            self.batch['hard']['data'].append(entry['hard']['data'])
            self.batch['hard']['mask'].append(entry['hard']['mask'])
            self.batch['reco']['data'].append(entry['reco']['data'])
            self.batch['reco']['mask'].append(entry['reco']['mask'])

        from torch.utils.data._utils.collate import default_collate
        self.batch['hard']['data'] = default_collate(self.batch['hard']['data'])
        self.batch['hard']['mask'] = default_collate(self.batch['hard']['mask'])
        self.batch['reco']['data'] = default_collate(self.batch['reco']['data'])
        self.batch['reco']['mask'] = default_collate(self.batch['reco']['mask'])


    def get_bins(self, feature, sample_vals, reco_val):
        """
        Defines the bin bounds for the histogram.
        For phi and eta, fix the range:
            phi -> [-pi, pi]
            eta -> [-5,  5]
        For pT, optionally fix a range around real value if pt_range is not None.
        Otherwise fallback is min->max for sample & real.
        """
        reco_val = float(reco_val)

        if feature == 'phi':
            return np.linspace(-math.pi, math.pi, self.bins)
        elif feature == 'eta':
            return np.linspace(-5.0, 5.0, self.bins)
        elif feature in ['pt', 'm', 'mass']:
            if self.pt_range is not None:
                half_range = self.pt_range / 2.0
                lower = max(0.0, reco_val - half_range)
                upper = max(lower + 1e-3, reco_val + half_range)
                return np.linspace(lower, upper, self.bins)
            else:
                s_np = sample_vals.cpu().numpy()
                qmax = np.quantile(s_np, 0.9999)
                upper = max(qmax, reco_val)
                lower = 0.0
                return np.linspace(lower, upper, self.bins)
        else:
            smin = float(sample_vals.min())
            smax = float(sample_vals.max())
            lower = min(smin, reco_val)
            upper = max(smax, reco_val)
            if abs(upper - lower) < 1e-9:
                upper = lower + 1.0
            return np.linspace(lower, upper, self.bins)

    # Helper function to plot 2D + marginals in a sub-grid
    def plot_particle(self, sample, reco, features, title):
        """
        sample: shape (N, F)   (N = number of samples, F = #features)
        reco:   shape (F,)     (#features)
        features: list of strings (like ["pt", "eta", "phi"])
        title:   str

        This function creates a 1-row, N-columns set of plots.
        Ensures:
          - pt is always on the y-axis.
          - eta vs phi has phi on x-axis.
        """

        assert sample.shape[1] == reco.shape[0]
        N = len(features)
        pairs = []

        # Always put pt on the y-axis
        if 'pt' in features:
            pt_idx = features.index('pt')
            for i in range(N):
                if i != pt_idx:
                    pairs.append((i, pt_idx))  # (x, pt)

        # Ensure eta vs phi has phi on x-axis
        if 'eta' in features and 'phi' in features:
            eta_idx = features.index('eta')
            phi_idx = features.index('phi')
            pairs.append((phi_idx, eta_idx))  # (phi, eta)

        num_pairs = len(pairs)

        fig = plt.figure(figsize=(6 * num_pairs, 5), dpi=300)
        fig.suptitle(title)
        gs_row = fig.add_gridspec(1, num_pairs, wspace=0.3)

        # Helper function to plot 2D + marginals in a sub-grid
        def plot_2d_marginals(fig, parent_spec, xvals, yvals, bins_x, bins_y,
                              label_x, label_y, realx, realy):

            sub_gs = gridspec.GridSpecFromSubplotSpec(
                2, 2,
                subplot_spec=parent_spec,
                width_ratios=[4,1],
                height_ratios=[1,4],
                wspace=0.05,
                hspace=0.05
            )

            ax_main  = fig.add_subplot(sub_gs[1,0])
            ax_top   = fig.add_subplot(sub_gs[0,0], sharex=ax_main)
            ax_right = fig.add_subplot(sub_gs[1,1], sharey=ax_main)

            # Hide tick labels for top & right
            plt.setp(ax_top.get_xticklabels(), visible=False)
            plt.setp(ax_right.get_yticklabels(), visible=False)

            # Make ticks appear inside the plot
            ax_main.tick_params(direction="in")
            ax_top.tick_params(direction="in")
            ax_right.tick_params(direction="in")

            if self.log_scale:
                hh = ax_main.hist2d(xvals, yvals,
                                    bins=[bins_x, bins_y],
                                    norm=matplotlib.colors.LogNorm())
            else:
                hh = ax_main.hist2d(xvals, yvals, bins=[bins_x, bins_y])

            ax_main.set_xlabel(label_x, fontsize=15)
            ax_main.set_ylabel(label_y, fontsize=15)

            # Top marginal histogram
            ax_top.hist(xvals, bins=bins_x, color="mediumpurple", alpha=0.9)
            ax_top.axvline(realx, color='r', linestyle='dashed', linewidth=2)  # Add vertical red line
            ax_top.set_yscale('log' if self.log_scale is True else 'linear')

            # Right marginal histogram
            ax_right.hist(yvals, bins=bins_y, orientation="horizontal", color="mediumseagreen", alpha=0.9)
            ax_right.axhline(realy, color='r', linestyle='dashed', linewidth=2)
            ax_right.set_xscale('log' if self.log_scale is True else 'linear')

            # Overplot real value
            ax_main.scatter(realx, realy, c='r', marker='x', s=75, linewidths=2)

            # Add colorbar BELOW the main plot
            cbar_ax = fig.add_axes([ax_main.get_position().x0,   # Left align with main plot
                                    ax_main.get_position().y0 - 0.15,  # Slightly below main plot
                                    ax_main.get_position().width,  # Same width as main plot
                                    0.02])  # Height of colorbar

            fig.colorbar(hh[3], cax=cbar_ax, orientation="horizontal", label='Frequency')

        # Fill each pair
        for c, (iF, jF) in enumerate(pairs):
            featX = features[iF]
            featY = features[jF]
            # Data
            xvals = sample[:, iF].clone()
            yvals = sample[:, jF].clone()
            realX = float(reco[iF])
            realY = float(reco[jF])

            # Handle circular phi
            if featX == "phi":
                xvals = angle_diff(xvals)
                realX = angle_diff(torch.tensor([realX]))[0].item()
            if featY == "phi":
                yvals = angle_diff(yvals)
                realY = angle_diff(torch.tensor([realY]))[0].item()

            bins_x = self.get_bins(featX, xvals, realX)
            bins_y = self.get_bins(featY, yvals, realY)

            label_x = self.label_names.get(featX, featX)
            label_y = self.label_names.get(featY, featY)

            parent_spec = gs_row[0, c]
            plot_2d_marginals(
                fig, parent_spec,
                xvals.cpu().numpy(), yvals.cpu().numpy(),
                bins_x, bins_y,
                label_x, label_y,
                realX, realY
            )

        return fig


    def make_sampling_plots(self,model,show=False, external_samples=None):
        if self.device is None:
            device = model.device
        else:
            device = self.device
        model = model.to(device)

        # Sample
        hard_data = [d.to(device) for d in self.batch['hard']['data']]
        hard_mask_exist = [m.to(device) for m in self.batch['hard']['mask']]
        reco_data = [d.to(device) for d in self.batch['reco']['data']]
        reco_mask_exist = [m.to(device) for m in self.batch['reco']['mask']]

        if external_samples is None:
            with torch.no_grad():
                samples = model.sample(
                    hard_data, hard_mask_exist,
                    reco_data, reco_mask_exist,
                    self.N_sample,
                    self.steps,
                    self.store_trajectories
                )
        else:
            samples = external_samples # User-provided samples

        samples = [s.cpu() for s in samples] # Move all to CPU
        reco_data = [r.cpu() for r in reco_data]
        reco_mask_exist = [m.cpu() for m in reco_mask_exist]

        # Inverse preprocessing
        if self.preprocessing is not None:
            for i in range(len(reco_data)):
                name = model.reco_particle_type_names[i]
                fields = list(model.reco_input_features_per_type[i])
                flow_fields = [fields[idx] for idx in model.flow_indices[i]]

                # Real data
                reco_data[i],_ = self.preprocessing.inverse(
                    name=name,
                    x=reco_data[i],
                    mask=reco_mask_exist[i],
                    fields=fields
                )
                # Samples
                reshaped = samples[i].reshape(
                    self.N_sample * self.N_event,
                    samples[i].shape[2],
                    samples[i].shape[3]
                )
                mask_reshaped = reco_mask_exist[i].unsqueeze(0).repeat_interleave(
                    self.N_sample, dim=0
                ).reshape(
                    self.N_sample*reco_mask_exist[i].shape[0],
                    reco_mask_exist[i].shape[1]
                )
                out, _ = self.preprocessing.inverse(
                    name=name,
                    x=reshaped,
                    mask=mask_reshaped,
                    fields=flow_fields
                )
                samples[i] = out.reshape(
                    self.N_sample,
                    self.N_event,
                    reco_data[i].shape[1],
                    len(flow_fields),
                )

        # Build figures
        figs = {}
        for event in range(self.N_event):
            for i,(n,flow_feats,flow_indices) in enumerate(zip(
                model.n_reco_particles_per_type,
                model.flow_input_features,
                model.flow_indices
            )):
                for j in range(n):
                    if reco_mask_exist[i][event,j]:
                        fig = self.plot_particle(
                            sample = samples[i][:, event, j, :],
                            reco   = reco_data[i][event,j, flow_indices],
                            features = flow_feats,
                            title = f'{model.reco_particle_type_names[i]} #{j} (event #{event})'
                        )
                        if show:
                            plt.show()
                        fig_name = f'event_{event}_obj_{model.reco_particle_type_names[i]}_{j}'
                        if self.suffix:
                            fig_name += f'_{self.suffix}'
                        figs[fig_name] = fig
        return figs

    def on_validation_epoch_end(self,trainer,pl_module):
        if trainer.sanity_checking:
            return
        if trainer.current_epoch == 0:
            return
        if trainer.current_epoch % self.frequency != 0:
            return

        figs = self.make_sampling_plots(pl_module)
        for figure_name, figure in figs.items():
            trainer.logger.experiment.log_figure(
                figure_name=figure_name,
                figure=figure,
                overwrite=True,
                step=trainer.current_epoch
            )
            plt.close(figure)



class BiasCallback(Callback):
    def __init__(self,dataset,preprocessing=None,N_sample=1,steps=20,store_trajectories=False,frequency=1,raw=False,bins=50,points=20,log_scale=False,device=None,suffix='',label_names={},N_batch=math.inf,batch_size=1024):
        super().__init__()

        # Attributes #
        self.dataset = dataset
        self.loader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
        self.preprocessing = preprocessing
        self.N_sample = N_sample
        self.steps = steps
        self.store_trajectories = store_trajectories
        self.N_batch = N_batch
        self.frequency = frequency
        self.bins = bins
        self.points = points
        self.log_scale = log_scale
        self.label_names = label_names
        self.device = device
        self.suffix = suffix


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


    def plot_particle(self,truth,mask,samples,features,title):
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
        for j, feat in enumerate(features):
            # Getting feature name #
            if features[j] in self.label_names.keys():
                feature_name = self.label_names[features[j]]
            else:
                feature_name = features[j]

            if feat == "phi": # Circular nature of phi demands different error calc
                samples[..., j] = angle_diff(samples[..., j])
                truth[..., j]   = angle_diff(truth[..., j])
                diff[..., j]    = angle_diff(diff[..., j])

            # 1D plot #
            diff_max = abs(diff[...,j]).max()
            diff_bins = np.linspace(-diff_max,diff_max,self.bins)
            axs[0, j].hist(
                diff[..., j].ravel(),
                bins=diff_bins,
                color='red',
                alpha=0.35,
                edgecolor='none'  # Remove the default edges
            )
            axs[0, j].hist(
                diff[..., j].ravel(),
                bins=diff_bins,
                histtype='step',  # Step outline
                color='grey',
                linewidth=1.5
            )

            axs[0,j].set_xlabel(fr'${feature_name} \text{{ (sampled)}} - {feature_name} \text{{ (true)}}$', fontsize=15)
            axs[0,j].set_ylabel('Frequency', fontsize=15)
            if self.log_scale:
                axs[0,j].set_yscale('log')

            # 2D plot #
            if features[j] in ['pt']:
                scale_bins = np.linspace(
                    0,
                    max(
                        torch.quantile(truth[0,:,j].ravel(),q=0.9999,interpolation='higher'),
                        torch.quantile(samples[...,j].ravel(),q=0.9999,interpolation='higher'),
                    ),
                    self.bins,
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
                    self.bins,
                )
            h = axs[1,j].hist2d(
                truth[...,j].ravel(),
                samples[...,j].ravel(),
                bins = (scale_bins,scale_bins),
                norm = matplotlib.colors.LogNorm() if self.log_scale else None,
            )
            axs[1,j].set_xlabel(f'${feature_name}$ (true)', fontsize=15)
            axs[1,j].set_ylabel(f'${feature_name}$ (sampled)', fontsize=15)
            plt.colorbar(h[3],ax=axs[1,j])

            # Bias plot #
            relative = features[j] in ['pt']
            quant_bins = torch.quantile(truth[0,:,j],q=torch.linspace(0.,1.,self.points+1))
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
            axs[2,j].set_xlabel(f'${feature_name}$ (true)', fontsize=15)
            if relative:
                axs[2,j].set_ylabel(fr'$\frac{{{feature_name} \text{{ (sampled)}} - {feature_name}(true)}}{{{feature_name} \text{{ (true)}}}}$', fontsize=15)
            else:
                axs[2,j].set_ylabel(fr'${feature_name} \text{{ (sampled)}} - {feature_name} \text{{ (true)}}$', fontsize=15)

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
            # Getting feature name #
            if features[j] in self.label_names.keys():
                feature_name = self.label_names[features[j]]
            else:
                feature_name = features[j]
            # Calculate qq plot #
            true_quantiles = torch.linspace(0,1,21)
            true_cuts = torch.quantile(truth[:,j],true_quantiles)
            sampled_quantiles = torch.zeros_like(true_quantiles)
            for k,cut in enumerate(true_cuts):
                sampled_quantiles[k] = (samples[:,j].ravel() <= cut).sum() / samples.shape[0]
            # Plot #
            ax.plot(
                true_quantiles,
                sampled_quantiles,
                marker = 'o',
                markersize = 3,
                color = colors[j],
                label = f'${feature_name}$',
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

    def make_bias_plots(self,model,show=False,disable_tqdm=False,external_samples=None):
       # Get samples for whole dataset #
        if self.device is None:
            device = model.device
        else:
            device = self.device
        model = model.to(device)

        N_reco = len(model.n_reco_particles_per_type)
        samples = [[] for _ in range(N_reco)]
        truth   = [[] for _ in range(N_reco)]
        mask    = [[] for _ in range(N_reco)]
        for batch_idx, batch in tqdm(enumerate(self.loader),desc='Predict',disable=disable_tqdm,leave=True,total=min(self.N_batch,len(self.loader)),position=0):
            if batch_idx >= self.N_batch:
                break

            # Get parts #
            hard_data = [data.to(device) for data in batch['hard']['data']]
            hard_mask_exist = [mask.to(device) for mask in batch['hard']['mask']]
            reco_data = [data.to(device) for data in batch['reco']['data']]
            reco_mask_exist = [mask.to(device) for mask in batch['reco']['mask']]

            if external_samples is None:
                # Sample #
                with torch.no_grad():
                    batch_samples = model.sample(
                        hard_data, hard_mask_exist,
                        reco_data, reco_mask_exist,
                        self.N_sample,
                        self.steps,
                        self.store_trajectories,
                    )
            else:
                batch_samples = external_samples # User-provided samples

            # Record #
            for i in range(N_reco):
                samples[i].append(batch_samples[i].cpu())
                truth[i].append(reco_data[i][...,model.flow_indices[i]].cpu())
                mask[i].append(reco_mask_exist[i].cpu())

        # Concat the whole samples list #
        samples = [torch.cat(sample,dim=1) for sample in samples]
        truth   = [torch.cat(t,dim=0) for t in truth]
        mask    = [torch.cat(m,dim=0) for m in mask]

        for i in range(len(truth)):  # Loop over reco particle types
            for j, feat in enumerate(model.flow_input_features[i]):
                if feat == "phi":
                    samples[i][..., j] = angle_diff(samples[i][..., j])
                    truth[i][..., j] = angle_diff(truth[i][..., j])

        # Inverse preprocessing if raw #
        if self.preprocessing is not None:
            for i in range(len(truth)):
                name = model.reco_particle_type_names[i]
                fields = model.reco_input_features_per_type[i]
                flow_fields = [fields[idx] for idx in model.flow_indices[i]]
                truth[i], _ = self.preprocessing.inverse(
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
                samples[i] = self.preprocessing.inverse(
                    name = name,
                    x = samples[i].reshape(self.N_sample*samples[i].shape[1],samples[i].shape[2],samples[i].shape[3]),
                    mask = mask[i].unsqueeze(0).repeat_interleave(self.N_sample,dim=0).reshape(self.N_sample*mask[i].shape[0],mask[i].shape[1]),
                    fields = flow_fields,
                )[0].reshape(self.N_sample,samples[i].shape[1],samples[i].shape[2],samples[i].shape[3])

        # Make figure plots #
        figs = {}
        for i,(truth_type,mask_type,samples_type) in enumerate(zip(truth,mask,samples)):
            for j in range(truth_type.shape[1]):
                fig = self.plot_particle(
                    truth = truth_type[:,j,:],
                    mask = mask_type[:,j],
                    samples = samples_type[:,:,j,:],
                    features = model.flow_input_features[i],
                    title = f'{model.reco_particle_type_names[i]} #{j}',
                )
                if show:
                    plt.show()
                figure_name = f'{model.reco_particle_type_names[i]}_{j}_bias'
                if len(self.suffix) > 0:
                    figure_name += f'_{self.suffix}'
                figs[figure_name] = fig

                fig = self.plot_quantiles(
                    truth = truth_type[:,j,:],
                    mask = mask_type[:,j],
                    samples = samples_type[:,:,j,:],
                    features = model.flow_input_features[i],
                    title = f'{model.reco_particle_type_names[i]} #{j}',
                )
                figure_name = f'{model.reco_particle_type_names[i]}_{j}_quantile'
                if len(self.suffix) > 0:
                    figure_name += f'_{self.suffix}'
                figs[figure_name] = fig

        return figs


    def on_validation_epoch_end(self,trainer,pl_module):
        if trainer.sanity_checking:  # optional skip
            return
        if trainer.current_epoch == 0:
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



class ModelCheckpoint(L.Callback):
    def __init__(self, save_every_n_epochs=10, save_dir="model_checkpoints"):
        super().__init__()
        self.save_every_n_epochs = save_every_n_epochs
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch  # Zero-indexed epoch
        # Save after every save_every_n_epochs (adjusting for zero-indexing)
        if (epoch + 1) % self.save_every_n_epochs == 0:
            ckpt_path = os.path.join(self.save_dir, f"model_epoch_{epoch+1}.ckpt")
            trainer.save_checkpoint(ckpt_path)
            print(f"Checkpoint saved at epoch {epoch+1}: {ckpt_path}")