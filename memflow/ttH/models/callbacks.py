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

        fig = plt.figure(figsize=(6 * num_pairs, 5))
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

            ax_main.set_xlabel(label_x, fontsize=12)
            ax_main.set_ylabel(label_y, fontsize=12)

            # Top marginal histogram
            ax_top.hist(xvals, bins=bins_x, color="mediumpurple", alpha=0.7)
            ax_top.axvline(realx, color='r', linestyle='dashed', linewidth=2)  # Add vertical red line
            ax_top.set_yscale('log')

            # Right marginal histogram
            ax_right.hist(yvals, bins=bins_y, orientation="horizontal", color="mediumseagreen", alpha=0.7)
            ax_right.axhline(realy, color='r', linestyle='dashed', linewidth=2)
            ax_right.set_xscale('log')

            # Overplot real value
            ax_main.scatter(realx, realy, c='r', marker='x', s=100, linewidths=2.5)

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


    def make_sampling_plots(self,model,show=False):
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

        with torch.no_grad():
            samples = model.sample(
                hard_data, hard_mask_exist,
                reco_data, reco_mask_exist,
                self.N_sample,
                self.steps,
                self.store_trajectories
            )

        # Move all to CPU
        samples = [s.cpu() for s in samples]
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