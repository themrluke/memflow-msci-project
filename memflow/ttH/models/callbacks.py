# Script Name: callbacks.py
# Author: Luke Johnson
# Description:
#    Callbacks for the models to produce bias plots and event level sampling plots during training and inference.
#    If samples are not provided, (e.g. during training), then inference happens in the sampling callback.
#    This only works for CFM based models though (not transfermer).


import os
import math
from tqdm import tqdm
import numpy as np
import lightning as L
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import scipy.stats as stats
from lightning.pytorch.callbacks import Callback
import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Optional



def angle_diff(
        delta_phi: torch.Tensor
) -> torch.Tensor:
    """
    Maps delta_phi into the interval [-pi, pi].

    Args:
        - delta_phi (torch.Tensor): Angle difference in radians.

    Returns:
        - torch.Tensor: Angle diff.
    """
    return (delta_phi + math.pi) % (2 * math.pi) - math.pi



class SamplingCallback(Callback):
    """
    Callback for the CFM model event-level sampling plots. During training,
    performs inference and generates plots at custom epoch intervals.
    After training, there is the option to pass in samples for faster plotting.
    A single plot per object and event, showing samples in 2D feature space.
    Includes marginal histograms for aesthetics.

    Parameters:
        - dataset (torch.utils.data.Dataset): The validation dataset used for sampling.
        - idx_to_monitor (List[int]): The indices of events to use for sampling.
        - preprocessing (Optional[object]): Preprocessing that was applied to the reco-level data.
        - N_sample (int): Number of samples to generate per particle
        - steps (int): Number of integration steps for CFM models
        - store_trajectories (bool): Whether or not to store intermediate integration steps.
        - frequency (int): Frequency to generate plots (in epochs).
        - bins (int): Number of histogram bins.
        - log_scale (bool): Whether to use log scale for histograms.
        - suffix (str): Suffix for saved figures.
        - label_names (Dict[str,str]): Dictionary mapping feature names to display labels.
        - device (Optional[torch.device]): Device for model inference.
        - pt_range (Optional[float]): Range for the pT values on histograms.
    """
    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            idx_to_monitor: List[int],
            preprocessing: Optional[object] = None,
            N_sample: int = 1,
            steps: int = 20,
            store_trajectories: bool = False,
            frequency: int = 1,
            bins: int = 50,
            log_scale: bool = False,
            suffix: str = '',
            label_names: Dict[str,str] = {},
            device: Optional[torch.device] = None,
            pt_range: Optional[float] = None
    ):
        super().__init__()

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
        self.pt_range = pt_range
        self.set_idx(idx_to_monitor)


    def set_idx(
            self,
            idx_to_monitor: List[int]
    ):
        """
        Processes indexes of events to plot.

        Args:
            - idx_to_monitor (List[int]): The indices of events to use for sampling.
        """
        self.idx_to_monitor = idx_to_monitor

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


    def get_bins(
            self,
            feature: str,
            sample_vals: torch.Tensor,
            reco_val: float
    ) -> np.ndarray:
        """
        Defines the bin bounds for the histogram. Ranges:
            - phi = [-pi, pi].
            - eta = [-5,  5].
            - pT = Custom range OR [min value, max value].

        Args:
            - feature (str): Name of the particle feature ('pt', 'eta', 'phi', 'mass).
            - sample_vals (torch.Tensor): The sampled values for feature.
            - reco_val (float): The real value for that feature.

        Returns:
            - numpy.ndarray: Bin edges for histogram.
        """
        reco_val = float(reco_val)

        if feature == 'phi':
            return np.linspace(-math.pi, math.pi, self.bins)
            #return np.linspace(sample_vals.min(), sample_vals.max(), self.bins) # To zoom in
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


    def plot_particle(
            self,
            sample: torch.Tensor,
            reco: torch.Tensor,
            features: List[str],
            title: str,
            cmap: str = 'BuGn'
    ) -> plt.Figure:
        """
        Creates plots for each pair of features.

        Args:
            - sample (torch.Tensor): Sampled feature values, shape (N_samples, N_features).
            - reco (torch.Tensor): True feature values, shape (N_features,).
            - features (List[str]): List containing the feature names ["pt", "eta", "phi"].
            - title (str): Plot title.
            - cmap (str, optional): Colormap for the hexbin plots.

        Returns:
            - plt.Figure: Generated plots.
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
        gs_row = fig.add_gridspec(1, num_pairs, wspace=0.25) # Adjust the horizontal separation

        # Helper function to plot 2D + marginals in a sub-grid
        def plot_2d_marginals(fig, parent_spec, xvals, yvals, bins_x, bins_y,
                              label_x, label_y, realx, realy):

            sub_gs = gridspec.GridSpecFromSubplotSpec(
                2, 2,
                subplot_spec=parent_spec,
                width_ratios=[4,1],
                height_ratios=[1,4],
                wspace=0,
                hspace=0,
            )

            ax_main  = fig.add_subplot(sub_gs[1,0])
            ax_top   = fig.add_subplot(sub_gs[0,0], sharex=ax_main)
            ax_right = fig.add_subplot(sub_gs[1,1], sharey=ax_main)

            ax_main.xaxis.set_tick_params(labelbottom=True)
            ax_main.yaxis.set_tick_params(labelleft=True)
            ax_main.tick_params(direction="in")

            ax_main.xaxis.set_major_locator(MaxNLocator(nbins=4))  # Increase or decrease nbins as needed
            ax_main.tick_params(axis='both', which='major', labelsize=18)  # Increases font size without affecting frequency

            ax_top.tick_params(axis='both', labelsize=12)   # Adjust the fontsize for top marginal ticks
            ax_right.tick_params(axis='both', labelsize=12) # Adjust the fontsize for right marginal ticks

            ax_main.set_xlabel(label_x, fontsize=24)
            ax_main.set_ylabel(label_y, fontsize=24)

            # Hide tick labels for top & right
            plt.setp(ax_top.get_xticklabels(), visible=False)
            plt.setp(ax_right.get_yticklabels(), visible=False)

            # Compute the extent from the provided bin arrays
            extent = [bins_x[0], bins_x[-1], bins_y[0], bins_y[-1]]

            if self.log_scale:
                hb = ax_main.hexbin(
                    xvals, yvals,
                    gridsize=self.bins,
                    extent=extent,
                    norm=matplotlib.colors.LogNorm(),
                    mincnt=1,  # Only show hexagons with at least one count
                    cmap=cmap
                )
            else:
                hb = ax_main.hexbin(
                    xvals, yvals,
                    gridsize=self.bins,
                    extent=extent,
                    mincnt=1,
                    cmap=cmap
                )

            # Extract a color from the main hist to use for marginal hist
            main_cmap = plt.get_cmap(cmap)
            marginal_color = main_cmap(0.5)

            # Top marginal histogram
            ax_top.hist(xvals, bins=bins_x, color=marginal_color, edgecolor='white', linewidth=0.1)
            ax_top.axvline(realx, color='lime' if cmap=='RdPu' else 'r', linestyle='dashed', linewidth=2)  # Add vertical red line
            ax_top.set_yscale('log' if self.log_scale is True else 'linear')

            # Right marginal histogram
            ax_right.hist(yvals, bins=bins_y, orientation="horizontal", color=marginal_color, edgecolor='white', linewidth=0.1)
            ax_right.axhline(realy, color='lime' if cmap=='RdPu' else 'r', linestyle='dashed', linewidth=2)
            ax_right.set_xscale('log' if self.log_scale is True else 'linear')

            remove_borders = True
            if remove_borders:
                # Remove only the tick labels (and spines) for the marginal plots,
                for ax in [ax_top, ax_right]:
                    for spine in ax.spines.values():
                        spine.set_visible(False)  # Hide the spines (borders)
                    ax_top.spines['bottom'].set_visible(True)
                    ax_right.spines['left'].set_visible(True)
                    # Hide tick labels only, not the ticks
                    plt.setp(ax.get_xticklabels(), visible=False)
                    plt.setp(ax.get_yticklabels(), visible=False)
                    ax.tick_params(axis='both', which='both', length=0, width=0, colors='none')   
                    # Clear any axis labels on the marginal plots
                    ax.set_xlabel('')
                    ax.set_ylabel('')
            else:
                # Make ticks appear inside the plot
                ax_top.tick_params(direction="in")
                ax_right.tick_params(direction="in")

            # Overplot real value
            ax_main.scatter(realx, realy, c='lime' if cmap=='RdPu' else 'r', marker='x', s=75, linewidths=2)

            # Add a colorbar BELOW the main plot
            cbar_ax = fig.add_axes([
                ax_main.get_position().x0,                    # Left align with main plot
                ax_main.get_position().y0 - 0.2,               # Place slightly below main plot
                ax_main.get_position().width,                 # Same width as main plot
                0.02                                          # Height of the colorbar
            ])
            cbar = fig.colorbar(hb, cax=cbar_ax, orientation="horizontal")
            cbar.ax.tick_params(labelsize=18)  # Increase font size of colorbar tick labels
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


    def make_sampling_plots(
            self,
            model: torch.nn.Module,
            external_samples: Optional[List[torch.Tensor]] = None,
            cmap: str = 'BuGn',
            save_dir: str = 'sampling_plots'
    )-> Dict[str, plt.Figure]:
        """
        Generates the event-level sampling plots.

        Args:
        - model (torch.nn.Module): The trained model used for inference (only works with CFM models).
        - external_samples (Optional[List[torch.Tensor]]): Pre-made samples to save time.
        - cmap (str, optional): Colormap for plots.
        - save_dir (str, optional): Directory to save the plots to.

        Returns:
            - Dict[str, plt.Figure]: Dictionary mapping plot names to matplotlib figures.
        """
        if self.device is None:
            device = model.device
        else:
            device = self.device
        model = model.to(device)

        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Sample
        hard_data = [d.to(device) for d in self.batch['hard']['data']]
        hard_mask_exist = [m.to(device) for m in self.batch['hard']['mask']]
        reco_data = [d.to(device) for d in self.batch['reco']['data']]
        reco_mask_exist = [m.to(device) for m in self.batch['reco']['mask']]

        # If pre-made samples are not passed in (e.g. during training), they can be made here
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
            samples = external_samples # pre-made samples

        samples = [s.cpu() for s in samples] # Move all to CPU
        reco_data = [r.cpu() for r in reco_data]
        reco_mask_exist = [m.cpu() for m in reco_mask_exist]

        # Inverse the preprocessing
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
                            title = f'{model.reco_particle_type_names[i]} #{j} (event #{event})',
                            cmap=cmap
                        )
                        fig_name = f'event_{event}_obj_{model.reco_particle_type_names[i]}_{j}'
                        if self.suffix:
                            fig_name += f'_{self.suffix}'

                        fig_path = os.path.join(save_dir, fig_name + ".png")
                        # Save figure
                        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                        plt.close(fig)  # Close to free memory
                        figs[fig_name] = fig
                        print(f"Saved: {fig_path}")

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
    """
    Callback for evaluating bias in the generated samples compared to real data.
    It computes coverage, mean, and mode statistics, and generates bias plots
    and quantile-quantile plots.

    Parameters:
        - dataset (torch.utils.data.Dataset): Validation dataset.
        - preprocessing (Optional[object]): Preprocessing applied to the reco-level data.
        - N_sample (int): Number of samples to draw per event.
        - steps (int): Number of integration steps for CFM models.
        - store_trajectories (bool): Whether to store intermediate integration steps.
        - frequency (int): Frequency of logging bias plots (in epochs).
        - bins (int): Number of bins for histograms.
        - points (int): Number of points for quantile comparisons.
        - log_scale (bool): Whether the plots should use log scale.
        - device (Optional[torch.device]): Device for model inference.
        - suffix (str): Suffix for saved figures.
        - label_names (Dict[str, str]): Mapping of feature names to display labels.
        - N_batch (int): Maximum number of batches to process.
        - batch_size (int): Number of events per batch in the DataLoader.
    """
    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            preprocessing: Optional[object] = None,
            N_sample: int = 1,
            steps: int = 20,
            store_trajectories: bool = False,
            frequency: int = 1,
            bins: int = 50,
            points: int = 20,
            log_scale: bool = False,
            device: Optional[torch.device] = None,
            suffix: str = '',
            label_names: Dict[str, str] = {},
            N_batch: int = math.inf,
            batch_size: int = 1024
):
        super().__init__()

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


    def compute_coverage(
            self,
            truth: torch.Tensor,
            diff: torch.Tensor,
            bins: torch.Tensor,
            relative: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes quantile coverage of the sampled data.

        Args:
            - truth (torch.Tensor): Truth values.
            - diff (torch.Tensor): Difference between sampled and true reco-level data.
            - bins (torch.Tensor): Bin edges.
            - relative (bool, optional): Whether to compute relative errors.

        Returns:
            - Tuple[torch.Tensor, torch.Tensor]: Centers of bins and quantile values.
        """
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


    def compute_means(
            self,
            truth: torch.Tensor,
            diff: torch.Tensor,
            bins: torch.Tensor,
            relative: bool = False
    ) -> torch.Tensor:
        """
        Computes the mean error within bins.

        Args:
            - truth (torch.Tensor): Truth values.
            - diff (torch.Tensor): Difference between sampled and true reco-level data.
            - bins (torch.Tensor): Bin edges.
            - relative (bool, optional): Whether or not to compute relative errors.

        Returns:
            - torch.Tensor: Mean error for each bin.
        """
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


    def compute_modes(
            self,
            truth: torch.Tensor,
            diff: torch.Tensor,
            bins: torch.Tensor,
            relative: bool = False
    ) -> torch.Tensor:
        """
        Computes the modes within bins.

        Args:
            - truth (torch.Tensor): Truth values.
            - diff (torch.Tensor): Difference between sampled and true reco-level data.
            - bins (torch.Tensor): Bin edges.
            - relative (bool, optional): Whether or not to compute relative errors.

        Returns:
            - torch.Tensor: Mode for each bin.
        """
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


    def plot_particle(
            self,
            truth: torch.Tensor,
            mask: torch.Tensor,
            samples: torch.Tensor,
            features: List[str],
            title: str
    ) -> plt.Figure:
        """
        Creates:
            - 1D histograms of the differences between the true and samled values for each feature across all events.
            - 2D scatter plot comparing the true vs sampled values.
            - Bias plot showing quantiles.

        Args:
            - truth (torch.Tensor): True vals.
            - mask (torch.Tensor): Mask tensor indicating valid events.
            - samples (torch.Tensor): Sampled values.
            - features (List[str]): List of feature names in true data.
            - title (str): Title of the plot.

        Returns:
            - plt.Figure: The generated Figure. 
        """
        feature_info = {
            "pt": {
                "latex": r"p_T",
                "units": "[GeV]",
                "limits": {
                    "jets": [30, 750],
                    "MET": [200, 1000]
                }
            },
            "eta": {
                "latex": r"\eta",
                "units": "",
                "limits": [-5, 5]
            },
            "phi": {
                "latex": r"\phi",
                "units": "[rad]",
                "limits": [-math.pi, math.pi]
            },
            "mass": {
                "latex": r"\text{Mass}",
                "units": "[GeV]",
                "limits": [0, 150]
            }
        }

        # truth shape [N,F]
        # mask shape [N]
        # samples shape [S,N,F]
        # S = samples size
        # N = number of events
        # F = number of features
        assert samples.shape[1] == truth.shape[0]
        assert truth.shape[1] == len(features)

        N = len(features)
        fig,axs = plt.subplots(ncols=N,nrows=3,figsize=(5*N,12), dpi=300)
        fig.suptitle(title)
        plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,hspace=0.3,wspace=0.35)

        if not mask.dtype == torch.bool:
            mask = mask > 0

        truth = truth[mask,:]
        samples = samples[:,mask,:]
        truth = truth.unsqueeze(0).repeat_interleave(repeats=samples.shape[0],dim=0)
        diff = samples-truth

        for j, feat in enumerate(features):
            # Getting feature name
            if features[j] in self.label_names.keys():
                feature_name = self.label_names[features[j]]
            else:
                feature_name = features[j]

            if feat == "phi": # Circular nature of phi demands different error calc
                samples[..., j] = angle_diff(samples[..., j])
                truth[..., j]   = angle_diff(truth[..., j])
                diff[..., j]    = angle_diff(diff[..., j])

            # 1D plot
            # Compute mean and std
            mu, sigma = diff[..., j].mean().item(), diff[..., j].std().item()
            # Normalize the x-axis: Keep mean but scale by σ
            normalized_diff = diff[..., j] / sigma  # Do NOT subtract mu!
            # Define bins for normalized histogram
            normalized_bins = np.linspace(-5, 5, 41)
            # diff_max = abs(diff[...,j]).max()
            # diff_bins = np.linspace(-diff_max,diff_max,self.bins)
            axs[0, j].hist(
                normalized_diff.ravel(),
                bins=normalized_bins,
                density=True,
                color='#7fb3d5',
                edgecolor='none',  # Remove the default edges
            )
            axs[0, j].hist(
                normalized_diff.ravel(),
                bins=normalized_bins,
                density=True,
                histtype='step',  # Step outline
                color='#1f618d',
                linewidth=1.5
            )

            # Generate x values for standard normal distribution
            normal_dist_x = np.linspace(-5, 5, 100)  # Standard range for N(0,1)
            # Compute the standard normal PDF
            standard_normal = stats.norm.pdf(normal_dist_x, 0, 1)  # Mean = 0, Std = 1
            # Overlay the normal distribution in red
            axs[0, j].plot(normal_dist_x, standard_normal, 'r-', linewidth=2, label="Standard Normal")

            # Use the mapped name if available, otherwise keep as-is
            feature_name_axis = feature_info[feat]["latex"]  # Get LaTeX label from dictionary
            unit_str = f"\\text{{ {feature_info[feat]['units']} }}" if feature_info[feat]["units"] else ""
            axs[0,j].set_xlabel(fr'$({feature_name_axis}^{{\text{{model}}}} - {feature_name_axis}^{{\text{{true}}}}) \, / \, \sigma $', fontsize=15)
            axs[0,j].set_ylabel('Density', fontsize=15)
            axs[0,j].set_yscale('log' if self.log_scale is True else 'linear')
            axs[0,j].set_xlim(-5, 5)

            # 2D plot
            # Set xy limits based on feature-specific ranges
            if feat in feature_info:
                if feat == "pt":  # Special case for pt (need to distinguish between jets and MET)
                    if "met" in title:
                        xlim = feature_info["pt"]["limits"]["MET"]
                        ylim = feature_info["pt"]["limits"]["MET"]
                    elif "jets" in title:
                        xlim = feature_info["pt"]["limits"]["jets"]
                        ylim = feature_info["pt"]["limits"]["jets"]
                    else:
                        raise ValueError(f"Particle type not found in the title string.")
                else:
                    xlim = feature_info[feat]["limits"]
                    ylim= feature_info[feat]["limits"]
            else:
                raise ValueError(f"Feature '{feat}' is not in feature_info dictionary.")

            # Define the number of bins to remain fixed
            num_visible_bins = 51 #self.bins  # Adjust as needed for resolution
            bin_width_x = (xlim[1] - xlim[0]) / num_visible_bins
            bin_width_y = (ylim[1] - ylim[0]) / num_visible_bins
            # Generate bins based on fixed bin width
            scale_bins_x = np.arange(xlim[0], xlim[1] + bin_width_x, bin_width_x)
            scale_bins_y = np.arange(ylim[0], ylim[1] + bin_width_y, bin_width_y)

            h = axs[1,j].hist2d(
                truth[...,j].ravel(),
                samples[...,j].ravel(),
                bins = (scale_bins_x,scale_bins_y),
                norm = matplotlib.colors.LogNorm(),
                cmap='viridis'
            )

            axs[1,j].set_xlabel(fr'${feature_name_axis}^{{\text{{true}}}}{unit_str}$', fontsize=22)
            axs[1,j].set_ylabel(fr'${feature_name_axis}^{{\text{{model}}}}{unit_str}$', fontsize=22)
            cbar = plt.colorbar(h[3],ax=axs[1,j])
            cbar.ax.tick_params(labelsize=18)  # Increase font size of colorbar tick labels

            # Bias plot
            relative = features[j] in ['pt', 'mass']
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
                label="Mean",
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
                label="Median",
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
            axs[2, j].set_xlim(min(centers), max(centers))

            axs[2,j].legend(loc='upper right',facecolor='white',framealpha=0.9,fontsize=14)

            axs[2,j].set_xlabel(fr'${feature_name_axis}^{{\text{{true}}}}{unit_str}$', fontsize=22)

            if relative:
                axs[2,j].set_ylabel(fr'$\frac{{{feature_name_axis}^{{\text{{model}}}} - {feature_name_axis}^{{\text{{true}}}}}}{{{feature_name_axis}^{{\text{{true}}}}}}$', fontsize=24)
                if 'jets' in title:
                    axs[2,j].set_ylim(-2, 2)
            else:
                axs[2,j].set_ylabel(fr'${feature_name_axis}^{{\text{{model}}}} - {feature_name_axis}^{{\text{{true}}}}{unit_str}$', fontsize=22)

            axs[0, j].tick_params(axis='both', which='both', labelsize=16)  # Top row
            axs[1, j].tick_params(axis='both', which='both', labelsize=16)  # Middle row
            axs[2, j].tick_params(axis='both', which='both', labelsize=16)  # Bottom row

        return fig


    def plot_quantiles(
            self,
            truth: torch.Tensor,
            mask: torch.Tensor,
            samples: torch.Tensor,
            features: List[str],
            title: str
    ) -> plt.Figure:
        """
        Generates a quantile-quantile (QQ) plot comparing the distribution of sampled and true data.

        Args:
            - truth (torch.Tensor): True vals.
            - mask (torch.Tensor): Mask tensor of shape indicating valid events.
            - samples (torch.Tensor): Sampled values.
            - features (List[str]): List of feature names in true data.
            - title (str): Title of the plot.


        Returns:
            - plt.Figure: The QQ plot figure.
        """
        if not mask.dtype == torch.bool:
            mask = mask > 0

        truth = truth[mask,:]
        samples = samples[:,mask,:]
        samples = samples.reshape(samples.shape[0]*samples.shape[1],samples.shape[2])

        fig,ax = plt.subplots(1,1,figsize=(6,5))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(features)))

        for j in range(len(features)):
            # Get feature name
            if features[j] in self.label_names.keys():
                feature_name = self.label_names[features[j]]
            else:
                feature_name = features[j]
            # Calculate qq
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


    def make_bias_plots(
            self,
            model,
            show: bool = False,
            disable_tqdm: bool = False,
            external_samples: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, plt.Figure]:
        """
        Generates all the bias plots including: QQ plot, bias plot, true vs gen scatter, differences histogram.

        Args:
            - model: The generative model that was used for sampling.
            - show (bool, optional): Whether to display plots interactively.
            - disable_tqdm (bool, optional): To disable progress bar.
            - external_samples (Optional[List[torch.Tensor]], optional): Pre-made samples to use instead of generating new ones.

        Returns:
            - Dict[str, plt.Figure]: A dictionary of figures.
        """
        # Get samples for whole dataset
        if self.device is None:
            device = model.device
        else:
            device = self.device
        model = model.to(device)

        N_reco = len(model.n_reco_particles_per_type)
        samples = [[] for _ in range(N_reco)]
        truth   = [[] for _ in range(N_reco)]
        mask    = [[] for _ in range(N_reco)]

        if external_samples is not None:
            samples = external_samples  # Already batched samples
            # Accumulate truth and mask over all batches, NOT just the first batch
            for batch_idx, batch in enumerate (self.loader):
                if batch_idx >= self.N_batch:
                    break
                hard_data = [data.to(device) for data in batch['hard']['data']]
                reco_data = [data.to(device) for data in batch['reco']['data']]
                reco_mask_exist = [mask.to(device) for mask in batch['reco']['mask']]
                for i in range(N_reco):
                    truth[i].append(reco_data[i][..., model.flow_indices[i]].cpu())
                    mask[i].append(reco_mask_exist[i].cpu())
            truth = [torch.cat(t, dim=0) for t in truth]
            mask = [torch.cat(m, dim=0) for m in mask]
        else:
            for batch_idx, batch in tqdm(enumerate(self.loader), ...):
                if batch_idx >= self.N_batch:
                    break
                hard_data = [data.to(device) for data in batch['hard']['data']]
                hard_mask_exist = [mask.to(device) for mask in batch['hard']['mask']]
                reco_data = [data.to(device) for data in batch['reco']['data']]
                reco_mask_exist = [mask.to(device) for mask in batch['reco']['mask']]
                with torch.no_grad():
                    batch_samples = model.sample(
                        hard_data, hard_mask_exist,
                        reco_data, reco_mask_exist,
                        self.N_sample,
                        self.steps,
                        self.store_trajectories,
                    )
                for i in range(N_reco):
                    samples[i].append(batch_samples[i].cpu())
                    truth[i].append(reco_data[i][..., model.flow_indices[i]].cpu())
                    mask[i].append(reco_mask_exist[i].cpu())
            samples = [torch.cat(sample, dim=1) for sample in samples]
            truth   = [torch.cat(t, dim=0) for t in truth]
            mask    = [torch.cat(m, dim=0) for m in mask]


        for i in range(len(truth)):  # Loop over reco particle types
            for j, feat in enumerate(model.flow_input_features[i]):
                if feat == "phi":
                    samples[i][..., j] = angle_diff(samples[i][..., j])
                    truth[i][..., j] = angle_diff(truth[i][..., j])

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
                    x = samples[i].reshape(self.N_sample*samples[i].shape[1],samples[i].shape[2],samples[i].shape[3]).cpu(),
                    mask = mask[i].unsqueeze(0).repeat_interleave(self.N_sample,dim=0).reshape(self.N_sample*mask[i].shape[0],mask[i].shape[1]).cpu(),
                    fields = flow_fields,
                )[0].reshape(self.N_sample,samples[i].shape[1],samples[i].shape[2],samples[i].shape[3])

        # Make the different plots
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
        if trainer.sanity_checking:
            return
        if trainer.current_epoch == 0:
            return
        if trainer.current_epoch % self.frequency != 0:
           return

        # Get figures
        figs = self.make_bias_plots(pl_module,disable_tqdm=True,show=False)

        # Log them
        for figure_name,figure in figs.items():
            trainer.logger.experiment.log_figure(
                figure_name = figure_name,
                figure = figure,
                overwrite = True,
                step = trainer.current_epoch,
            )
            plt.close(figure)


class MultiModelHistogramPlotter:
    """
    Callback for evaluating bias in the generated samples compared to real data.
    It computes coverage, mean, and mode statistics, and generates bias plots
    and quantile-quantile plots.

    Parameters:
        - dataset (torch.utils.data.Dataset): Validation dataset.
        - preprocessing (Optional[object]): Preprocessing applied to the reco-level data.
        - N_sample (int): Number of samples to draw per event.
        - steps (int): Number of integration steps for CFM models.
        - store_trajectories (bool): Whether to store intermediate integration steps.
        - bins (int): Number of bins for histograms.
        - log_scale (bool): Whether the plots should use log scale.
        - device (Optional[torch.device]): Device for model inference.
        - suffix (str): Suffix for saved figures.
        - label_names (Dict[str, str]): Mapping of feature names to display labels.
        - N_batch (int): Maximum number of batches to process.
        - batch_size (int): Number of events per batch in the DataLoader.

    Note:
        The external samples provided to `make_error_plots` should be a list (of length up to 3)
        where each element is a list of torch.Tensors corresponding to each reco particle type.
    """
    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            preprocessing: Optional[object] = None,
            N_sample: int = 1,
            steps: int = 20,
            store_trajectories: bool = False,
            bins: int = 41,
            log_scale: bool = False,
            device: Optional[torch.device] = None,
            suffix: str = '',
            label_names: Dict[str, str] = {},
            N_batch: int = math.inf,
            batch_size: int = 1024
    ):
        super().__init__()

        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.preprocessing = preprocessing
        self.N_sample = N_sample
        self.steps = steps
        self.store_trajectories = store_trajectories
        self.N_batch = N_batch
        self.bins = bins
        self.log_scale = log_scale
        self.label_names = label_names
        self.device = device
        self.suffix = suffix

    # def plot_particle(
    #         self,
    #         truth: torch.Tensor,
    #         mask: torch.Tensor,
    #         samples_list: List[torch.Tensor],
    #         features: List[str],
    #         title: str,
    #         normalize_global: bool = False
    # ) -> plt.Figure:
    #     """
    #     Creates 1D histograms of the differences between the true and model-sampled values
    #     for each feature across all events. Overlays histograms from multiple models.

    #     Args:
    #         - truth (torch.Tensor): True values with shape [N, F].
    #         - mask (torch.Tensor): Boolean mask tensor of shape [N].
    #         - samples_list (List[torch.Tensor]): List of sample tensors (one per model) each with shape [S, N, F].
    #         - features (List[str]): List of feature names.
    #         - title (str): Title of the plot.
    #         - normalize_global (bool): Normalize by individual std (False) or by Parallel Transfusion std (True).

    #     Returns:
    #         - plt.Figure: The generated Figure.
    #     """
    #     # Define feature-specific display information
    #     feature_info = {
    #         "pt": {
    #             "latex": r"p_T",
    #             "units": "[GeV]",
    #             "limits": {
    #                 "jets": [30, 750],
    #                 "MET": [200, 1000]
    #             }
    #         },
    #         "eta": {
    #             "latex": r"\eta",
    #             "units": "",
    #             "limits": [-5, 5]
    #         },
    #         "phi": {
    #             "latex": r"\phi",
    #             "units": "[rad]",
    #             "limits": [-math.pi, math.pi]
    #         },
    #         "mass": {
    #             "latex": r"\text{Mass}",
    #             "units": "[GeV]",
    #             "limits": [0, 150]
    #         }
    #     }

    #     N = len(features)
    #     fig, axs = plt.subplots(ncols=N, nrows=1, figsize=(6 * N, 5), dpi=300)
    #     fig.suptitle(title)
    #     plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, hspace=0.3, wspace=0.2)

    #     if not mask.dtype == torch.bool:
    #         mask = mask > 0

    #     # Apply the mask to the truth tensor once
    #     masked_truth = truth[mask, :]  # shape: [N_mask, F]

    #     # Define colors and labels for up to three models.
    #     line_colors = ['#d62728', '#2ca02c', '#9467bd']
    #     model_labels = ['Transfermer', 'Parallel Transfusion', 'Transfer CFM']

    #     # Compute global standard deviations if normalize_global is True
    #     if normalize_global:
    #         reference_samples = samples_list[1]  # "Parallel Transfusion" samples
    #         masked_ref_samples = reference_samples[:, mask, :]  # shape: [S, N_mask, F]
    #         repeated_truth = masked_truth.unsqueeze(0).repeat(masked_ref_samples.shape[0], 1, 1)
    #         diff_ref = masked_ref_samples - repeated_truth  # shape: [S, N_mask, F]
    #         global_stds = diff_ref.std(dim=(0, 1), keepdim=False)  # shape: [F]
    #     else:
    #         global_stds = None  # Will be computed per model instead

    #     # Loop over each feature
    #     for j, feat in enumerate(features):
    #         # Define bins for normalized histogram
    #         normalized_bins = np.linspace(-5, 5, self.bins)
    #         # For each model, compute and plot the histogram
    #         for m, samples in enumerate(samples_list):
    #             # samples assumed shape: [S, N, F]; apply mask along the event (N) dimension
    #             masked_samples = samples[:, mask, :]  # shape: [S, N_mask, F]
    #             # Repeat truth along sample dimension to match shape
    #             repeated_truth = masked_truth.unsqueeze(0).repeat(masked_samples.shape[0], 1, 1)
    #             diff = masked_samples - repeated_truth  # shape: [S, N_mask, F]

    #             # If the feature is 'phi', account for its circular nature
    #             if feat == "phi":
    #                 # Use the angle_diff function on the appropriate slices
    #                 diff_feature = angle_diff(diff[..., j])
    #             else:
    #                 diff_feature = diff[..., j]

    #             # Choose normalization strategy
    #             if normalize_global:
    #                 sigma = global_stds[j].item()  # Use the fixed standard deviation
    #             else:
    #                 sigma = diff_feature.std().item()  # Compute per model
    #             if sigma == 0:
    #                 sigma = 1.0  # Avoid division by zero

    #             normalized_diff = diff_feature / sigma

    #             # Plot the histogram for this model on the same axes
    #             axs[j].hist(
    #                 normalized_diff.ravel(),
    #                 bins=normalized_bins,
    #                 density=True,
    #                 histtype='step',
    #                 linewidth=2,
    #                 color=line_colors[m],
    #                 label=model_labels[m]
    #             )

    #         # Overlay the standard normal distribution for reference
    #         normal_dist_x = np.linspace(-4.5, 5.5, 1000)
    #         standard_normal = stats.norm.pdf(normal_dist_x, 0, 1)
    #         axs[j].plot(normal_dist_x, standard_normal, 'k', linewidth=2, label="Normal")

    #         # Define custom legend handles
    #         legend_handles = [
    #             Line2D([0], [0], color='k', linewidth=2, linestyle='-', label='Normal'),
    #             Line2D([0], [0], color=line_colors[0], linewidth=2, linestyle='-', label=model_labels[0]),
    #             Line2D([0], [0], color=line_colors[1], linewidth=2, linestyle='-', label=model_labels[1]),
    #             Line2D([0], [0], color=line_colors[2], linewidth=2, linestyle='-', label=model_labels[2]),
    #         ]
    #         if normalize_global:
    #             legend_handles.append(Line2D([0], [0], color='none', label=fr'$\sigma \, = \, {sigma:.2f}$'))

    #         # Use mapped LaTeX name if available
    #         feature_name_axis = feature_info[feat]["latex"]
    #         axs[j].set_xlabel(
    #             fr'$({feature_name_axis}^{{\text{{model}}}} - {feature_name_axis}^{{\text{{true}}}}) \, / \, \sigma $',
    #             fontsize=18
    #         )
    #         axs[j].set_ylabel('Density', fontsize=18)
    #         axs[j].tick_params(axis='both', which='major', labelsize=14)
    #         axs[j].set_yscale('log' if self.log_scale else 'linear')
    #         axs[j].set_xlim(-4.5, 5.5)
    #         axs[j].legend(frameon=False, handles=legend_handles, handlelength=1.5, loc='upper right', fontsize=12)

    #     return fig


    def plot_particle( # Same as above but outputs figures seperately
            self,
            truth: torch.Tensor,
            mask: torch.Tensor,
            samples_list: List[torch.Tensor],
            features: List[str],
            title: str,
            normalize_global: bool = False
    ) -> dict:
        """
        Creates individual 1D histograms of the differences between the true and model-sampled values
        for each feature across all events. Overlays histograms from multiple models.

        Args:
            - truth (torch.Tensor): True values with shape [N, F].
            - mask (torch.Tensor): Boolean mask tensor of shape [N].
            - samples_list (List[torch.Tensor]): List of sample tensors (one per model) each with shape [S, N, F].
            - features (List[str]): List of feature names.
            - title (str): Title prefix for each plot.
            - normalize_global (bool): Normalize by individual std (False) or by Parallel Transfusion std (True).

        Returns:
            - dict: A dictionary of Matplotlib Figures, keyed by feature names.
        """
        feature_info = {
            "pt": {
                "latex": r"p_T",
                "units": "[GeV]",
                "limits": {
                    "jets": [30, 750],
                    "MET": [200, 1000]
                }
            },
            "eta": {
                "latex": r"\eta",
                "units": "",
                "limits": [-5, 5]
            },
            "phi": {
                "latex": r"\phi",
                "units": "[rad]",
                "limits": [-math.pi, math.pi]
            },
            "mass": {
                "latex": r"\text{Mass}",
                "units": "[GeV]",
                "limits": [0, 150]
            }
        }

        if not mask.dtype == torch.bool:
            mask = mask > 0

        # Apply the mask to the truth tensor once
        masked_truth = truth[mask, :]  # shape: [N_mask, F]

        # Define colors and labels for up to three models.
        line_colors = ['#d62728', '#2ca02c', '#9467bd']
        model_labels = ['Transfermer', 'Parallel Transfusion', 'Transfer CFM']

        # Compute global standard deviations if normalize_global is True
        if normalize_global:
            reference_samples = samples_list[1]  # "Parallel Transfusion" samples
            masked_ref_samples = reference_samples[:, mask, :]  # shape: [S, N_mask, F]
            repeated_truth = masked_truth.unsqueeze(0).repeat(masked_ref_samples.shape[0], 1, 1)
            diff_ref = masked_ref_samples - repeated_truth  # shape: [S, N_mask, F]
            global_stds = diff_ref.std(dim=(0, 1), keepdim=False)  # shape: [F]
        else:
            global_stds = None  # Will be computed per model instead

        figures = {}  # Store individual plots

        for j, feat in enumerate(features):
            fig, ax = plt.subplots(figsize=(6, 5), dpi=300)  # Create a separate figure for each feature
            fig.suptitle(f"{title} - {feat}")

            # Define bins for normalized histogram
            normalized_bins = np.linspace(-5, 5, self.bins)

            # For each model, compute and plot the histogram
            for m, samples in enumerate(samples_list):
                masked_samples = samples[:, mask, :]  # shape: [S, N_mask, F]
                repeated_truth = masked_truth.unsqueeze(0).repeat(masked_samples.shape[0], 1, 1)
                diff = masked_samples - repeated_truth  # shape: [S, N_mask, F]

                if feat == "phi":
                    diff_feature = angle_diff(diff[..., j])
                else:
                    diff_feature = diff[..., j]

                # Choose normalization strategy
                if normalize_global:
                    sigma = global_stds[j].item()  
                else:
                    sigma = diff_feature.std().item()  
                if sigma == 0:
                    sigma = 1.0  

                normalized_diff = diff_feature / sigma

                # Plot the histogram for this model
                ax.hist(
                    normalized_diff.ravel(),
                    bins=normalized_bins,
                    density=True,
                    histtype='step',
                    linewidth=2,
                    color=line_colors[m],
                    label=model_labels[m]
                )

            # Overlay the standard normal distribution for reference
            normal_dist_x = np.linspace(-4.5, 5.5, 1000)
            standard_normal = stats.norm.pdf(normal_dist_x, 0, 1)
            ax.plot(normal_dist_x, standard_normal, 'k', linewidth=2, label="Normal")

            # Define custom legend handles
            legend_handles = [
                Line2D([0], [0], color='k', linewidth=2, linestyle='-', label='Normal'),
                Line2D([0], [0], color=line_colors[0], linewidth=2, linestyle='-', label=model_labels[0]),
                Line2D([0], [0], color=line_colors[1], linewidth=2, linestyle='-', label=model_labels[1]),
                Line2D([0], [0], color=line_colors[2], linewidth=2, linestyle='-', label=model_labels[2]),
            ]
            if normalize_global:
                legend_handles.append(Line2D([0], [0], color='none', label=fr'$\sigma \, = \, {sigma:.2f}$'))

            # Use mapped LaTeX name if available
            feature_name_axis = feature_info[feat]["latex"]
            ax.set_xlabel(
                fr'$({feature_name_axis}^{{\text{{model}}}} - {feature_name_axis}^{{\text{{true}}}}) \, / \, \sigma $',
                fontsize=22
            )
            ax.set_ylabel('Density', fontsize=22)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.set_yscale('log' if self.log_scale else 'linear')
            ax.set_xlim(-4.5, 5.5)
            ax.legend(frameon=False, handles=legend_handles, handlelength=1.5, loc='upper right', fontsize=14)

            # Store the figure
            figures[feat] = fig

        return figures


    def make_error_plots(
            self,
            model,
            show: bool = False,
            disable_tqdm: bool = False,
            external_samples: Optional[List[List[torch.Tensor]]] = None,
            normalize_global: bool = False
    ) -> Dict[str, plt.Figure]:
        """
        Generates difference histograms for each reco particle type and particle index by comparing the true data to
        external samples generated by up to three different models. If fewer than three sample sets are provided,
        it falls back to the available ones.

        Args:
            - model: The generative model used for sampling.
            - show (bool, optional): Whether to display plots interactively.
            - disable_tqdm (bool, optional): To disable progress bars.
            - external_samples (Optional[List[List[torch.Tensor]]], optional): Pre-made samples from up to 3 models.
              Each element of the outer list should be a list of torch.Tensors corresponding to each reco particle type.

        Returns:
            - Dict[str, plt.Figure]: A dictionary mapping figure names to their corresponding figures.
        """
        # Get the device
        device = self.device if self.device is not None else model.device
        model = model.to(device)

        N_reco = len(model.n_reco_particles_per_type)
        truth = [[] for _ in range(N_reco)]
        mask = [[] for _ in range(N_reco)]

        if external_samples is not None:
            # external_samples is assumed to be a list (for each model) of sample lists (one per reco type)
            n_models = len(external_samples)
            # Accumulate truth and mask over all batches
            for batch_idx, batch in enumerate(self.loader):
                if batch_idx >= self.N_batch:
                    break
                # Get the reco-level data and masks, moving them to the proper device
                reco_data = [data.to(device) for data in batch['reco']['data']]
                reco_mask_exist = [mask_tensor.to(device) for mask_tensor in batch['reco']['mask']]
                for i in range(N_reco):
                    truth[i].append(reco_data[i][..., model.flow_indices[i]].cpu())
                    mask[i].append(reco_mask_exist[i].cpu())
            truth = [torch.cat(t, dim=0) for t in truth]
            mask = [torch.cat(m, dim=0) for m in mask]
        else:
            raise ValueError('Must provide pre-made external samples from model(s)')

        # Adjust phi values for each model's samples per reco type
        for m in range(len(external_samples)):
            for i in range(len(external_samples[m])):
                for j, feat in enumerate(model.flow_input_features[i]):
                    if feat == "phi":
                        external_samples[m][i][..., j] = angle_diff(external_samples[m][i][..., j])

        # Apply preprocessing if provided, to both truth and each model's external samples
        if self.preprocessing is not None:
            for i in range(len(truth)):
                name = model.reco_particle_type_names[i]
                fields = model.reco_input_features_per_type[i]
                flow_fields = [fields[idx] for idx in model.flow_indices[i]]
                truth[i], _ = self.preprocessing.inverse(
                    name=name,
                    x=truth[i],
                    mask=mask[i],
                    fields=flow_fields,
                )
                for m in range(len(external_samples)):
                    samples_tensor = external_samples[m][i]
                    processed_samples = self.preprocessing.inverse(
                        name=name,
                        x=samples_tensor.reshape(self.N_sample * samples_tensor.shape[1],
                                                 samples_tensor.shape[2],
                                                 samples_tensor.shape[3]).cpu(),
                        mask=mask[i].unsqueeze(0).repeat_interleave(self.N_sample, dim=0)
                              .reshape(self.N_sample * mask[i].shape[0], mask[i].shape[1]).cpu(),
                        fields=flow_fields,
                    )[0].reshape(self.N_sample,
                                 samples_tensor.shape[1],
                                 samples_tensor.shape[2],
                                 samples_tensor.shape[3])
                    external_samples[m][i] = processed_samples

        # Make the bias plots
        figs = {}
        # Loop over reco particle types
        for i, (truth_type, mask_type) in enumerate(zip(truth, mask)):
            # For each reco type, collect the corresponding sample tensors from all models
            samples_list_for_type = [external_samples[m][i] for m in range(len(external_samples))]
            # Loop over particles within this reco type (e.g., different jets)
            for j in range(truth_type.shape[1]):
                fig = self.plot_particle(
                    truth=truth_type[:, j, :],
                    mask=mask_type[:, j],
                    samples_list=[s[:, :, j, :] for s in samples_list_for_type],
                    features=model.flow_input_features[i],
                    title=f'{model.reco_particle_type_names[i]} #{j}',
                    normalize_global=normalize_global
                )
                figure_name = f'{model.reco_particle_type_names[i]}_{j}_bias'
                if self.suffix:
                    figure_name += f'_{self.suffix}'
                print(figure_name)

                # Save each figure in the dictionary separately
                for feature_name, fig in fig.items():
                    figure_name = f'{model.reco_particle_type_names[i]}_{j}_{feature_name}_bias'
                    if self.suffix:
                        figure_name += f'_{self.suffix}'

                    save_path = f'sampling_plots/histogram_errors_{figure_name}.png'
                    fig.savefig(save_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)  # Close to free memory
                    figs[figure_name] = fig  # Store the figure in the dictionary
                    print(f"Saved: {save_path}")

                    if show:
                        fig.show()  # Show each figure individually

        return figs



class ModelCheckpoint(L.Callback):
    """
    Callback during training to save the model's hyperparameters and weights.
    Can resume training from a checkpoint or use the checkpoint file for inference.
    Avoids having to rerun the model training every time log back in.

    Parameters:
     - save_every_n_epochs (int): How frequently during training to save a checkpoint.
     - save_dir (str): Directory to save all the model .ckpt files.
    """
    def __init__(
            self,
            save_every_n_epochs: int = 10,
            save_dir: str = "model_checkpoints"
    ):
        super().__init__()
        self.save_every_n_epochs = save_every_n_epochs
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)


    def on_train_epoch_end(self,trainer, pl_module):
        epoch = trainer.current_epoch  # Zero-indexed epoch
        # Save after every save_every_n_epochs (adjusting for zero-indexing)
        if (epoch + 1) % self.save_every_n_epochs == 0:
            ckpt_path = os.path.join(self.save_dir, f"model_epoch_{epoch+1}.ckpt")
            trainer.save_checkpoint(ckpt_path)
            print(f"Checkpoint saved at epoch {epoch+1}: {ckpt_path}")