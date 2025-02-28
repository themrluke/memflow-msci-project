import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import vector
import awkward as ak
import os

vector.register_awkward()

class torch_wrapper(torch.nn.Module):
    """Wraps a model to a torchdyn-compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        # Concatenates the time tensor t to the input x.
        return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))


###############################################################################
# FEATURE DISTRIBUTIONS CLASS
###############################################################################
class FeatureDistributions:
    """
    Class to compare histograms of real vs. generated data features.

    Methods:
        compare_distributions: Compare one generated dataset to the truth.
        compare_distributions_multiple: Compare two generated datasets to the truth.
    """

    def __init__(self, model, preprocessing=None, real_mask=None):
        """
        Parameters:
            model: An object with attributes:
                   - reco_particle_type_names: list of str
                   - reco_input_features_per_type: list (per particle type) of list of str
                   - flow_indices: list (per particle type) of list of int
            preprocessing: Optional object with an inverse() method.
            real_mask: Optional torch.Tensor mask.
            log_scale: Whether to use logarithmic y-scale for plots.
        """
        self.model = model
        self.preprocessing = preprocessing
        self.real_mask = real_mask

    def compare_distributions(self, real_data, gen_data, ptype_idx,
                              feat_idx=0, nbins=50, feat_name="Feature"):
        """
        Compare histograms of real vs. generated data for a single feature,
        with a ratio subplot.
        """
        # Select the correct particle type and mask
        real_data = real_data[ptype_idx]
        real_mask = self.real_mask[ptype_idx]
        gen_data = gen_data[ptype_idx]

        # Get field names.
        real_fields = self.model.reco_input_features_per_type[ptype_idx]
        gen_fields = [real_fields[idx] for idx in self.model.flow_indices[ptype_idx]]

        if self.preprocessing is not None:
            name = self.model.reco_particle_type_names[ptype_idx]
            real_data = real_data.cpu()
            gen_data = gen_data.cpu()
            if gen_data.ndim == 4:
                gen_data = gen_data.reshape(gen_data.shape[0] * gen_data.shape[1],
                                            gen_data.shape[2],
                                            gen_data.shape[3])
            if real_mask is not None:
                real_mask = real_mask.cpu()
            else:
                real_mask = torch.ones(real_data.shape[0], real_data.shape[1], dtype=torch.bool)

            N_sample = gen_data.shape[0] // real_data.shape[0]
            gen_mask = real_mask.repeat((N_sample, 1))

            # Debug prints (can be commented out)
            print("gen_mask shape:", gen_mask.shape)
            print("real_mask shape:", real_mask.shape)
            masked_percentage = 100 * (gen_mask == False).sum().item() / gen_mask.numel()
            print(f"Percentage of masked jets: {masked_percentage:.2f}%")
            real_masked_percentage = 100 * (real_mask == False).sum().item() / real_mask.numel()
            print(f"Percentage of real masked jets: {real_masked_percentage:.2f}%")
            print('real data shape before inverse preprocessing', real_data.shape)
            print('gen data shape before inverse preprocessing', gen_data.shape)

            # Apply inverse preprocessing.
            real_data, _ = self.preprocessing.inverse(
                name=name,
                x=real_data,
                mask=real_mask,
                fields=real_fields
            )
            gen_data, _ = self.preprocessing.inverse(
                name=name,
                x=gen_data,
                mask=gen_mask,
                fields=gen_fields
            )

        # Flatten values using the mask.
        real_data = real_data[real_mask.bool()]
        gen_data = gen_data[gen_mask.bool()]
        if ptype_idx == 1 and feat_idx == 1:  # special case for MET phi
            real_vals = real_data[..., feat_idx+1].cpu().numpy().ravel()
        else:
            real_vals = real_data[..., feat_idx].cpu().numpy().ravel()
        gen_vals = gen_data[..., feat_idx].cpu().numpy().ravel()

        # Compute histogram bins.
        bins = np.linspace(min(real_vals.min(), gen_vals.min()),
                           max(real_vals.max(), gen_vals.max()),
                           nbins + 1)
        # Compute density-normalized histograms.
        hist_real, _ = np.histogram(real_vals, bins=bins, density=True)
        hist_gen, _ = np.histogram(gen_vals, bins=bins, density=True)
        real_counts, _ = np.histogram(real_vals, bins=bins)
        gen_counts, _ = np.histogram(gen_vals, bins=bins)
        bin_widths = np.diff(bins)
        total_real = np.sum(real_counts)
        total_gen = np.sum(gen_counts)
        real_errors = np.sqrt(real_counts) / (total_real * bin_widths)
        gen_errors = (np.sqrt(gen_counts) / (total_gen * bin_widths)) * np.sqrt(gen_data.shape[0] // real_data.shape[0])
        ratio = np.divide(hist_gen, hist_real, where=hist_real > 0)
        real_uncertainty = real_errors / hist_real

        # Create figure.
        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0},
                                sharex=True, figsize=(6, 5), dpi=300)
        axs[0].step(bins[:-1], hist_real, where="post",
                    label="Truth", linewidth=1.8, color='#1f77b4')
        axs[0].fill_between(bins[:-1], hist_real - real_errors, hist_real + real_errors,
                            step="post", color='#1f77b4', alpha=0.3)
        axs[0].step(bins[:-1], hist_gen, where="post",
                    label="Generated", linewidth=1.8, color='#ff7f0e')
        axs[0].fill_between(bins[:-1], hist_gen - gen_errors, hist_gen + gen_errors,
                            step="post", color='#ff7f0e', alpha=0.3)
        axs[0].set_ylabel("Density", fontsize=16)
        axs[0].legend(fontsize=12)
        axs[0].tick_params(axis='x', which='both', length=0, labelbottom=False)

        axs[1].axhline(1.0, color='black', linestyle='dashed', linewidth=1)
        axs[1].step(bins[:-1], ratio, where="post",
                    color='#ff7f0e', linewidth=1.5, label=r"$\frac{Gen}{Truth}$")
        axs[1].fill_between(bins[:-1],
                            1 - real_uncertainty,
                            1 + real_uncertainty,
                            step="post", color='#1f77b4', alpha=0.3)
        axs[1].set_ylabel(r"$\frac{\text{Gen}}{\text{Truth}}$", fontsize=16)
        axs[1].set_xlabel(feat_name, fontsize=16)


        axs[0].set_yscale("log")
        if ptype_idx == 0:
            if feat_idx == 0:
                axs[0].set_xlim(30, 1500)
                axs[0].set_ylim(2e-8, 1e-2)
                axs[1].set_ylim(0.5, 1.5)
            elif feat_idx == 1:
                axs[0].set_xlim(-5, 5)
                axs[0].set_ylim(3e-4, 1e0)
                axs[1].set_ylim(0.8, 1.2)
        if ptype_idx == 1:
            if feat_idx == 0:
                axs[0].set_xlim(200, 1200)
                axs[0].set_ylim(3e-7, 1e-2)
                axs[1].set_ylim(0.5, 1.5)

        if ptype_idx == 0 and feat_idx == 2:
            axs[0].ticklabel_format(style='sci', axis='y', scilimits=(-1,1))
            axs[0].set_xlim(-math.pi, math.pi)
        if ptype_idx == 1 and feat_idx == 1:
            axs[0].ticklabel_format(style='sci', axis='y', scilimits=(-1,1))
            axs[0].set_xlim(-math.pi, math.pi)

        plt.tight_layout()
        plt.show()

    def compare_distributions_multiple(self, real_data, gen_data_1, gen_data_2, ptype_idx,
                                         feat_idx=0, nbins=50, feat_name="Feature"):
        """
        Compare histograms of real vs. two generated datasets for a single feature,
        with a ratio subplot.
        """
        real_data = real_data[ptype_idx]
        real_mask = self.real_mask[ptype_idx]
        gen_data_1 = gen_data_1[ptype_idx]
        gen_data_2 = gen_data_2[ptype_idx]
        real_fields = self.model.reco_input_features_per_type[ptype_idx]
        gen_fields = [real_fields[idx] for idx in self.model.flow_indices[ptype_idx]]

        if self.preprocessing is not None:
            name = self.model.reco_particle_type_names[ptype_idx]
            real_data = real_data.cpu()
            gen_data_1 = gen_data_1.cpu()
            gen_data_2 = gen_data_2.cpu()
            if gen_data_1.ndim == 4:
                gen_data_1 = gen_data_1.reshape(gen_data_1.shape[0] * gen_data_1.shape[1],
                                                gen_data_1.shape[2], gen_data_1.shape[3])
            if gen_data_2.ndim == 4:
                gen_data_2 = gen_data_2.reshape(gen_data_2.shape[0] * gen_data_2.shape[1],
                                                gen_data_2.shape[2], gen_data_2.shape[3])
            if real_mask is not None:
                real_mask = real_mask.cpu()
            else:
                real_mask = torch.ones(real_data.shape[0], real_data.shape[1], dtype=torch.bool)
            N_sample_1 = gen_data_1.shape[0] // real_data.shape[0]
            N_sample_2 = gen_data_2.shape[0] // real_data.shape[0]
            gen_mask_1 = real_mask.repeat((N_sample_1, 1))
            gen_mask_2 = real_mask.repeat((N_sample_2, 1))
            real_data, _ = self.preprocessing.inverse(name=name, x=real_data, mask=real_mask, fields=real_fields)
            gen_data_1, _ = self.preprocessing.inverse(name=name, x=gen_data_1, mask=gen_mask_1, fields=gen_fields)
            gen_data_2, _ = self.preprocessing.inverse(name=name, x=gen_data_2, mask=gen_mask_2, fields=gen_fields)

        real_data = real_data[real_mask.bool()]
        gen_data_1 = gen_data_1[gen_mask_1.bool()]
        gen_data_2 = gen_data_2[gen_mask_2.bool()]

        if ptype_idx == 1 and feat_idx == 1:
            real_vals = real_data[..., feat_idx+1].cpu().numpy().ravel()
        else:
            real_vals = real_data[..., feat_idx].cpu().numpy().ravel()
        gen_vals_1 = gen_data_1[..., feat_idx].cpu().numpy().ravel()
        gen_vals_2 = gen_data_2[..., feat_idx].cpu().numpy().ravel()

        bins = np.linspace(min(real_vals.min(), gen_vals_1.min(), gen_vals_2.min()),
                           max(real_vals.max(), gen_vals_1.max(), gen_vals_2.max()),
                           nbins + 1)
        hist_real, _ = np.histogram(real_vals, bins=bins, density=True)
        hist_gen_1, _ = np.histogram(gen_vals_1, bins=bins, density=True)
        hist_gen_2, _ = np.histogram(gen_vals_2, bins=bins, density=True)
        real_counts, _ = np.histogram(real_vals, bins=bins)
        gen_counts_1, _ = np.histogram(gen_vals_1, bins=bins)
        gen_counts_2, _ = np.histogram(gen_vals_2, bins=bins)
        bin_widths = np.diff(bins)
        total_real = np.sum(real_counts)
        total_gen_1 = np.sum(gen_counts_1)
        total_gen_2 = np.sum(gen_counts_2)
        real_errors = np.sqrt(real_counts) / (total_real * bin_widths)
        gen_errors_1 = np.sqrt(gen_counts_1) / (total_gen_1 * bin_widths) * np.sqrt(N_sample_1)
        gen_errors_2 = np.sqrt(gen_counts_2) / (total_gen_2 * bin_widths) * np.sqrt(N_sample_2)
        ratio_1 = np.divide(hist_gen_1, hist_real, where=hist_real > 0)
        ratio_2 = np.divide(hist_gen_2, hist_real, where=hist_real > 0)
        ratio_error_1 = np.divide(gen_errors_1, hist_real, where=hist_real > 0)
        ratio_error_2 = np.divide(gen_errors_2, hist_real, where=hist_real > 0)

        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0},
                                sharex=True, figsize=(6, 5), dpi=300)
        axs[0].step(bins[:-1], hist_real, where="post", label="Truth", linewidth=1.5, color='#1f77b4')
        axs[0].step(bins[:-1], hist_gen_1, where="post", label="Transfermer", linewidth=1.5, color='#d62728')
        axs[0].step(bins[:-1], hist_gen_2, where="post", label="Parallel Transfusion", linewidth=1.5, color='#2ca02c')
        axs[0].fill_between(bins[:-1], hist_real - real_errors, hist_real + real_errors,
                            step="post", color='#1f77b4', alpha=0.3)
        axs[0].fill_between(bins[:-1], hist_gen_1 - gen_errors_1, hist_gen_1 + gen_errors_1,
                            step="post", color='#d62728', alpha=0.3)
        axs[0].fill_between(bins[:-1], hist_gen_2 - gen_errors_2, hist_gen_2 + gen_errors_2,
                            step="post", color='#2ca02c', alpha=0.3)
        axs[0].set_ylabel("Density", fontsize=16)
        axs[0].legend(fontsize=10)

        axs[1].axhline(1.0, color='black', linestyle='dashed', linewidth=1)
        axs[1].step(bins[:-1], ratio_1, where="post", color='#d62728', linewidth=1.5, label="Gen 1 / Truth")
        axs[1].step(bins[:-1], ratio_2, where="post", color='#2ca02c', linewidth=1.5, label="Gen 2 / Truth")
        axs[1].fill_between(bins[:-1], ratio_1 - ratio_error_1, ratio_1 + ratio_error_1,
                            step="post", color='#d62728', alpha=0.3)
        axs[1].fill_between(bins[:-1], ratio_2 - ratio_error_2, ratio_2 + ratio_error_2,
                            step="post", color='#2ca02c', alpha=0.3)
        axs[1].set_ylabel(r"$\frac{\text{Gen}}{\text{Truth}}$", fontsize=16)
        axs[1].set_xlabel(feat_name, fontsize=16)

        axs[0].tick_params(axis='y', which='major', labelsize=10)
        axs[1].tick_params(axis='both', which='major', labelsize=10)
        axs[0].tick_params(axis='y', which='minor', labelsize=10)
        axs[1].tick_params(axis='both', which='minor', labelsize=10)

        if ptype_idx == 0:
            if feat_idx == 0:
                axs[0].set_yscale("log")
                axs[0].set_xlim(30, 1500)
                axs[0].set_ylim(2e-8, 1e-2)
                axs[1].set_ylim(0.5, 1.5)
            elif feat_idx == 1:
                axs[0].set_yscale("log")
                axs[0].set_xlim(-5, 5)
                axs[0].set_ylim(3e-4, 1e0)
                axs[1].set_ylim(0.8, 1.2)
            elif feat_idx == 3:
                axs[0].set_yscale("log")
                axs[0].set_xlim(0, 160)
                axs[0].set_ylim(3e-7, 1e-1)
                axs[1].set_ylim(0.5, 1.5)
        if ptype_idx == 1:
            if feat_idx == 0:
                axs[0].set_yscale("log")
                axs[0].set_xlim(200, 1200)
                axs[0].set_ylim(3e-7, 1e-2)
                axs[1].set_ylim(0.5, 1.5)

        if ptype_idx == 0 and feat_idx == 2:
            axs[0].ticklabel_format(style='sci', axis='y', scilimits=(-1,1))
            axs[0].set_xlim(-math.pi, math.pi)
        if ptype_idx == 1 and feat_idx == 1:
            axs[0].ticklabel_format(style='sci', axis='y', scilimits=(-1,1))
            axs[0].set_xlim(-math.pi, math.pi)

        plt.tight_layout()
        plt.show()


###############################################################################
# TRAJECTORIES PLOTS CLASS
###############################################################################
class TrajectoriesPlots:
    """
    Class to create trajectory plots.

    Methods:
        plot_trajectories_2d: Plots a 2D scatter of trajectories (start, intermediate, end).
        plot_trajectories_grid: Plots a grid with density, vector field, and trajectories.
    """

    def __init__(self, model):
        self.model = model
        self.device = model.device

    def plot_trajectories_2d(self, all_traj: torch.Tensor,
                             type_idx: int = 0,
                             feat_idx_x: int = 0,
                             feat_idx_y: int = 1,
                             max_points: int = 2000,
                             num_events: int = 5,
                             mode: str = "multiple_events",
                             event_idx: int = 0,
                             object_idx: int = 0,
                             preprocessing=None,
                             batch=None):
        """
        2D scatter plot from initial to final positions for the chosen particle type.
        """
        N_sample, steps_plus_1, B, sum_reco, len_flow_feats = all_traj.shape
        offset = sum(self.model.n_reco_particles_per_type[:type_idx])
        n_type = self.model.n_reco_particles_per_type[type_idx]

        if mode == "multiple_events":
            num_events = min(num_events, B)
            sub_traj = all_traj[:, :, :num_events, offset: offset + n_type, :]
            if preprocessing is None:
                sub_traj_2d = sub_traj[..., [feat_idx_x, feat_idx_y]]
                sub_traj_2d = sub_traj_2d.reshape(N_sample, steps_plus_1, -1, 2)
                sample_idx = 0
                traj = sub_traj_2d[sample_idx]
            else:
                full_traj = sub_traj[0]
                full_traj = full_traj.reshape(steps_plus_1, -1, len_flow_feats)
                fields = self.model.reco_input_features_per_type[type_idx]
                if hasattr(self.model, 'flow_indices'):
                    desired_num = len(self.model.flow_indices[type_idx])
                    if full_traj.shape[-1] != desired_num:
                        indices = self.model.flow_indices[type_idx]
                        indices_tensor = torch.tensor(indices, device=self.device)
                        full_traj = full_traj.index_select(dim=-1, index=indices_tensor)
                    flow_fields = [fields[idx] for idx in self.model.flow_indices[type_idx]]
                else:
                    flow_fields = fields
                inv_mask = torch.ones(full_traj.shape[0], full_traj.shape[1], device=self.device)
                name = self.model.reco_particle_type_names[type_idx]
                inv_data, _ = preprocessing.inverse(name=name, x=full_traj, mask=inv_mask, fields=flow_fields)
                traj = inv_data[..., [feat_idx_x, feat_idx_y]]
        elif mode == "single_event":
            reco_data = [d.to(self.device) for d in batch['reco']['data']]
            reco_mask_exist = [m.to(self.device) for m in batch['reco']['mask']]
            reco_data = [r.cpu() for r in reco_data]
            reco_mask_exist = [m.cpu() for m in reco_mask_exist]
            if event_idx >= B:
                raise ValueError(f'Event index ({event_idx}) larger than batch_size ({B})')
            if object_idx >= n_type:
                raise ValueError(f'Object index {object_idx} out of range {n_type}')
            sub_traj = all_traj[:, :, event_idx:event_idx+1, offset: offset+n_type, :]
            sub_traj = sub_traj[:, :, 0, object_idx, :]
            if preprocessing is None:
                traj = sub_traj[..., [feat_idx_x, feat_idx_y]]
            else:
                raw_traj = sub_traj.clone()
                name = self.model.reco_particle_type_names[type_idx]
                fields = list(self.model.reco_input_features_per_type[type_idx])
                if hasattr(self.model, 'flow_indices'):
                    flow_fields = [fields[idx] for idx in self.model.flow_indices[type_idx]]
                    indices_tensor = torch.tensor(self.model.flow_indices[type_idx], device=sub_traj.device)
                    sub_traj = sub_traj.index_select(dim=-1, index=indices_tensor)
                    raw_traj = raw_traj.index_select(dim=-1, index=indices_tensor)
                else:
                    flow_fields = fields
                    indices_tensor = None
                N_sample_local, T, F_sel = sub_traj.shape
                reshaped = sub_traj.reshape(N_sample_local * T, 1, F_sel)
                inv_mask = torch.ones(reshaped.shape[0], 1, device=sub_traj.device)
                inv_data, _ = preprocessing.inverse(name=name, x=reshaped, mask=inv_mask, fields=flow_fields)
                inv_data = inv_data.squeeze(1).reshape(N_sample_local, T, -1)
                traj = inv_data.clone()
                chosen_features = self.model.flow_input_features[type_idx]
                if chosen_features[feat_idx_x] == "phi":
                    traj[..., feat_idx_x] = raw_traj[..., feat_idx_x]
                if chosen_features[feat_idx_y] == "phi":
                    traj[..., feat_idx_y] = raw_traj[..., feat_idx_y]
                traj = traj[..., [feat_idx_x, feat_idx_y]]
        else:
            raise ValueError("Invalid mode. Choose 'multiple_events' or 'single_event'.")

        feature_names = {"pt": r"$p_T$ [GeV]", "eta": r"$\eta$", "phi": r"$\phi$ [rad]"}
        chosen_features = self.model.flow_input_features[type_idx]
        x_label = feature_names.get(chosen_features[feat_idx_x], chosen_features[feat_idx_x])
        y_label = feature_names.get(chosen_features[feat_idx_y], chosen_features[feat_idx_y])
        particle_name = self.model.reco_particle_type_names[type_idx]

        plt.figure(figsize=(6,6))
        if mode == "multiple_events":
            plt.scatter(traj[0, :, 0], traj[0, :, 1], s=5, c="black", alpha=0.8, label="Start", zorder=1)
            for i in range(traj.shape[1]):
                plt.plot(traj[:, i, 0], traj[:, i, 1], c="olive", alpha=0.2, linewidth=0.8, zorder=2)
            plt.scatter(traj[-1, :, 0], traj[-1, :, 1], s=5, c="royalblue", alpha=1.0, label="End", zorder=3)
        else:
            for i in range(traj.shape[0]):
                plt.plot(traj[i, :, 0], traj[i, :, 1], c="olive", alpha=0.5, linewidth=0.8)
                plt.scatter(traj[i, 0, 0], traj[i, 0, 1], s=5, c="black", alpha=0.8, zorder=1)
                plt.scatter(traj[i, -1, 0], traj[i, -1, 1], s=5, c="royalblue", alpha=1.0, zorder=2)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f"Trajectory for {particle_name} ({mode})")
        plt.legend()
        plt.show()

    def plot_trajectories_grid(self, all_traj: torch.Tensor, custom_timesteps,
                               type_idx: int = 0,
                               feat_idx_x: int = 0,
                               feat_idx_y: int = 1,
                               max_points: int = 2000,
                               event_idx: int = 0,
                               object_idx: int = 0,
                               batch=None,
                               grid_size=20):
        """
        Plots a grid of 3 rows per custom time:
          1) Density heatmap
          2) Velocity vector field (from model.velocity_net)
          3) Trajectories highlighting the current position.
        """
        device = self.device
        single_event_batch = {
            "hard": {
                "data": [d[event_idx: event_idx+1].to(device) for d in batch["hard"]["data"]],
                "mask": [m[event_idx: event_idx+1].to(device) for m in batch["hard"]["mask"]],
            },
            "reco": {
                "data": [d[event_idx: event_idx+1].to(device) for d in batch["reco"]["data"]],
                "mask": [m[event_idx: event_idx+1].to(device) for m in batch["reco"]["mask"]],
            }
        }
        N_sample, steps_plus_1, B, sum_reco, len_flow_feats = all_traj.shape
        offset = sum(self.model.n_reco_particles_per_type[:type_idx])
        n_type = self.model.n_reco_particles_per_type[type_idx]
        if event_idx >= B:
            raise ValueError(f'Event index ({event_idx}) larger than batch_size ({B})')
        if object_idx >= n_type:
            raise ValueError(f'Object index ({object_idx}) out of range {n_type})')
        sub_traj = all_traj[:, :, event_idx:event_idx+1, offset: offset+n_type, :]
        sample_traj = sub_traj[:, :, 0, object_idx, :]
        sample_traj = sample_traj.transpose(0, 1)
        traj = sample_traj[..., [feat_idx_x, feat_idx_y]]
        if traj.shape[1] > max_points:
            print(f'Selecting {max_points} out of {traj.shape[1]} to plot.')
            traj = traj[:, :max_points, :]

        global_x_min = traj[:, :, 0].min().item()
        global_x_max = traj[:, :, 0].max().item()
        global_y_min = traj[:, :, 1].min().item()
        global_y_max = traj[:, :, 1].max().item()
        n_cols = len(custom_timesteps)
        fig, axes = plt.subplots(3, n_cols, figsize=(5 * n_cols, 15))
        if n_cols == 1:
            axes = axes[:, None]

        feature_names = {"pt": r"$p_T$", "eta": r"$\eta$", "phi": r"$\phi$"}
        chosen_features = self.model.flow_input_features[type_idx]
        x_label = feature_names.get(chosen_features[feat_idx_x], chosen_features[feat_idx_x])
        y_label = feature_names.get(chosen_features[feat_idx_y], chosen_features[feat_idx_y])
        particle_name = self.model.reco_particle_type_names[type_idx]

        with torch.no_grad():
            cond_out = self.model.conditioning(
                single_event_batch["hard"]["data"], single_event_batch["hard"]["mask"],
                single_event_batch["reco"]["data"], single_event_batch["reco"]["mask"],
            )
            context_full = cond_out[:, 1:, :]
        offset_obj = offset + object_idx
        obj_context = context_full[:, offset_obj: offset_obj+1, :]

        Nx = grid_size
        Ny = grid_size
        xs = np.linspace(global_x_min, global_x_max, Nx)
        ys = np.linspace(global_y_min, global_y_max, Ny)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        points_2d = torch.from_numpy(np.stack([gx.ravel(), gy.ravel()], axis=-1)).float().to(device)
        all_mags = []
        for t in custom_timesteps:
            t = int(round(t))
            if t < 0 or t >= steps_plus_1:
                continue
            t_val = t / (steps_plus_1 - 1)
            t_tensor = torch.full((points_2d.shape[0], 1), t_val, device=device)
            sin_phi = torch.zeros_like(points_2d[:, 0])
            cos_phi = torch.ones_like(points_2d[:, 0])
            points_4d = torch.cat([points_2d, sin_phi.unsqueeze(1), cos_phi.unsqueeze(1)], dim=1)
            cflat = obj_context.reshape(1, -1)
            c_rep = cflat.repeat(points_2d.shape[0], 1)
            net_in = torch.cat([c_rep, points_4d, t_tensor], dim=1)
            with torch.no_grad():
                v_pred_4 = self.model.velocity_net(net_in)
            v_pred = v_pred_4[:, [0, 1]].cpu().numpy()
            mag = np.sqrt(v_pred[:, 0]**2 + v_pred[:, 1]**2)
            all_mags.append(mag)
        global_max = np.max([m.max() for m in all_mags])
        global_min = np.min([m.min() for m in all_mags])
        shared_norm = matplotlib.colors.Normalize(vmin=global_min, vmax=global_max)

        for col, t in enumerate(custom_timesteps):
            t = int(round(t))
            if t < 0 or t >= steps_plus_1:
                print(f"Skipping timestep {t} (out of range)")
                continue
            fontsize = 20
            points = traj[t].cpu().numpy()
            axes[0, col].hist2d(points[:, 0], points[:, 1],
                                bins=50,
                                density=True,
                                cmap="viridis",
                                range=[[global_x_min, global_x_max], [global_y_min, global_y_max]])
            axes[0, col].set_title(f"T = {t / (steps_plus_1 - 1)}", fontsize=25)
            axes[0, col].set_xlabel(x_label, fontsize=fontsize)
            axes[0, col].set_ylabel(y_label, fontsize=fontsize)
            axes[0, col].set_xlim(global_x_min, global_x_max)
            axes[0, col].set_ylim(global_y_min, global_y_max)

            t_val = t / (steps_plus_1 - 1)
            t_tensor = torch.full((points_2d.shape[0], 1), t_val, device=device)
            sin_phi = torch.zeros_like(points_2d[:, 0])
            cos_phi = torch.ones_like(points_2d[:, 0])
            points_4d = torch.cat([points_2d, sin_phi.unsqueeze(1), cos_phi.unsqueeze(1)], dim=1)
            cflat = obj_context.reshape(1, -1)
            c_rep = cflat.repeat(points_2d.shape[0], 1)
            net_in = torch.cat([c_rep, points_4d, t_tensor], dim=1)
            with torch.no_grad():
                v_pred_4 = self.model.velocity_net(net_in)
            v_pred = v_pred_4[:, [0, 1]].cpu().numpy()
            vx = v_pred[:, 0].reshape(Nx, Ny)
            vy = v_pred[:, 1].reshape(Nx, Ny)
            mag = np.sqrt(vx**2 + vy**2)
            ax_mid = axes[1, col]
            Q = ax_mid.quiver(gx, gy, vx, vy, mag, pivot='mid', cmap='coolwarm',
                              scale=3, scale_units='xy', angles='xy', width=0.02, norm=shared_norm)
            ax_mid.set_xlim(global_x_min, global_x_max)
            ax_mid.set_ylim(global_y_min, global_y_max)
            ax_mid.set_xlabel(x_label, fontsize=fontsize)
            ax_mid.set_ylabel(y_label, fontsize=fontsize)

            num_traj = traj.shape[1]
            idx = np.random.choice(num_traj, size=min(10000, num_traj), replace=False)
            for i in idx:
                axes[2, col].scatter(traj[0, i, 0].cpu().numpy(), traj[0, i, 1].cpu().numpy(),
                                     s=7, color="black", alpha=0.8, zorder=1)
                axes[2, col].plot(traj[:t+1, i, 0].cpu().numpy(), traj[:t+1, i, 1].cpu().numpy(),
                                  color="olive", alpha=0.5, linewidth=0.8, zorder=2)
                axes[2, col].scatter(traj[t, i, 0].cpu().numpy(), traj[t, i, 1].cpu().numpy(),
                                     s=7, color="blue", alpha=1.0, zorder=3)
            axes[2, col].set_xlabel(x_label, fontsize=fontsize)
            axes[2, col].set_ylabel(y_label, fontsize=fontsize)
            axes[2, col].set_xlim(global_x_min, global_x_max)
            axes[2, col].set_ylim(global_y_min, global_y_max)

        plt.tight_layout()
        plt.show()


###############################################################################
# HIGH-LEVEL DISTRIBUTIONS CLASS
###############################################################################
class HighLevelDistributions:
    """
    Class for plotting high-level observables using vectorized operations.
    
    Observables include:
      - Energy of the leading jet (E_j1)
      - Transverse momentum of the leading jet (pT_j1)
      - Δϕ between j1 and j2 (using jets[:,0].deltaphi(jets[:,1]))
      - ΔR between j1 and j2 (using jets[:,0].deltaR(jets[:,1]))
      - ΔR between MET and dijet (using vector addition)
      - H_T, the scalar sum of jet pₜ's
      - Minimum dijet invariant mass
       
    The input jet data is assumed to have shape (n_events, nJets, n_features),
    with features ordered as specified by feat_idx_map (e.g. {"pt": 0, "eta": 1, "phi": 2, "mass": 3}).
    
    The jets are provided in “btag” order by default (i.e. the first two jets are the btag‐ordered ones)
    but if you set jet_ordering to "pt", then the jets will be completely reordered by descending pₜ.
    """
    def __init__(self, model, preprocessing, real_data: torch.Tensor, real_mask: torch.Tensor, 
                 gen_data: torch.Tensor, feat_idx_map: dict, 
                 gen_data2: torch.Tensor = None, jet_ordering="btag"):
        self.feat_idx_map = feat_idx_map
        self.jet_ordering = jet_ordering
        # If real_data has shape [n_events, nJets, 5] with last dimension [pt, eta, phi, mass, btag],
        # drop the last column so that real_data matches the 4-feature shape:

        # Undo preprocessing for jets and MET (for truth and gen_data1)
        jets_real, jets_real_mask, jets_gen, jets_gen_mask = undo_preprocessing(
            model, real_data[0], model.reco_input_features_per_type[0], real_mask[0],
            gen_data[0], [model.reco_input_features_per_type[0][i] for i in model.flow_indices[0]],
            0, preprocessing
        )
        met_real, _, met_gen, _ = undo_preprocessing(
            model, real_data[1], model.reco_input_features_per_type[1], real_mask[1],
            gen_data[1], [model.reco_input_features_per_type[1][i] for i in model.flow_indices[1]],
            1, preprocessing
        )
        print("Before reshaping: jets_gen.shape =", jets_gen.shape)
        # Reshape generated data: for jets_gen and met_gen
        n_events = jets_real.shape[0]
        gen_total = jets_gen.shape[0]  # This equals (n_samples * n_events)
        n_samples = gen_total // n_events
        jets_gen = jets_gen.reshape(n_samples, n_events, *jets_gen.shape[1:])
        # For MET, we assume one object per event; its shape remains (n_events, 1, n_met_features)
        if met_gen is not None:
            met_total = met_gen.shape[0]
            n_samples_met = met_total // n_events
            met_gen = met_gen.reshape(n_samples_met, n_events, *met_gen.shape[1:])

        # For gen_data2, reuse the same masks as for gen_data1.
        if gen_data2 is not None:

            # Use undo_preprocessing for gen_data2, but reuse real_mask[0] for jets and real_mask[1] for MET.
            _, _, jets_gen2 , _ = undo_preprocessing(
                model, real_data[0], model.reco_input_features_per_type[0], real_mask[0],
                gen_data2[0], [model.reco_input_features_per_type[0][i] for i in model.flow_indices[0]],
                0, preprocessing
            )
            print("Before reshaping: jets_gen2.shape =", jets_gen2.shape)
            gen2_total = jets_gen2.shape[0]  # This equals (n_samples * n_events)
            n_samples2 = gen2_total // n_events
            jets_gen2 = jets_gen2.reshape(n_samples2, n_events, *jets_gen2.shape[1:])
            _, _ , met_gen2 , _ = undo_preprocessing(
                model, real_data[1], model.reco_input_features_per_type[1], real_mask[1],
                gen_data2[1], [model.reco_input_features_per_type[1][i] for i in model.flow_indices[1]],
                1, preprocessing
            )
            met_gen2 = met_gen2.reshape(n_samples_met, n_events, *met_gen2.shape[1:])
        else:
            jets_gen2 = met_gen2 = None

        # Convert tensors to NumPy arrays.
        self.jets_real = jets_real.cpu().numpy()  # Truth jets (shape: [n_events, nJets, n_truth_features])
        self.jets_gen = jets_gen.cpu().numpy()    # Gen Model 1 jets (shape: [n_samples, n_events, nJets, n_gen_features])
        self.met_real = met_real.cpu().numpy() if met_real is not None else None
        self.met_gen = met_gen.cpu().numpy() if met_gen is not None else None
        if jets_gen2 is not None:
            self.jets_gen2 = jets_gen2.cpu().numpy()
            self.met_gen2 = met_gen2.cpu().numpy() if met_gen2 is not None else None

        # Optionally reorder jets by descending pT if jet_ordering=="pt".
        if self.jet_ordering == "pt":
            pt_idx = self.feat_idx_map["pt"]
            # Reorder truth jets: shape is [n_events, nJets, n_truth_features]
            sorted_indices_real = np.argsort(-self.jets_real[:, :, pt_idx], axis=1)
            self.jets_real = np.take_along_axis(self.jets_real, sorted_indices_real[:, :, None], axis=1)
            
            # Reorder generated jets (model 1): shape is [n_samples, n_events, nJets, n_gen_features]
            sorted_indices_gen = np.argsort(-self.jets_gen[..., pt_idx], axis=2)
            self.jets_gen = np.take_along_axis(self.jets_gen, sorted_indices_gen[..., None], axis=2)
            
            # Reorder generated jets (model 2), if available:
            if self.jets_gen2 is not None:
                sorted_indices_gen2 = np.argsort(-self.jets_gen2[..., pt_idx], axis=2)
                self.jets_gen2 = np.take_along_axis(self.jets_gen2, sorted_indices_gen2[..., None], axis=2)

    def _to_vector(self, jets):
        # Create a high-level Awkward Array from jets.
        jets_ak = ak.Array(jets)
        return ak.zip({
            "pt": jets_ak[:, :, self.feat_idx_map["pt"]],
            "eta": jets_ak[:, :, self.feat_idx_map["eta"]],
            "phi": jets_ak[:, :, self.feat_idx_map["phi"]],
            "mass": jets_ak[:, :, self.feat_idx_map["mass"]]
        }, with_name="Momentum4D")

    def _to_vector_met(self, met):
        # Convert to a NumPy array from an Awkward Array for ease of inspection.
        met_arr = ak.to_numpy(ak.Array(met))
        if met_arr.shape[-1] == 4:
            # Truth MET: order is (pt, eta, phi, mass)
            pt   = met_arr[:, 0, 0]
            eta  = met_arr[:, 0, 1]
            phi  = met_arr[:, 0, 2]
            mass = met_arr[:, 0, 3]
        elif met_arr.shape[-1] == 2:
            # Generated MET: order is (pt, phi)
            pt   = met_arr[:, 0, 0]
            phi  = met_arr[:, 0, 1]
            eta  = np.zeros_like(pt)
            mass = np.zeros_like(pt)
        else:
            raise ValueError("Unexpected number of MET features.")
        return vector.array({"pt": pt, "eta": eta, "phi": phi, "mass": mass})



    
    def plot_E_j1(self):
        self.compare_observable(lambda jets: jets[:, 0].E, nbins=100,
                                  xlabel=r"$E_{j_1}$ [GeV]", observable_name="E_j1", log_scale=True)
    
    def plot_pT_j1(self):
        self.compare_observable(lambda jets: jets[:, 0].pt, nbins=100,
                                  xlabel=r"$p_{T, j_1}$ [GeV]", observable_name="pT_j1", log_scale=True)
    
    def plot_dphi_j1j2(self):
        self.compare_observable(lambda jets: jets[:, 0].deltaphi(jets[:, 1]), nbins=50,
                                  xlabel=r"$\Delta\phi(j_1,j_2)$ [rad]", observable_name="dphi_j1j2")
    
    def plot_dR_j1j2(self):
        self.compare_observable(lambda jets: jets[:, 0].deltaR(jets[:, 1]), nbins=50,
                                  xlabel=r"$\Delta R(j_1,j_2)$", observable_name="dR_j1j2", log_scale=True)
    
    def plot_HT(self):
        self.compare_observable(lambda jets: ak.sum(jets.pt, axis=1), nbins=50,
                                  xlabel=r"$H_T$ [GeV]", observable_name="HT", log_scale=True)
    
    def plot_dR_met_jj(self):
        """Plot ΔR between MET and the dijet system (j₁+j₂)."""
        def obs(jets, met):
            j1j2 = jets[:,0] + jets[:,1]
            return met.deltaR(j1j2)
        self.compare_observable(obs, nbins=50,
                                            xlabel=r"$\Delta R(\mathrm{MET},jj)$", observable_name="dR_met_jj", log_scale=True)



    def plot_min_mass_jj(self):
        def obs(jets):
            dijets = ak.combinations(jets, 2, replacement=False, axis=1)
            j1, j2 = ak.unzip(dijets)
            return ak.min((j1 + j2).mass, axis=1)
        self.compare_observable(obs, nbins=50,
                                  xlabel=r"$m_{jj}^{\min}$ [GeV]", observable_name="min_mass_jj", log_scale=True)
    

    def compare_observable(self, observable_function, nbins=50, xlabel="Feature", observable_name="Observable", log_scale=False):
        # Real data observable:
        real_vec = self._to_vector(self.jets_real)
        if observable_name == "dR_met_jj":
            real_vec_met = self._to_vector_met(self.met_real)
            real_obs = observable_function(real_vec, real_vec_met)
        else:
            real_obs = observable_function(real_vec)
        real_vals = ak.to_numpy(real_obs).ravel()

        # Generated data observable:
        n_samples = self.jets_gen.shape[0]  # e.g. 100 samples
        n_events = self.jets_gen.shape[1]   # e.g. 18364 events
        gen_flat = self.jets_gen.reshape(n_samples * n_events, *self.jets_gen.shape[2:])
        gen_vec = self._to_vector(gen_flat)
        if observable_name == "dR_met_jj":
            met_gen_flat = self.met_gen.reshape(n_samples * n_events, *self.met_gen.shape[2:])
            gen_vec_met = self._to_vector_met(met_gen_flat)
            gen_obs = observable_function(gen_vec, gen_vec_met)
        else:
            gen_obs = observable_function(gen_vec)
        gen_vals = ak.to_numpy(gen_obs).reshape(-1)


        # Second generated dataset:
        if self.jets_gen2 is not None:
            gen2_flat = self.jets_gen2.reshape(n_samples * n_events, *self.jets_gen2.shape[2:])
            gen_vec2 = self._to_vector(gen2_flat)
            if observable_name == "dR_met_jj":
                met_gen2_flat = self.met_gen2.reshape(n_samples * n_events, *self.met_gen2.shape[2:])
                gen_vec_met2 = self._to_vector_met(met_gen2_flat)
                gen_obs2 = observable_function(gen_vec2, gen_vec_met2)
            else:
                gen_obs2 = observable_function(gen_vec2)
            gen_vals2 = ak.to_numpy(gen_obs2).reshape(-1)

        else:
            gen_vals2 = np.full_like(real_vals, np.nan)

        # Determine common bins:
        min_val = min(real_vals.min(), gen_vals.min(), gen_vals2.min())
        max_val = max(real_vals.max(), gen_vals.max(), gen_vals2.max())
        bins = np.linspace(min_val, max_val, nbins+1)

        # Histograms:
        hist_real, _ = np.histogram(real_vals, bins=bins, density=True)
        hist_gen, _ = np.histogram(gen_vals, bins=bins, density=True)
        hist_gen2, _ = np.histogram(gen_vals2, bins=bins, density=True)
        counts_real, _ = np.histogram(real_vals, bins=bins)
        counts_gen, _ = np.histogram(gen_vals, bins=bins)
        counts_gen2, _ = np.histogram(gen_vals2, bins=bins)
        bin_widths = np.diff(bins)

        total_real = counts_real.sum()
        total_gen = counts_gen.sum()
        total_gen2 = counts_gen2.sum()

        real_errors = np.sqrt(counts_real) / (total_real * bin_widths)
        gen_errors = np.sqrt(counts_gen) / (total_gen * bin_widths) / np.sqrt(n_samples)
        gen_errors2 = np.sqrt(counts_gen2) / (total_gen2 * bin_widths) / np.sqrt(n_samples)

        ratio = np.divide(hist_gen, hist_real, where=hist_real > 0)
        ratio2 = np.divide(hist_gen2, hist_real, where=hist_real > 0)
        ratio_error = np.divide(gen_errors, hist_real, where=hist_real > 0)
        ratio_error2 = np.divide(gen_errors2, hist_real, where=hist_real > 0)

        # Plotting:
        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3,1], 'hspace': 0},
                                sharex=True, figsize=(6,5), dpi=150)
        axs[0].step(bins[:-1], hist_real, where='post', label="Truth", linewidth=1.5, color='#1f77b4')
        axs[0].step(bins[:-1], hist_gen, where='post', label="Gen Model 1", linewidth=1.5, color='#d62728')
        axs[0].step(bins[:-1], hist_gen2, where='post', label="Gen Model 2", linewidth=1.5, color='#2ca02c')
        axs[0].fill_between(bins[:-1], hist_real - real_errors, hist_real + real_errors,
                            step='post', color='#1f77b4', alpha=0.3)
        axs[0].fill_between(bins[:-1], hist_gen - gen_errors, hist_gen + gen_errors,
                            step='post', color='#d62728', alpha=0.3)
        axs[0].fill_between(bins[:-1], hist_gen2 - gen_errors2, hist_gen2 + gen_errors2,
                            step='post', color='#2ca02c', alpha=0.3)
        axs[0].set_ylabel("Density", fontsize=16)
        axs[0].legend(fontsize=10)
        if log_scale:
            axs[0].set_yscale("log")

        axs[1].axhline(1.0, color='black', linestyle='dashed', linewidth=1)
        axs[1].step(bins[:-1], ratio, where='post', linewidth=1.5, color='#d62728', label="Gen 1/Truth")
        axs[1].step(bins[:-1], ratio2, where='post', linewidth=1.5, color='#2ca02c', label="Gen 2/Truth")
        axs[1].fill_between(bins[:-1], ratio - ratio_error, ratio + ratio_error,
                            step='post', color='#d62728', alpha=0.3)
        axs[1].fill_between(bins[:-1], ratio2 - ratio_error2, ratio2 + ratio_error2,
                            step='post', color='#2ca02c', alpha=0.3)
        axs[1].set_xlabel(xlabel, fontsize=16)
        axs[1].set_ylabel(r"Gen/Truth", fontsize=16)

        print(f"Truth values: Min={real_vals.min()}, Max={real_vals.max()}, Size={real_vals.shape}")
        print(f"Gen Model 1 values: Min={gen_vals.min()}, Max={gen_vals.max()}, Size={gen_vals.shape}")
        print(f"Gen Model 2 values: Min={gen_vals2.min()}, Max={gen_vals2.max()}, Size={gen_vals2.shape}")

        plt.tight_layout()
        plt.show()


    def plot_all(self):
        self.plot_E_j1()
        self.plot_pT_j1()
        self.plot_dphi_j1j2()
        self.plot_dR_j1j2()
        self.plot_HT()
        self.plot_dR_met_jj()
        self.plot_min_mass_jj()




def undo_preprocessing(model, real_data, real_fields, real_mask, gen_data, gen_fields, ptype_idx, preprocessing):
    name = model.reco_particle_type_names[ptype_idx]
    real_data = real_data.cpu()
    gen_data = gen_data.cpu()
    if gen_data.ndim == 4:
        gen_data = gen_data.reshape(gen_data.shape[0] * gen_data.shape[1],
                                        gen_data.shape[2], gen_data.shape[3])
    if real_mask is not None:
        real_mask = real_mask.cpu()
    else:
        real_mask = torch.ones(real_data.shape[0], real_data.shape[1], dtype=torch.bool)
    N_sample = gen_data.shape[0] // real_data.shape[0]
    gen_mask = real_mask.repeat((N_sample, 1))
    real_data, _ = preprocessing.inverse(name=name, x=real_data, mask=real_mask, fields=real_fields)
    gen_data, _ = preprocessing.inverse(name=name, x=gen_data, mask=gen_mask, fields=gen_fields)

    return real_data, real_mask, gen_data, gen_mask
