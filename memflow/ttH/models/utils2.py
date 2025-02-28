import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import os

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
    Class for plotting high-level observables such as:
      - Energy of the leading jet (E_j1, computed from pt, eta, mass)
      - Transverse momentum of the leading jet (pT_j1)
      - Δϕ between jet1 and jet2 (|Δϕ_j1,j2|)
      - ΔR between jet1 and jet2 (√((Δη)² + (Δϕ)²))
      - ΔR between MET and the dijet system (ΔR_MET,jj)
      - H_T, the scalar sum of jet pT's
    Optionally, if you have two generated models (e.g. model1 and model2), you can pass both.
    """
    def __init__(self, model, preprocessing, real_data: torch.Tensor, real_mask: torch.Tensor, 
                 gen_data: torch.Tensor, feat_idx_map: dict, 
                 gen_data2: torch.Tensor = None):
        """
        Parameters:
            real_data: Tuple (jets_real, met_real) – truth data tensors.
            real_mask: Tuple (jets_real_mask, met_real_mask) – masks for truth data.
            gen_data: Tuple (jets_gen, met_gen) – generated data tensors.
            feat_idx_map: Dictionary mapping feature names to indices.
            gen_data2: Optional Tuple (jets_gen2, met_gen2) – second generated dataset.
        """
        # Get the feature lists from the model.
        jets_real_fields = model.reco_input_features_per_type[0]
        jets_gen_fields = [jets_real_fields[idx] for idx in model.flow_indices[0]]
        met_real_fields = model.reco_input_features_per_type[1]
        met_gen_fields = [met_real_fields[idx] for idx in model.flow_indices[1]]

        # Undo preprocessing for jets
        jets_real, jets_real_mask, jets_gen, jets_gen_mask = undo_preprocessing(
            model, real_data[0], jets_real_fields, real_mask[0], gen_data[0], jets_gen_fields, 0, preprocessing
        )
        # Undo preprocessing for MET
        met_real, met_real_mask, met_gen, met_gen_mask = undo_preprocessing(
            model, real_data[1], met_real_fields, real_mask[1], gen_data[1], met_gen_fields, 1, preprocessing
        )

        if gen_data2 is not None:
            _, _, jets_gen2, jets_gen_mask2 = undo_preprocessing(
                model, real_data[0], jets_real_fields, real_mask[0], gen_data2[0], jets_gen_fields, 0, preprocessing
            )
            _, _, met_gen2, met_gen_mask2 = undo_preprocessing(
                model, real_data[1], met_real_fields, real_mask[1], gen_data2[1], met_gen_fields, 1, preprocessing
            )
        else:
            jets_gen2, met_gen2, jets_gen_mask2, met_gen_mask2 = None, None, None, None

        self.feat_idx_map = feat_idx_map

        # The undo_preprocessing function returns tensors that are still “flattened” over samples and events.
        # For jets, we expect:
        #   real jets: shape (n_events, nJets, n_features)
        #   gen jets: shape (n_samples*n_events, nJets, n_features)
        # For MET:
        #   real MET: shape (n_events, 1, n_met_features)
        #   gen MET: shape (n_samples*n_events, 1, n_met_features)
        #
        # We now reshape generated data back to (n_samples, n_events, ...)
        n_events = jets_real.shape[0]
        # For jets_gen:
        gen_total = jets_gen.shape[0]  # n_samples * n_events
        n_samples = gen_total // n_events
        jets_gen = jets_gen.reshape(n_samples, n_events, *jets_gen.shape[1:])
        # Also reshape the corresponding mask:
        jets_gen_mask = jets_gen_mask.reshape(n_samples, n_events, *jets_gen_mask.shape[1:])
        # For MET:
        if met_gen is not None:
            met_total = met_gen.shape[0]  # n_samples * n_events
            n_samples_met = met_total // n_events
            # We expect n_samples_met to equal n_samples.
            met_gen = met_gen.reshape(n_samples_met, n_events, *met_gen.shape[1:])
            met_gen_mask = met_gen_mask.reshape(n_samples_met, n_events, *met_gen_mask.shape[1:])

        if jets_gen2 is not None:
            gen_total2 = jets_gen2.shape[0]
            n_samples2 = gen_total2 // n_events
            jets_gen2 = jets_gen2.reshape(n_samples2, n_events, *jets_gen2.shape[1:])
            jets_gen_mask2 = jets_gen_mask2.reshape(n_samples2, n_events, *jets_gen_mask2.shape[1:])
            if met_gen2 is not None:
                met_total2 = met_gen2.shape[0]
                n_samples_met2 = met_total2 // n_events
                met_gen2 = met_gen2.reshape(n_samples_met2, n_events, *met_gen2.shape[1:])
                met_gen_mask2 = met_gen_mask2.reshape(n_samples_met2, n_events, *met_gen_mask2.shape[1:])

        # Convert all to NumPy arrays
        self.jets_real = jets_real.cpu().numpy()
        self.jets_gen = jets_gen.cpu().numpy()  # shape: (n_samples, n_events, nJets, n_features)
        self.met_real = met_real.cpu().numpy() if met_real is not None else None
        self.met_gen = met_gen.cpu().numpy() if met_gen is not None else None
        if jets_gen2 is not None:
            self.jets_gen2 = jets_gen2.cpu().numpy()
            self.met_gen2 = met_gen2.cpu().numpy() if met_gen2 is not None else None

        # --- Now perform sorting of jets per event ---
        pt_idx = self.feat_idx_map["pt"]

        # Sort real jets in descending pT per event.
        sorted_indices_real = np.argsort(-self.jets_real[:, :, pt_idx], axis=1)
        mask_real = jets_real_mask.numpy().astype(bool)  # shape: (n_events, nJets)
        tiled_indices = np.tile(np.arange(self.jets_real.shape[1]), (self.jets_real.shape[0], 1))
        sorted_indices_real[mask_real == 0] = tiled_indices[mask_real == 0]
        self.jets_real = np.take_along_axis(self.jets_real, sorted_indices_real[:, :, None], axis=1)

        # For generated jets (model 1), sort each event by the median pT across samples.
        median_pt_gen = np.median(self.jets_gen[..., pt_idx], axis=0)  # shape: (n_events, nJets)
        sorted_indices_gen = np.argsort(-median_pt_gen, axis=1)  # shape: (n_events, nJets)
        mask_gen = np.any(jets_gen_mask.numpy(), axis=0)  # shape: (n_events, nJets)
        tiled_indices_gen = np.tile(np.arange(self.jets_gen.shape[2]), (self.jets_gen.shape[1], 1))
        sorted_indices_gen[mask_gen == 0] = tiled_indices_gen[mask_gen == 0]
        self.jets_gen = np.take_along_axis(self.jets_gen, sorted_indices_gen[None, :, :, None], axis=2)

        if self.jets_gen2 is not None:
            median_pt_gen2 = np.median(self.jets_gen2[..., pt_idx], axis=0)
            sorted_indices_gen2 = np.argsort(-median_pt_gen2, axis=1)
            mask_gen2 = np.any(jets_gen_mask2.numpy(), axis=0)
            tiled_indices_gen2 = np.tile(np.arange(self.jets_gen2.shape[2]), (self.jets_gen2.shape[1], 1))
            sorted_indices_gen2[mask_gen2 == 0] = tiled_indices_gen2[mask_gen2 == 0]
            self.jets_gen2 = np.take_along_axis(self.jets_gen2, sorted_indices_gen2[None, :, :, None], axis=2)

        print("Shapes after preprocessing and sorting:")
        print(f"  jets_real: {self.jets_real.shape}")
        print(f"  jets_gen: {self.jets_gen.shape}")
        if self.jets_gen2 is not None:
            print(f"  jets_gen2: {self.jets_gen2.shape}")

    def _compute_energy(self, pt, eta, mass):
        """Computes energy using E^2 = p_T^2 + m^2 cosh^2(η)."""
        return np.sqrt(pt**2 + (mass * np.cosh(eta))**2)

    def _plot_distribution_comparison(self, real_vals, gen_vals1, gen_vals2, observable_name, xlabel, 
                                        nbins=50, log_scale=False, model1_label="Model 1", model2_label="Model 2"):
        """
        Helper to plot a two-panel figure:
         - Top panel: Step histograms (with Poisson error bands) for truth, gen model1, and gen model2.
         - Bottom panel: Ratio of each generated model to truth.
        All arrays are assumed to be 1D.
        """
        # Determine common bins across all data
        bins = np.linspace(min(real_vals.min(), gen_vals1.min(), gen_vals2.min()),
                           max(real_vals.max(), gen_vals1.max(), gen_vals2.max()),
                           nbins+1)
        # Compute density-normalized histograms
        hist_real, _ = np.histogram(real_vals, bins=bins, density=True)
        hist_gen1, _ = np.histogram(gen_vals1, bins=bins, density=True)
        hist_gen2, _ = np.histogram(gen_vals2, bins=bins, density=True)
        # Also compute raw counts for Poisson errors
        N_sample_1 = gen_vals1.shape[0]
        N_sample_2 = gen_vals2.shape[0]
        counts_real, _ = np.histogram(real_vals, bins=bins)
        counts_gen1, _ = np.histogram(gen_vals1, bins=bins)
        counts_gen2, _ = np.histogram(gen_vals2, bins=bins)
        bin_widths = np.diff(bins)
        total_real = np.sum(counts_real)
        total_gen1 = np.sum(counts_gen1)
        total_gen2 = np.sum(counts_gen2)
        error_real = np.sqrt(counts_real) / (total_real * bin_widths)
        error_gen1 = np.sqrt(counts_gen1) / (total_gen1 * bin_widths) * np.sqrt(N_sample_1)
        error_gen2 = np.sqrt(counts_gen2) / (total_gen2 * bin_widths) * np.sqrt(N_sample_2)
        # Ratio: generated/truth (avoid division by zero)
        ratio1 = np.divide(hist_gen1, hist_real, where=hist_real>0)
        ratio2 = np.divide(hist_gen2, hist_real, where=hist_real>0)
        ratio_error1 = np.divide(error_gen1, hist_real, where=hist_real>0)
        ratio_error2 = np.divide(error_gen2, hist_real, where=hist_real>0)

        # Create figure with two subplots: top for histograms, bottom for ratio
        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0},
                                sharex=True, figsize=(6,5), dpi=150)
        # Top panel: step histograms with error bands
        axs[0].step(bins[:-1], hist_real, where='post', label="Truth", linewidth=1.5, color='#1f77b4')
        axs[0].fill_between(bins[:-1], hist_real - error_real, hist_real + error_real, step='post', color='#1f77b4', alpha=0.3)
        axs[0].step(bins[:-1], hist_gen1, where='post', label=model1_label, linewidth=1.5, color='#d62728')
        axs[0].fill_between(bins[:-1], hist_gen1 - error_gen1, hist_gen1 + error_gen1, step='post', color='#d62728', alpha=0.3)
        axs[0].step(bins[:-1], hist_gen2, where='post', label=model2_label, linewidth=1.5, color='#2ca02c')
        axs[0].fill_between(bins[:-1], hist_gen2 - error_gen2, hist_gen2 + error_gen2, step='post', color='#2ca02c', alpha=0.3)
        axs[0].set_ylabel("Density", fontsize=16)
        axs[0].legend(fontsize=12)
        if log_scale:
            axs[0].set_yscale("log")
        # Bottom panel: Ratio plots
        axs[1].axhline(1.0, color='black', linestyle='dashed', linewidth=1)
        axs[1].step(bins[:-1], ratio1, where='post', linewidth=1.5, color='#d62728')
        axs[1].fill_between(bins[:-1], ratio1 - ratio_error1, ratio1 + ratio_error1, step='post', color='#d62728', alpha=0.3)
        axs[1].step(bins[:-1], ratio2, where='post', linewidth=1.5, color='#2ca02c')
        axs[1].fill_between(bins[:-1], ratio2 - ratio_error2, ratio2 + ratio_error2, step='post', color='#2ca02c', alpha=0.3)
        axs[1].set_xlabel(xlabel, fontsize=16)
        axs[1].set_ylabel(r"$\frac{\text{Gen}}{\text{Truth}}$", fontsize=16)

        if observable_name == "E_j1":
            axs[1].set_xlim(0, 2000)
            axs[0].set_ylim(1e-8, 1e-2)
            axs[1].set_ylim(0.5, 1.5)
        elif observable_name == "pT_j1":
            axs[1].set_xlim(0, 2000)
            #axs[0].set_ylim(1e-8, 1e-2)
            axs[1].set_ylim(0.5, 1.5)
        elif observable_name =="dphi_j1,j2":
            axs[1].set_xlim(-math.pi, math.pi)
            axs[1].set_ylim(0.5, 1.5)
        elif observable_name =="dR_j1,j2":
            axs[1].set_xlim(0, 6)
            axs[0].set_ylim(4e-4, 1e0)
            axs[1].set_ylim(0.5,1.5)
        elif observable_name =="dR_MET,jj":
            axs[1].set_xlim(0, 4)
            axs[0].set_ylim(1e-5, 1e1)
            axs[1].set_ylim(0.5,1.5)
        elif observable_name =="Min(m_j1j2)":
            axs[1].set_xlim(0, 200)
            axs[0].set_ylim(1e-6, 1e0)
            axs[1].set_ylim(0.5,1.5)
        elif observable_name =="H_T":
            axs[1].set_xlim(0, 5000)
            #axs[0].set_ylim(1e-8, 1e-2)
            axs[1].set_ylim(0.5,1.5)


        plt.tight_layout()
        plt.show()

    # For convenience, if no second model is provided, call the above with gen_vals2 = gen_vals1.
    def _plot_distribution_comparison_single(self, real_vals, gen_vals, observable_name, xlabel, nbins=50, log_scale=False, model_label="Model"):
        self._plot_distribution_comparison(real_vals, gen_vals, gen_vals, observable_name, xlabel, nbins, log_scale, model_label, model_label)

    # Now update each plot function to use the new helper if a second model is provided.
    def plot_E_j1(self, nbins=100):
        """Plot the Energy of the leading jet (computed)."""
        pt_idx, eta_idx, mass_idx = self.feat_idx_map["pt"], self.feat_idx_map["eta"], self.feat_idx_map["mass"]
        # Real data
        pt_real = self.jets_real[:, 0, pt_idx]
        eta_real = self.jets_real[:, 0, eta_idx]
        mass_real = self.jets_real[:, 0, mass_idx]
        E_real = self._compute_energy(pt_real, eta_real, mass_real)
        # Model 1
        pt_gen = self.jets_gen[:, :, 0, pt_idx]
        eta_gen = self.jets_gen[:, :, 0, eta_idx]
        mass_gen = self.jets_gen[:, :, 0, mass_idx]
        E_gen = self._compute_energy(pt_gen, eta_gen, mass_gen)
        # If second model is provided
        if self.jets_gen2 is not None:
            pt_gen2 = self.jets_gen2[:, :, 0, pt_idx]
            eta_gen2 = self.jets_gen2[:, :, 0, eta_idx]
            mass_gen2 = self.jets_gen2[:, :, 0, mass_idx]
            E_gen2 = self._compute_energy(pt_gen2, eta_gen2, mass_gen2)
            self._plot_distribution_comparison(E_real, E_gen, E_gen2, "E_j1", r"$E_{j_1}$ [GeV]", log_scale=True, nbins=nbins)
        else:
            self._plot_distribution_comparison_single(E_real, E_gen, "E_j1", r"$E_{j_1}$ [GeV]", log_scale=True, model_label="Model", nbins=nbins)

    def plot_pT_j1(self, nbins=100):
        """Plot pT of the leading jet."""
        pt_idx = self.feat_idx_map["pt"]
        real_vals = self.jets_real[:, 0, pt_idx]
        gen_vals = self.jets_gen[:, :, 0, pt_idx]
        if self.jets_gen2 is not None:
            gen_vals2 = self.jets_gen2[:, :, 0, pt_idx]
            self._plot_distribution_comparison(real_vals, gen_vals, gen_vals2, "pT_j1", r"$p_{T,j_1}$ [GeV]", log_scale=True, nbins=nbins)
        else:
            self._plot_distribution_comparison_single(real_vals, gen_vals, "pT_j1", r"$p_{T,j_1}$ [GeV]", log_scale=True, model_label="Model", nbins=nbins)

    def plot_dphi_j1j2(self, nbins=50):
        """Plot Δϕ between jet1 and jet2, with proper circular wrapping."""
        phi_idx = self.feat_idx_map["phi"]
        delta_phi_real = self.jets_real[:, 0, phi_idx] - self.jets_real[:, 1, phi_idx]
        delta_phi_real = (delta_phi_real + np.pi) % (2 * np.pi) - np.pi
        delta_phi_gen = self.jets_gen[:, :, 0, phi_idx] - self.jets_gen[:, :, 1, phi_idx]
        delta_phi_gen = (delta_phi_gen + np.pi) % (2 * np.pi) - np.pi
        if self.jets_gen2 is not None:
            delta_phi_gen2 = self.jets_gen2[:, :, 0, phi_idx] - self.jets_gen2[:, :, 1, phi_idx]
            delta_phi_gen2 = (delta_phi_gen2 + np.pi) % (2 * np.pi) - np.pi
            self._plot_distribution_comparison(delta_phi_real, delta_phi_gen, delta_phi_gen2, "dphi_j1,j2", r"$\Delta\phi_{j_1,j_2}$ [rad]", log_scale=False, nbins=nbins)
        else:
            self._plot_distribution_comparison_single(delta_phi_real, delta_phi_gen, "dphi_j1,j2", r"$\Delta\phi_{j_1,j_2}$ [rad]", log_scale=False, model_label="Model", nbins=nbins)

    def plot_dR_j1j2(self, nbins=100):
        """Plot ΔR between jet1 and jet2."""
        eta_idx, phi_idx = self.feat_idx_map["eta"], self.feat_idx_map["phi"]
        delta_eta_real = self.jets_real[:, 0, eta_idx] - self.jets_real[:, 1, eta_idx]
        delta_phi_real = (self.jets_real[:, 0, phi_idx] - self.jets_real[:, 1, phi_idx] + np.pi) % (2 * np.pi) - np.pi
        delta_R_real = np.sqrt(delta_eta_real**2 + delta_phi_real**2)
        delta_eta_gen = self.jets_gen[:, :, 0, eta_idx] - self.jets_gen[:, :, 1, eta_idx]
        delta_phi_gen = (self.jets_gen[:, :, 0, phi_idx] - self.jets_gen[:, :, 1, phi_idx] + np.pi) % (2 * np.pi) - np.pi
        delta_R_gen = np.sqrt(delta_eta_gen**2 + delta_phi_gen**2)
        if self.jets_gen2 is not None:
            delta_eta_gen2 = self.jets_gen2[:, :, 0, eta_idx] - self.jets_gen2[:, :, 1, eta_idx]
            delta_phi_gen2 = (self.jets_gen2[:, :, 0, phi_idx] - self.jets_gen2[:, :, 1, phi_idx] + np.pi) % (2 * np.pi) - np.pi
            delta_R_gen2 = np.sqrt(delta_eta_gen2**2 + delta_phi_gen2**2)
            self._plot_distribution_comparison(delta_R_real, delta_R_gen, delta_R_gen2, "dR_j1,j2", r"$\Delta R_{j_1,j_2}$", log_scale=True, nbins=nbins)
        else:
            self._plot_distribution_comparison_single(delta_R_real, delta_R_gen, "dR_j1,j2", r"$\Delta R_{j_1,j_2}$", log_scale=True, model_label="Model", nbins=nbins)
    
    def plot_dR_MET_jj(self, nbins=100):
        """Plot ΔR between MET and the dijet system (ΔR_MET,jj)."""
        if self.met_real is None or self.met_gen is None:
            print("MET data not provided.")
            return
        eta_idx = self.feat_idx_map["eta"]
        phi_idx = self.feat_idx_map["phi"]

        # Truth MET: assume shape (n_events, 1, 2)
        phi_MET_real = self.met_real[:, 0, 1]
        eta_MET_real = np.zeros_like(phi_MET_real)
        eta_jj_real = 0.5 * (self.jets_real[:, 0, eta_idx] + self.jets_real[:, 1, eta_idx])
        phi_jj_real = 0.5 * (self.jets_real[:, 0, phi_idx] + self.jets_real[:, 1, phi_idx])
        delta_eta_real = eta_MET_real - eta_jj_real
        delta_phi_real = (phi_MET_real - phi_jj_real + np.pi) % (2 * np.pi) - np.pi
        dR_real = np.sqrt(delta_eta_real**2 + delta_phi_real**2)

        # Generated MET (model 1)
        if self.met_gen.ndim == 4:
            phi_MET_gen = self.met_gen[:, :, 0, 1]
        else:
            phi_MET_gen = self.met_gen[:, :, 1]
        eta_MET_gen = np.zeros_like(phi_MET_gen)
        eta_jj_gen = 0.5 * (self.jets_gen[:, :, 0, eta_idx] + self.jets_gen[:, :, 1, eta_idx])
        phi_jj_gen = 0.5 * (self.jets_gen[:, :, 0, phi_idx] + self.jets_gen[:, :, 1, phi_idx])
        delta_eta_gen = eta_MET_gen - eta_jj_gen
        delta_phi_gen = (phi_MET_gen - phi_jj_gen + np.pi) % (2 * np.pi) - np.pi
        dR_gen = np.sqrt(delta_eta_gen**2 + delta_phi_gen**2)

        # Generated MET (model 2)
        if self.met_gen2 is not None:
            if self.met_gen2.ndim == 4:
                phi_MET_gen2 = self.met_gen2[:, :, 0, 1]
            elif self.met_gen2.ndim == 3:
                phi_MET_gen2 = self.met_gen2[:, :, 1]
            else:
                raise ValueError("Unexpected met_gen2 dimensions.")
            eta_MET_gen2 = np.zeros_like(phi_MET_gen2)
            eta_jj_gen2 = 0.5 * (self.jets_gen2[:, :, 0, eta_idx] + self.jets_gen2[:, :, 1, eta_idx])
            phi_jj_gen2 = 0.5 * (self.jets_gen2[:, :, 0, phi_idx] + self.jets_gen2[:, :, 1, phi_idx])
            delta_eta_gen2 = eta_MET_gen2 - eta_jj_gen2
            delta_phi_gen2 = (phi_MET_gen2 - phi_jj_gen2 + np.pi) % (2 * np.pi) - np.pi
            dR_gen2 = np.sqrt(delta_eta_gen2**2 + delta_phi_gen2**2)
            self._plot_distribution_comparison(dR_real, dR_gen, dR_gen2, "dR_MET,jj", r"$\Delta R_{\text{MET},jj}$", log_scale=True, nbins=nbins)
        else:
            self._plot_distribution_comparison_single(dR_real, dR_gen, "dR_MET,jj", r"$\Delta R_{\text{MET},jj}$", log_scale=True, model_label="Model", nbins=nbins)

    def plot_min_m_jj(self, nbins=100):
        """Plot the minimum invariant mass m_j1j2 among all jet pairs."""
        pt_idx, eta_idx, phi_idx = self.feat_idx_map["pt"], self.feat_idx_map["eta"], self.feat_idx_map["phi"]
        def compute_min_mjj(jets):
            N = jets.shape[0]
            nJets = jets.shape[1]
            if nJets < 2:
                return np.zeros(N)
            min_mjj = np.full(N, np.inf)
            for i in range(nJets):
                for j in range(i+1, nJets):
                    pt1, eta1, phi1 = jets[:, i, pt_idx], jets[:, i, eta_idx], jets[:, i, phi_idx]
                    pt2, eta2, phi2 = jets[:, j, pt_idx], jets[:, j, eta_idx], jets[:, j, phi_idx]
                    delta_eta = eta1 - eta2
                    delta_phi = (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi
                    mjj = np.sqrt(2 * pt1 * pt2 * (np.cosh(delta_eta) - np.cos(delta_phi)))
                    min_mjj = np.minimum(min_mjj, mjj)
            return min_mjj
        min_mjj_real = compute_min_mjj(self.jets_real)
        min_mjj_gen = np.array([compute_min_mjj(self.jets_gen[i]) for i in range(self.jets_gen.shape[0])])
        if self.jets_gen2 is not None:
            min_mjj_gen2 = np.array([compute_min_mjj(self.jets_gen2[i]) for i in range(self.jets_gen2.shape[0])])
            self._plot_distribution_comparison(min_mjj_real, min_mjj_gen, min_mjj_gen2, "Min(m_j1j2)", r"$m^{\text{min}}_{j_1,j_2\in \text{jets}}$ [GeV]", log_scale=True, nbins=nbins)
        else:
            self._plot_distribution_comparison_single(min_mjj_real, min_mjj_gen, "Min(m_j1j2)", r"$m^{\text{min}}_{j_1,j_2\in \text{jets}}$ [GeV]", log_scale=True, model_label="Model", nbins=nbins)

    def plot_H_T(self, nbins=100):
        """Plot H_T, the sum of jet transverse momenta."""
        pt_idx = self.feat_idx_map["pt"]
        real_vals = np.sum(self.jets_real[:, :, pt_idx], axis=1)
        gen_vals = np.sum(self.jets_gen[:, :, :, pt_idx], axis=2)
        if self.jets_gen2 is not None:
            gen_vals2 = np.sum(self.jets_gen2[:, :, :, pt_idx], axis=2)
            self._plot_distribution_comparison(real_vals, gen_vals, gen_vals2, "H_T", r"$H_T$ [GeV]", log_scale=True, nbins=nbins)
        else:
            self._plot_distribution_comparison_single(real_vals, gen_vals, "H_T", r"$H_T$ [GeV]", log_scale=True, model_label="Model", nbins=nbins)

    def plot_all(self):
        """Convenience method to plot all high-level distributions."""
        self.plot_E_j1()
        self.plot_pT_j1()
        self.plot_dphi_j1j2()
        self.plot_dR_j1j2()
        self.plot_H_T()
        self.plot_dR_MET_jj()
        self.plot_min_m_jj()


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
