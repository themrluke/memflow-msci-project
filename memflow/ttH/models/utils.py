# utils.py

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import os

class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))


def compare_distributions(model, real_data, gen_data, ptype_idx,
                                     feat_idx=0,
                                     nbins=50, feat_name="Feature",
                                     preprocessing=None,
                                     real_mask=None,
                                     log_scale=False):
    """
    Compare histograms of real vs. generated data for a single feature with a ratio subplot.

    Parameters
    ----------
    model : object
        Must have:
          - reco_particle_type_names: list of str
          - reco_input_features_per_type: list (per particle type) of list of str
          - flow_indices: list (per particle type) of list of int
    real_data, gen_data : torch.Tensor
        Real data should have shape (B, nParticles, nFeatures).
        gen_data may have an extra sample dimension, e.g. (N_sample, B, nParticles, nFeatures).
    ptype_idx : int
        Particle type index (to select the proper field names).
    feat_idx : int
        The feature indices (in the real and generated data) to compare.
    nbins : int, optional
        Number of histogram bins.
    feat_name : str, optional
        Label for the x-axis (e.g. r"$p_T$").
    preprocessing : object, optional
        A preprocessing object with an inverse() method.
    real_mask : torch.Tensor, optional
        A mask for the real data; if provided, it is used in the inverse() call.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    """

    real_data = real_data[ptype_idx]
    real_mask = real_mask[ptype_idx]
    gen_data = gen_data[ptype_idx]

    # Retrieve full field names from the model.
    real_fields = model.reco_input_features_per_type[ptype_idx]  # e.g. ["pt", "eta", "phi", "E"]
    # For generated data, pick the corresponding fields using flow indices.
    gen_fields = [real_fields[idx] for idx in model.flow_indices[ptype_idx]]

    if preprocessing is not None:
        name = model.reco_particle_type_names[ptype_idx]

        # Move data to CPU.
        real_data = real_data.cpu()
        gen_data = gen_data.cpu()
        # If gen_data has 4 dimensions (e.g. [N_sample, B, nParticles, nFeatures]),
        # reshape it to (N_sample * B, nParticles, nFeatures)
        if gen_data.ndim == 4:
            gen_data = gen_data.reshape(gen_data.shape[0] * gen_data.shape[1],
                                        gen_data.shape[2],
                                        gen_data.shape[3])
        # For the real data, use the provided mask or create a dummy one.
        if real_mask is not None:
            real_mask = real_mask.cpu()
        else:
            real_mask = torch.ones(real_data.shape[0], real_data.shape[1], dtype=torch.bool)

        # Extract N_sample from the original shape of gen_data
        N_sample = gen_data.shape[0] // real_data.shape[0]  # Number of samples per event

        # Repeat real_mask along the first axis to align with gen_data
        gen_mask = real_mask.repeat((N_sample, 1))  # Shape: (N_sample * B, nParticles)

        # Print the mask itself for debugging
        print("gen_mask shape:", gen_mask.shape)
        print("real_mask shape:", real_mask.shape)

        # Print the mask itself
        print("gen_mask:", gen_mask.shape)
        print('real mask: ', real_mask.shape)

        # Calculate and print the percentage of masked jets
        masked_percentage = 100 * (gen_mask == False).sum().item() / gen_mask.numel()
        print(f"Percentage of masked jets: {masked_percentage:.2f}%")

        # Calculate and print the percentage of masked jets
        real_masked_percentage = 100 * (real_mask == False).sum().item() / real_mask.numel()
        print(f"Percentage of real masked jets: {real_masked_percentage:.2f}%")

        print('real data shape before inverse preprocessing', real_data.shape)
        print('gen data shape before inverse preprocessing', gen_data.shape)

        # Apply inverse preprocessing.
        real_data, _ = preprocessing.inverse(
            name=name,
            x=real_data,
            mask=real_mask,
            fields=real_fields
        )
        gen_data, _ = preprocessing.inverse(
            name=name,
            x=gen_data,
            mask=gen_mask,
            fields=gen_fields
        )

    # --- Extract and flatten the feature values ---
    real_data = real_data[real_mask.bool()]
    gen_data = gen_data[gen_mask.bool()]

    if ptype_idx == 1 and feat_idx == 1: # For MET, phi is at different element in sample data
        real_vals = real_data[..., feat_idx+1].cpu().numpy().ravel()
    else:
        real_vals = real_data[..., feat_idx].cpu().numpy().ravel()
    gen_vals  = gen_data[..., feat_idx].cpu().numpy().ravel()

    # --- Compute histogram bins ---
    bins = np.linspace(min(real_vals.min(), gen_vals.min()),
                       max(real_vals.max(), gen_vals.max()),
                       nbins + 1)

    # --- Compute density-normalized histograms ---
    hist_real, _ = np.histogram(real_vals, bins=bins, density=True)
    hist_gen, _ = np.histogram(gen_vals, bins=bins, density=True)

    # --- Compute Poisson uncertainties (scaled for density) ---
    real_counts, _ = np.histogram(real_vals, bins=bins)
    gen_counts, _ = np.histogram(gen_vals, bins=bins)
    bin_widths = np.diff(bins)
    total_real = np.sum(real_counts)
    total_gen = np.sum(gen_counts)
    real_errors = np.sqrt(real_counts) / (total_real * bin_widths)
    gen_errors = (np.sqrt(gen_counts) / (total_gen * bin_widths)) * np.sqrt(gen_data.shape[0] // real_data.shape[0])

    # --- Compute the ratio (Gen/Real) and propagate uncertainties ---
    ratio = np.divide(hist_gen, hist_real, where=hist_real > 0)
    real_uncertainty = real_errors / hist_real

    # --- Create the figure with two subplots ---
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0},
                            sharex=True, figsize=(6, 5), dpi=300)
    plt.subplots_adjust(hspace=0)  # Ensures no gap between plots

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

    # Enable logarithmic scale if specified
    if log_scale:
        axs[0].set_yscale("log")
        if ptype_idx == 0: # Jets
            if feat_idx == 0: # pT
                axs[0].set_xlim(30, 1500)
                axs[0].set_ylim(2e-8,1e-2)
                axs[1].set_ylim(0.5, 1.5)
            elif feat_idx == 1: # eta
                axs[0].set_xlim(-5, 5)
                axs[0].set_ylim(3e-4,1e0)
                axs[1].set_ylim(0.8, 1.2)
        if ptype_idx == 1: # MET
            if feat_idx == 0: #pT
                axs[0].set_xlim(200, 1200)
                axs[0].set_ylim(3e-7,1e-2)
                axs[1].set_ylim(0.5, 1.5)
    else:
        axs[0].ticklabel_format(style='sci', axis='y', scilimits=(-1,1))

    if ptype_idx == 0 and feat_idx == 2: # Jets phi
            axs[0].set_xlim(-math.pi, math.pi)
            #axs[1].set_ylim(0.95, 1.05)
    if ptype_idx == 1 and feat_idx == 1: # MET phi
            axs[0].set_xlim(-math.pi, math.pi)
            #axs[1].set_ylim(0.95, 1.05)

    plt.tight_layout()
    plt.show()

    return None


def compare_distributions_multiple(model, real_data, gen_data_1, gen_data_2, ptype_idx,
                                   feat_idx=0, nbins=50, feat_name="Feature",
                                   preprocessing=None, real_mask=None, log_scale=False):
    """
    Compare histograms of real vs. two generated datasets for a single feature with a ratio subplot.

    Parameters
    ----------
    model : object
        Must have:
          - reco_particle_type_names: list of str
          - reco_input_features_per_type: list (per particle type) of list of str
          - flow_indices: list (per particle type) of list of int
    real_data, gen_data_1, gen_data_2 : torch.Tensor
        Real data should have shape (B, nParticles, nFeatures).
        Generated data may have an extra sample dimension, e.g. (N_sample, B, nParticles, nFeatures).
    ptype_idx : int
        Particle type index (to select the proper field names).
    feat_idx : int
        The feature indices (in the real and generated data) to compare.
    nbins : int, optional
        Number of histogram bins.
    feat_name : str, optional
        Label for the x-axis.
    preprocessing : object, optional
        A preprocessing object with an inverse() method.
    real_mask : torch.Tensor, optional
        A mask for the real data; if provided, it is used in the inverse() call.
    log_scale : bool, optional
        Whether to use logarithmic y-scale.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    """

    real_data = real_data[ptype_idx]
    real_mask = real_mask[ptype_idx]
    gen_data_1 = gen_data_1[ptype_idx]
    gen_data_2 = gen_data_2[ptype_idx]

    # Retrieve full field names from the model.
    real_fields = model.reco_input_features_per_type[ptype_idx]
    gen_fields = [real_fields[idx] for idx in model.flow_indices[ptype_idx]]

    if preprocessing is not None:
        name = model.reco_particle_type_names[ptype_idx]

        # Move data to CPU.
        real_data = real_data.cpu()
        gen_data_1 = gen_data_1.cpu()
        gen_data_2 = gen_data_2.cpu()

        # Reshape generated data if it has extra sampling dimension
        if gen_data_1.ndim == 4:
            gen_data_1 = gen_data_1.reshape(gen_data_1.shape[0] * gen_data_1.shape[1], 
                                            gen_data_1.shape[2], gen_data_1.shape[3])
        if gen_data_2.ndim == 4:
            gen_data_2 = gen_data_2.reshape(gen_data_2.shape[0] * gen_data_2.shape[1], 
                                            gen_data_2.shape[2], gen_data_2.shape[3])

        # Ensure real mask is on CPU
        if real_mask is not None:
            real_mask = real_mask.cpu()
        else:
            real_mask = torch.ones(real_data.shape[0], real_data.shape[1], dtype=torch.bool)

        # Extract number of samples per event for generated data
        N_sample_1 = gen_data_1.shape[0] // real_data.shape[0]
        N_sample_2 = gen_data_2.shape[0] // real_data.shape[0]

        # Repeat real_mask along the first axis to align with generated data
        gen_mask_1 = real_mask.repeat((N_sample_1, 1))  # Shape: (N_sample_1 * B, nParticles)
        gen_mask_2 = real_mask.repeat((N_sample_2, 1))  # Shape: (N_sample_2 * B, nParticles)

        # Apply inverse preprocessing.
        real_data, _ = preprocessing.inverse(name=name, x=real_data, mask=real_mask, fields=real_fields)
        gen_data_1, _ = preprocessing.inverse(name=name, x=gen_data_1, mask=gen_mask_1, fields=gen_fields)
        gen_data_2, _ = preprocessing.inverse(name=name, x=gen_data_2, mask=gen_mask_2, fields=gen_fields)

    # --- Extract and flatten the feature values ---
    real_data = real_data[real_mask.bool()]
    gen_data_1 = gen_data_1[gen_mask_1.bool()]
    gen_data_2 = gen_data_2[gen_mask_2.bool()]

    if ptype_idx == 1 and feat_idx == 1: # For MET, phi is at different element in sample data
        real_vals = real_data[..., feat_idx+1].cpu().numpy().ravel()
    else:
        real_vals = real_data[..., feat_idx].cpu().numpy().ravel()

    gen_vals_1 = gen_data_1[..., feat_idx].cpu().numpy().ravel()
    gen_vals_2 = gen_data_2[..., feat_idx].cpu().numpy().ravel()

    # --- Compute histogram bins ---
    bins = np.linspace(min(real_vals.min(), gen_vals_1.min(), gen_vals_2.min()),
                       max(real_vals.max(), gen_vals_1.max(), gen_vals_2.max()),
                       nbins + 1)

    # --- Compute density-normalized histograms ---
    hist_real, _ = np.histogram(real_vals, bins=bins, density=True)
    hist_gen_1, _ = np.histogram(gen_vals_1, bins=bins, density=True)
    hist_gen_2, _ = np.histogram(gen_vals_2, bins=bins, density=True)

    # --- Compute Poisson uncertainties for generated data ---
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

    # --- Compute the ratio (Gen/Real) ---
    ratio_1 = np.divide(hist_gen_1, hist_real, where=hist_real > 0)
    ratio_2 = np.divide(hist_gen_2, hist_real, where=hist_real > 0)

    # --- Compute the ratio (Gen/Real) and propagate uncertainties ---
    ratio_1 = np.divide(hist_gen_1, hist_real, where=hist_real > 0)
    ratio_2 = np.divide(hist_gen_2, hist_real, where=hist_real > 0)

    # Compute ratio uncertainty
    ratio_error_1 = np.divide(gen_errors_1, hist_real, where=hist_real > 0)
    ratio_error_2 = np.divide(gen_errors_2, hist_real, where=hist_real > 0)

    # --- Create figure with two subplots ---
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0}, 
                            sharex=True, figsize=(6, 5), dpi=300)
    plt.subplots_adjust(hspace=0)

    axs[0].step(bins[:-1], hist_real, where="post", label="Truth", linewidth=1.5, color='#1f77b4')
    axs[0].step(bins[:-1], hist_gen_1, where="post", label="Transfermer", linewidth=1.5, color='#d62728')
    axs[0].step(bins[:-1], hist_gen_2, where="post", label="Parallel Transfusion", linewidth=1.5, color='#2ca02c')
    axs[0].fill_between(bins[:-1], hist_real - real_errors, hist_real + real_errors,
                        step="post", color='#1f77b4', alpha=0.3)
    axs[0].fill_between(bins[:-1], hist_gen_1 - gen_errors_1, hist_gen_1 + gen_errors_1,
                        step="post", color='#d62728', alpha=0.3)
    axs[0].fill_between(bins[:-1], hist_gen_2 - gen_errors_2, hist_gen_2 + gen_errors_2,
                        step="post", color='#2ca02c', alpha=0.3)

    axs[0].set_ylabel("Density", fontsize=22)
    axs[0].legend(fontsize=14)

    axs[1].axhline(1.0, color='black', linestyle='dashed', linewidth=1)
    axs[1].step(bins[:-1], ratio_1, where="post", color='#d62728', linewidth=1.5, label="Gen 1 / Truth")
    axs[1].step(bins[:-1], ratio_2, where="post", color='#2ca02c', linewidth=1.5, label="Gen 2 / Truth")
    axs[1].fill_between(bins[:-1], ratio_1 - ratio_error_1, ratio_1 + ratio_error_1,
                        step="post", color='#d62728', alpha=0.3)
    axs[1].fill_between(bins[:-1], ratio_2 - ratio_error_2, ratio_2 + ratio_error_2,
                        step="post", color='#2ca02c', alpha=0.3)
    axs[1].set_ylabel(r"$\frac{\text{Gen}}{\text{Truth}}$", fontsize=22)
    axs[1].set_xlabel(feat_name, fontsize=22)

    axs[0].tick_params(axis='y', which='major', labelsize=14)
    axs[1].tick_params(axis='both', which='major', labelsize=14)
    axs[0].tick_params(axis='y', which='minor', labelsize=14)
    axs[1].tick_params(axis='both', which='minor', labelsize=14)

    # Enable logarithmic scale if specified
    if log_scale:
        axs[0].set_yscale("log")
        if ptype_idx == 0: # Jets
            if feat_idx == 0: # pT
                axs[0].set_xlim(30, 1500)
                axs[0].set_ylim(2e-8,1e-2)
                axs[1].set_ylim(0.5, 1.5)
            elif feat_idx == 1: # eta
                axs[0].set_xlim(-5, 5)
                axs[0].set_ylim(3e-4,1e0)
                axs[1].set_ylim(0.8, 1.2)
            elif feat_idx == 3: # Mass
                axs[0].set_xlim(0, 160)
                axs[0].set_ylim(3e-7,1e-1)
                axs[1].set_ylim(0.5, 1.5)
        if ptype_idx == 1: # MET
            if feat_idx == 0: #pT
                axs[0].set_xlim(200, 1200)
                axs[0].set_ylim(3e-7,1e-2)
                axs[1].set_ylim(0.5, 1.5)
    else:
        axs[0].ticklabel_format(style='sci', axis='y', scilimits=(-1,1))

    if ptype_idx == 0 and feat_idx == 2: # Jets phi
            axs[0].set_xlim(-math.pi, math.pi)
            #axs[1].set_ylim(0.95, 1.05)
    if ptype_idx == 1 and feat_idx == 1: # MET phi
            axs[0].set_xlim(-math.pi, math.pi)
            #axs[1].set_ylim(0.95, 1.05)

    plt.tight_layout()
    plt.show()

    return None



def plot_sampling_distributions(real_data, gen_data_samples, feat_names, event_idx=0, object_name="jets"):
    """
    Compare real data vs. multiple generated samples for a specific event with 2D heatmaps.

    Args:
        real_data: (B, nParticles, nFeatures) for jets, or (B, 1, nFeatures) for MET.
        gen_data_samples: List of tensors [(B, nParticles, nFeatures)] for jets,
                          or [(B, 1, nFeatures)] for MET.
        feat_names: List of feature names (e.g., ["pt", "eta", "phi"]).
        event_idx: Which event in the batch to visualize.
        object_name: Name of the object type ("jets" or "met").

    Returns:
        None. Displays plots.
    """
    is_met = object_name == "met"
    gen_data = torch.stack(gen_data_samples, dim=0)  # shape (nSamples, B, nParticles, nFeatures)

    # Define feature pairs for 2D plotting
    if is_met:
        feature_pairs = [("pt", "phi")]  # MET: only pt vs phi
    else:
        feature_pairs = [("pt", "phi"), ("pt", "eta"), ("eta", "phi")]  # Jets: 3 pairs

    # Map feature names to indices
    feat_idx_map = {name: idx for idx, name in enumerate(feat_names)}

    # MET: No looping over particles
    if is_met:
        plt.figure(figsize=(8, 6))
        for pair_idx, (feat_x, feat_y) in enumerate(feature_pairs):
            idx_x = feat_idx_map[feat_x]
            idx_y = feat_idx_map[feat_y]

            # Extract real values for MET
            real_values_x = real_data[event_idx, 0, idx_x].cpu().numpy()
            real_values_y = real_data[event_idx, 0, idx_y].cpu().numpy()

            # Extract generated values for MET
            gen_values_x = gen_data[:, event_idx, 0, idx_x].cpu().numpy()
            gen_values_y = gen_data[:, event_idx, 0, idx_y].cpu().numpy()

            # Plot 2D histogram for generated samples
            plt.subplot(1, len(feature_pairs), pair_idx + 1)
            plt.hist2d(gen_values_x, gen_values_y, bins=50, density=True, cmap="viridis")
            plt.colorbar(label="Density")
            plt.scatter(real_values_x, real_values_y, color="red", label="True Value", edgecolor="white", s=80)
            plt.xlabel(feat_x)
            plt.ylabel(feat_y)
            plt.title(f"{feat_x} vs {feat_y}")
            plt.legend()

        plt.tight_layout()
        plt.show()

    # Jets: Loop through particles
    else:
        num_particles = real_data.size(1)
        for particle_idx in range(num_particles):  # Loop over particles
            plt.figure(figsize=(12, 4 * len(feature_pairs)))
            plt.suptitle(f"{object_name.capitalize()} {particle_idx} in Event {event_idx}")

            for pair_idx, (feat_x, feat_y) in enumerate(feature_pairs):
                idx_x = feat_idx_map[feat_x]
                idx_y = feat_idx_map[feat_y]

                # Extract real values for the feature pair
                real_values_x = real_data[event_idx, particle_idx, idx_x].cpu().numpy()
                real_values_y = real_data[event_idx, particle_idx, idx_y].cpu().numpy()

                # Extract generated values for the feature pair
                gen_values_x = gen_data[:, event_idx, particle_idx, idx_x].cpu().numpy()
                gen_values_y = gen_data[:, event_idx, particle_idx, idx_y].cpu().numpy()

                # Plot 2D histogram for generated samples
                plt.subplot(len(feature_pairs), 1, pair_idx + 1)
                plt.hist2d(gen_values_x, gen_values_y, bins=50, density=True, cmap="viridis")
                plt.colorbar(label="Density")
                plt.scatter(real_values_x, real_values_y, color="red", label="True Value", edgecolor="white", s=80)

                # Set labels and title
                plt.xlabel(feat_x)
                plt.ylabel(feat_y)
                plt.legend()
                plt.title(f"{feat_x} vs {feat_y}")

            plt.tight_layout()
            plt.show()



def plot_trajectories_2d(
        all_traj: torch.Tensor,
        model,
        type_idx: int = 0,
        feat_idx_x: int = 0,
        feat_idx_y: int = 1,
        max_points: int = 2000,
        num_events: int = 5,
        mode: str = "multiple_events",
        event_idx: int = 0,
        object_idx: int = 0,
        preprocessing = None,
        batch=None
):
    """
    2D scatter from x0->x1 for the chosen type and features.
    Each point on scatter plot represents a specific particle at a designated time step.
    The whole plots includes multiple particles (of same type) across multiple events.

    Args:
        all_traj: shape (N_sample, steps+1, B, sum_reco, len_flow_feats)
            The full trajectory data (each step) from model.sample(..., store_trajectories=True).
        model: the trained CFM model, with attributes:
            model.n_reco_particles_per_type
            model.flow_input_features
            model.reco_particle_type_names
        type_idx: Which reco particle type to visualize (0 for jets, 1 for MET, etc.)
        feat_idx_x, feat_idx_y: which feature indices to plot on x,y axes
        event_id: optional integer or string to add in the plot title
        max_points: subsample if the total #points is too large.
        max_points: Maximum number of points to plot (subsampling applied in multiple_events mode).
        num_events: In multiple_events mode, how many events to include.
        mode: "multiple_events" (default) or "single_event".
        event_idx: For single_event mode, the event index to use.
        object_idx: For single_event mode, the object (within the selected type) to plot.
        preprocessing: the preprocessing object to use for the inverse transformation.
    """
    device = model.device
    # N_sample = Number of independent samples generated
    # steps_plus_1 = Number of timesteps along trajectory
    # B = Batch size (number of events)
    N_sample, steps_plus_1, B, sum_reco, len_flow_feats = all_traj.shape

    # The user specifies the index of the desired type of particle to plot
    # The correct section of data is extracted from all_traj
    # E.g. If type_idx=0 => offset=0
    # If type_idx=1 => offset = n_reco_particles_per_type[0]
    offset = sum(model.n_reco_particles_per_type[:type_idx])
    n_type = model.n_reco_particles_per_type[type_idx] # Number of that type of particle

    if mode == "multiple_events":
        num_events = min(num_events, B) # Ensure num_events doesn't exceed batch size
        # Slice out that type and portion of events
        sub_traj = all_traj[:, :, :num_events, offset : offset + n_type, :] # shape => [N_sample, steps+1, num_events, n_type, len_flow_feats]

        if preprocessing is None:
            # Pick the 2 features to plot
            sub_traj_2d = sub_traj[..., [feat_idx_x, feat_idx_y]] # => shape [N_sample, steps+1, num_events, n_type, 2]

            # Flatten B,n_type => a single dimension
            sub_traj_2d = sub_traj_2d.reshape(N_sample, steps_plus_1, -1, 2) # => shape [N_sample, steps+1, B*n_type, 2]

            sample_idx = 0 # Select the first sample for plotting
            traj = sub_traj_2d[sample_idx]  # shape [steps+1, B*n_type, 2]

        else:
            # In inverse mode, perform the inverse transformation on the full feature set.
            full_traj = sub_traj[0]  # shape: [steps+1, num_events, n_type, len_flow_feats]
            # Flatten event and particle dimensions.
            full_traj = full_traj.reshape(steps_plus_1, -1, len_flow_feats)  # shape: [steps+1, num_points, len_flow_feats]
            # If the tensor's last dimension does not match the expected number of fields,
            # and if the model provides flow_indices, select those features.
            fields = model.reco_input_features_per_type[type_idx]
            if hasattr(model, 'flow_indices'):
                desired_num = len(model.flow_indices[type_idx])
                if full_traj.shape[-1] != desired_num:
                    indices = model.flow_indices[type_idx]
                    indices_tensor = torch.tensor(indices, device=device)
                    full_traj = full_traj.index_select(dim=-1, index=indices_tensor)
                flow_fields = [fields[idx] for idx in model.flow_indices[type_idx]]
            else:
                flow_fields = fields

            # Create a mask of ones.
            inv_mask = torch.ones(full_traj.shape[0], full_traj.shape[1], device=device)
            name = model.reco_particle_type_names[type_idx]
            # Apply inverse preprocessing on the full feature set.
            inv_data, _ = preprocessing.inverse(
                name=name,
                x=full_traj,
                mask=inv_mask,
                fields=flow_fields
            )
            # Now select the desired two features for plotting.
            traj = inv_data[..., [feat_idx_x, feat_idx_y]]

    elif mode == "single_event":
        reco_data = [d.to(device) for d in batch['reco']['data']]
        reco_mask_exist = [m.to(device) for m in batch['reco']['mask']]
        reco_data = [r.cpu() for r in reco_data]
        reco_mask_exist = [m.cpu() for m in reco_mask_exist]

        # Ensure event and object indices are within range.
        # Ensure indices are in range.
        if event_idx >= B:
            raise ValueError(f'Event index ({event_idx}) larger than batch_size ({B})')
        if object_idx >= n_type:
            raise ValueError(f'Object index {object_idx} out of range {n_type}')

        # Slice out the chosen event and type.
        # This yields shape: [N_sample, steps+1, 1, n_type, len_flow_feats]
        sub_traj = all_traj[:, :, event_idx:event_idx+1, offset: offset+n_type, :]
        # Select the desired object (object_idx) from the type dimension.
        # Now shape: [N_sample, steps+1, len_flow_feats]
        sub_traj = sub_traj[:, :, 0, object_idx, :]

        if preprocessing is None:
            # Select the two features.
            traj = sub_traj[..., [feat_idx_x, feat_idx_y]]  # shape: [N_sample, steps+1, 2]

        else:  # Apply inverse preprocessing
             # Save a copy of the raw generated samples (after any flow-index reordering) so we can use
            # non-inversed phi values if needed.
            raw_traj = sub_traj.clone()

            name = model.reco_particle_type_names[type_idx]
            fields = list(model.reco_input_features_per_type[type_idx])
            if hasattr(model, 'flow_indices'):
                flow_fields = [fields[idx] for idx in model.flow_indices[type_idx]]
                indices_tensor = torch.tensor(model.flow_indices[type_idx], device=sub_traj.device)
                sub_traj = sub_traj.index_select(dim=-1, index=indices_tensor)
                raw_traj = raw_traj.index_select(dim=-1, index=indices_tensor)
            else:
                flow_fields = fields
                indices_tensor = None

            # Inverse preprocessing for generated samples.
            if preprocessing is None:
                # If no preprocessing is provided, simply select the two features.
                traj = sub_traj[..., [feat_idx_x, feat_idx_y]]
            else:
                N_sample_local, T, F_sel = sub_traj.shape
                reshaped = sub_traj.reshape(N_sample_local * T, 1, F_sel)  # shape: [N_sample_local*T, 1, F_sel]
                inv_mask = torch.ones(reshaped.shape[0], 1, device=sub_traj.device)
                inv_data, _ = preprocessing.inverse(
                    name=name,
                    x=reshaped,
                    mask=inv_mask,
                    fields=flow_fields
                )
                inv_data = inv_data.squeeze(1).reshape(N_sample_local, T, -1)
                # Start with the inverse-transformed data.
                traj = inv_data.clone()
                chosen_features = model.flow_input_features[type_idx]
                # If the x-axis feature is phi, override it with the raw value.
                if chosen_features[feat_idx_x] == "phi":
                    traj[..., feat_idx_x] = raw_traj[..., feat_idx_x]
                # Similarly, if the y-axis feature is phi, override it.
                if chosen_features[feat_idx_y] == "phi":
                    traj[..., feat_idx_y] = raw_traj[..., feat_idx_y]
                # Finally, select only the two features.
                traj = traj[..., [feat_idx_x, feat_idx_y]]


    else:
        raise ValueError("Invalid mode. Choose 'multiple_events' or 'single_event'.")

    feature_names = {
        "pt": r"$p_T$ [GeV]",
        "eta": r"$\eta$",
        "phi": r"$\phi$ [rad]"
    }
    # Extract feature names for the axis labels
    chosen_features = model.flow_input_features[type_idx]
    x_label = chosen_features[feat_idx_x]
    y_label = chosen_features[feat_idx_y]
    # Replace labels if they exist in the mapping
    x_label = feature_names.get(x_label, x_label)
    y_label = feature_names.get(y_label, y_label)
    particle_name = model.reco_particle_type_names[type_idx] # Select particle name

    # Make the plot
    plt.figure(figsize=(6,6))

    if mode == "multiple_events":
        plt.scatter(traj[0, :, 0], traj[0, :, 1], s=5, c="black", alpha=0.8, label="Start", zorder=1) # Start points
        for i in range(traj.shape[1]): # Plot intermediate points as trajectory lines
            plt.plot(traj[:, i, 0], traj[:, i, 1], c="olive", alpha=0.2, linewidth=0.8, zorder=2)  # Connect points with a line
        plt.scatter(traj[-1, :, 0], traj[-1, :, 1], s=5, c="royalblue", alpha=1.0, label="End", zorder=3)# end

    else:
        # Plot each sample's trajectory.
        for i in range(traj.shape[0]):
            plt.plot(traj[i, :, 0], traj[i, :, 1], c="olive", alpha=0.5, linewidth=0.8)
            plt.scatter(traj[i, 0, 0], traj[i, 0, 1], s=5, c="black", alpha=0.8, zorder=1)
            plt.scatter(traj[i, -1, 0], traj[i, -1, 1], s=5, c="royalblue", alpha=1.0, zorder=2)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"Trajectory for {particle_name} ({mode})")
    plt.legend()
    plt.show()


def plot_trajectories_grid(
        all_traj: torch.Tensor,
        model,
        custom_timesteps,
        type_idx: int = 0,
        feat_idx_x: int = 0,
        feat_idx_y: int = 1,
        max_points: int = 2000,
        event_idx: int = 0,
        object_idx: int = 0,
        batch=None,
        grid_size=20
    ):
    """
    Plots a 3-row grid (per time t):
      1) Top row: 2D density (hist2d) at that time step
      2) Middle row: Velocity field (quiver) from compute_velocity(...) or velocity_net
      3) Bottom row: Full trajectories, highlighting the current position

    Args:
        all_traj: shape (N_sample, steps+1, B, sum_reco, 2)
                  The 2D trajectories for (x,y), typically from model.sample(..., store_trajectories=True).
        model: A CFM model with .conditioning(...) and .velocity_net (or compute_velocity(...)).
        custom_timesteps: List of integer time indices in [0..steps].
        type_idx: Which reco type to look at (if you have multiple).
        feat_idx_x, feat_idx_y: The indices in your 2D sub-trajectory corresponding to “x” and “y”.
        max_points: Not used here but kept for consistency.
        mode: "single_event" in this example (plot one event).
        event_idx: Which event from the batch to plot.
        object_idx: Which object within that type to plot.
        preprocessing: If you want to inverse-transform the points for plotting in raw space. (Optional)
        batch: The original dictionary: {"hard": {"data": [...], "mask": [...]}, "reco": {"data": [...], "mask": [...]}}.
        grid_size: The resolution in each axis for the velocity quiver grid.
    """
    device = model.device

    # Data needed for the velocity vector field calculation
    single_event_batch = {
        "hard": {
            "data": [d[event_idx : event_idx+1].to(device) for d in batch["hard"]["data"]],
            "mask": [m[event_idx : event_idx+1].to(device) for m in batch["hard"]["mask"]],
        },
        "reco": {
            "data": [d[event_idx : event_idx+1].to(device) for d in batch["reco"]["data"]],
            "mask": [m[event_idx : event_idx+1].to(device) for m in batch["reco"]["mask"]],
        }
    }

    # Unpack the trajectory shape.
    N_sample, steps_plus_1, B, sum_reco, len_flow_feats = all_traj.shape
    # Determine offset and number of particles for the chosen type.
    offset = sum(model.n_reco_particles_per_type[:type_idx])
    n_type = model.n_reco_particles_per_type[type_idx]

    # Ensure indices are in range.
    if event_idx >= B:
        raise ValueError(f'Event index ({event_idx}) larger than batch_size ({B})')
    if object_idx >= n_type:
        raise ValueError(f'Object index ({object_idx}) out of range {n_type})')
    # Select one event.
    sub_traj = all_traj[:, :, event_idx:event_idx+1, offset: offset+n_type, :]  # shape: [N_sample, steps+1, 1, n_type, len_flow_feats]
    # Select the desired object.
    sample_traj = sub_traj[:, :, 0, object_idx, :]  # shape: [N_sample, steps+1, len_flow_feats]
    # Transpose so that time is the first dimension.
    sample_traj = sample_traj.transpose(0, 1)  # shape: [steps+1, N_sample, len_flow_feats]

    # Now select only the two features.
    traj = sample_traj[..., [feat_idx_x, feat_idx_y]]  # shape: [steps+1, num_points, 2]
    if traj.shape[1] > max_points:
        print(f'Selecting {max_points} out of {traj.shape[1]} to plot.')
        traj = traj[:, :max_points, :]

    # Compute global x/y limits from the trajectory (use all points)
    global_x_min = traj[:, :, 0].min().item()
    global_x_max = traj[:, :, 0].max().item()
    global_y_min = traj[:, :, 1].min().item()
    global_y_max = traj[:, :, 1].max().item()

    # Set up the grid: 3 rows (density, vector field, trajectories) x n_cols columns.
    n_cols = len(custom_timesteps)
    fig, axes = plt.subplots(3, n_cols, figsize=(5 * n_cols, 15))
    if n_cols == 1:
        axes = axes[:, None]

    feature_names = {
        "pt": r"$p_T$",
        "eta": r"$\eta$",
        "phi": r"$\phi$"
    }
    # Extract feature names for the axis labels
    chosen_features = model.flow_input_features[type_idx]
    x_label = chosen_features[feat_idx_x]
    y_label = chosen_features[feat_idx_y]
    # Replace labels if they exist in the mapping
    x_label = feature_names.get(x_label, x_label)
    y_label = feature_names.get(y_label, y_label)
    particle_name = model.reco_particle_type_names[type_idx] # Select particle name

    # Get Transformer context for the single event for the velocity vector field
    with torch.no_grad():
        cond_out = model.conditioning(
            single_event_batch["hard"]["data"], single_event_batch["hard"]["mask"],
            single_event_batch["reco"]["data"], single_event_batch["reco"]["mask"],
        )
        # cond_out shape: [1, sum_reco+1, embed_dim]
        # Remove the null token => [1, sum_reco, embed_dim]
        context_full = cond_out[:, 1:, :]

    # If you want velocity specifically for the single object_idx in this type:
    # pick out that slice from context_full.  shape => [1, embed_dim]
    offset_obj = offset + object_idx
    obj_context = context_full[:, offset_obj : offset_obj+1, :]  # => [1, 1, embed_dim]

    # --- Middle row: Vector field ---
    Nx = grid_size
    Ny = grid_size
    xs = np.linspace(global_x_min, global_x_max, Nx)
    ys = np.linspace(global_y_min, global_y_max, Ny)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")  # shape [Nx, Ny]

    # Flatten => shape [Nx*Ny, 2]
    points_2d = torch.from_numpy(
        np.stack([gx.ravel(), gy.ravel()], axis=-1)
    ).float().to(device)

    # Precompute global vector field magnitudes for all timesteps
    all_mags = []
    for t in custom_timesteps:
        t = int(round(t))
        if t < 0 or t >= steps_plus_1:
            continue
        t_val = t / (steps_plus_1 - 1)
        t_tensor = torch.full((points_2d.shape[0], 1), t_val, device=device)

        # Prepare 4D points (assume sinφ=0, cosφ=1)
        sin_phi = torch.zeros_like(points_2d[:, 0])
        cos_phi = torch.ones_like(points_2d[:, 0])
        points_4d = torch.cat([points_2d, sin_phi.unsqueeze(1), cos_phi.unsqueeze(1)], dim=1)

        # Replicate context and build network input
        cflat = obj_context.reshape(1, -1)  # obj_context from your code above
        c_rep = cflat.repeat(points_2d.shape[0], 1)
        net_in = torch.cat([c_rep, points_4d, t_tensor], dim=1)

        with torch.no_grad():
            v_pred_4 = model.velocity_net(net_in)
        v_pred = v_pred_4[:, [0, 1]].cpu().numpy()  # shape: [N_points, 2]
        mag = np.sqrt(v_pred[:, 0]**2 + v_pred[:, 1]**2)
        all_mags.append(mag)

    global_max = np.max([m.max() for m in all_mags])
    global_min = np.min([m.min() for m in all_mags])
    # Create a shared normalization object
    shared_norm = matplotlib.colors.Normalize(vmin=global_min, vmax=global_max)

    # Now loop over custom timesteps for plotting
    for col, t in enumerate(custom_timesteps):
        t = int(round(t))
        if t < 0 or t >= steps_plus_1:
            print(f"Skipping timestep {t} (out of range)")
            continue

        fontsize = 20
        # --- Top row: Density heatmap (same as before) ---
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

        # --- Middle row: Vector field ---
        # (Reuse the grid we already computed: points_2d)
        t_val = t / (steps_plus_1 - 1)
        t_tensor = torch.full((points_2d.shape[0], 1), t_val, device=device)
        sin_phi = torch.zeros_like(points_2d[:, 0])
        cos_phi = torch.ones_like(points_2d[:, 0])
        points_4d = torch.cat([points_2d, sin_phi.unsqueeze(1), cos_phi.unsqueeze(1)], dim=1)
        cflat = obj_context.reshape(1, -1)
        c_rep = cflat.repeat(points_2d.shape[0], 1)
        net_in = torch.cat([c_rep, points_4d, t_tensor], dim=1)

        with torch.no_grad():
            v_pred_4 = model.velocity_net(net_in)
        v_pred = v_pred_4[:, [0, 1]].cpu().numpy()
        vx = v_pred[:, 0].reshape(Nx, Ny)
        vy = v_pred[:, 1].reshape(Nx, Ny)
        mag = np.sqrt(vx**2 + vy**2)

        ax_mid = axes[1, col]
        # Pass shared_norm to quiver for consistent coloring
        Q = ax_mid.quiver(gx, gy, vx, vy, mag, pivot='mid', cmap='coolwarm', 
                        scale=3, scale_units='xy', angles='xy', width=0.02, norm=shared_norm)
        ax_mid.set_xlim(global_x_min, global_x_max)
        ax_mid.set_ylim(global_y_min, global_y_max)
        ax_mid.set_xlabel(x_label, fontsize=fontsize)
        ax_mid.set_ylabel(y_label, fontsize=fontsize)
        # Use the shared norm for the colorbar
        # fig.colorbar(Q, ax=ax_mid, label="|velocity|", norm=shared_norm)

        # --- Bottom row: Trajectories (same as before) ---
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


def save_samples(samples, filename):
    base_dir = "saved_samples"
    if base_dir and not os.path.exists(base_dir): # Create dir if it doesn't exist
        os.makedirs(base_dir)
    file_path = os.path.join(base_dir, filename)
    torch.save(samples, file_path)
    print(f"Samples saved to {file_path}")


def load_samples(filename):
    base_dir = "saved_samples"
    file_path = os.path.join(base_dir, filename)
    if os.path.exists(file_path):
        samples = torch.load(file_path)
        print(f"Samples loaded from {file_path}")
        return samples
    else:
        print(f"File {file_path} does not exist.")
        return None