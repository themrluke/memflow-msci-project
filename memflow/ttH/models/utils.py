# utils.py

import matplotlib.pyplot as plt
import numpy as np
import torch
import os

class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))


def compare_distributions(real_data, gen_data, real_feat_idx=0, gen_feat_idx=0, nbins=50, feat_name="Feature"):
    """
    Compare histograms of real vs. generated data for a single feature.
    real_data, gen_data: shape (B, nParticles, nFeatures)
    """
    real_vals = real_data[..., real_feat_idx].cpu().numpy().ravel()
    gen_vals  = gen_data[..., gen_feat_idx].cpu().numpy().ravel()

    plt.figure(figsize=(6,4))
    plt.hist(real_vals, bins=nbins, density=True, histtype='step', label="Real")
    plt.hist(gen_vals,  bins=nbins, density=True, histtype='step', label="Generated")
    plt.xlabel(feat_name)
    plt.ylabel("Density")
    plt.title(f"{feat_name}: Real vs Generated")
    plt.legend()
    plt.show()


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
        preprocessing = None
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

            # sub-sample if too large to ensure readability
            if traj.shape[1] > max_points:
                traj = traj[:, :max_points, :]

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
                    indices_tensor = torch.tensor(indices, device=full_traj.device)
                    full_traj = full_traj.index_select(dim=-1, index=indices_tensor)
                flow_fields = [fields[idx] for idx in model.flow_indices[type_idx]]
            else:
                flow_fields = fields

            # Create a mask of ones.
            inv_mask = torch.ones(full_traj.shape[0], full_traj.shape[1], device=full_traj.device)
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
            if traj.shape[1] > max_points:
                traj = traj[:, :max_points, :]

    elif mode == "single_event":
        # Ensure event and object indices are within range.
        if event_idx >= B:
            event_idx = B - 1
        if object_idx >= n_type:
            object_idx = 0
        # Slice out the chosen event and type.
        # This yields shape: [N_sample, steps+1, 1, n_type, len_flow_feats]
        sub_traj = all_traj[:, :, event_idx:event_idx+1, offset: offset+n_type, :]
        # Select the desired object (object_idx) from the type dimension.
        # Now shape: [N_sample, steps+1, len_flow_feats]
        sub_traj = sub_traj[:, :, 0, object_idx, :]

        if preprocessing is None:
            # Select the two features.
            traj = sub_traj[..., [feat_idx_x, feat_idx_y]]  # shape: [N_sample, steps+1, 2]

        else:
            N_sample_local, T, F_full = sub_traj.shape
            inv_data = sub_traj.reshape(N_sample_local * T, F_full).unsqueeze(1)  # shape: [N_sample_local*T, 1, F_full]
            # If the full feature dimension doesn't match the expected number, use flow_indices to select the correct features.
            fields = model.reco_input_features_per_type[type_idx]
            if hasattr(model, 'flow_indices'):
                desired_num = len(model.flow_indices[type_idx])
                if inv_data.shape[-1] != desired_num:
                    indices = model.flow_indices[type_idx]
                    indices_tensor = torch.tensor(indices, device=inv_data.device)
                    inv_data = inv_data.index_select(dim=-1, index=indices_tensor)
                flow_fields = [fields[idx] for idx in model.flow_indices[type_idx]]
            else:
                flow_fields = fields
            inv_mask = torch.ones(inv_data.shape[0], 1, device=sub_traj.device)
            name = model.reco_particle_type_names[type_idx]
            inv_data, _ = preprocessing.inverse(
                name=name,
                x=inv_data,
                mask=inv_mask,
                fields=flow_fields
            )
            inv_data = inv_data.squeeze(1).reshape(N_sample_local, T, -1)
            traj = inv_data[..., [feat_idx_x, feat_idx_y]]

    else:
        raise ValueError("Invalid mode. Choose 'multiple_events' or 'single_event'.")

    # Extract feature names for the axis labels
    chosen_features = model.flow_input_features[type_idx]
    x_label = chosen_features[feat_idx_x]
    y_label = chosen_features[feat_idx_y]
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