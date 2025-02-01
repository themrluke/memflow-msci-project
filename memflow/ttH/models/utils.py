import matplotlib.pyplot as plt
import numpy as np
import torch

# Implement some helper functions


class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))


def compare_distributions(real_data, gen_data, feat_idx=0, nbins=50, feat_name="Feature"):
    """
    Compare histograms of real vs. generated data for a single feature.
    real_data, gen_data: shape (B, nParticles, nFeatures)
    """
    real_vals = real_data[..., feat_idx].cpu().numpy().ravel()
    gen_vals  = gen_data[..., feat_idx].cpu().numpy().ravel()

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
        type_idx=0,
        feat_idx_x=0,
        feat_idx_y=1,
        max_points=2000,
        num_events=5,
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
    """
    # N_sample = Number of independent samples generated
    # steps_plus_1 = Number of timesteps along trajectory
    # B = Batch size (number of events)
    N_sample, steps_plus_1, B, sum_reco, len_flow_feats = all_traj.shape

    num_events = min(num_events, B) # Ensure num_events doesn't exceed batch size

    # The user specifies the index of the desired type of particle to plot
    # The correct section of data is extracted from all_traj
    # E.g. If type_idx=0 => offset=0
    # If type_idx=1 => offset = n_reco_particles_per_type[0]
    offset = sum(model.n_reco_particles_per_type[:type_idx])
    n_type = model.n_reco_particles_per_type[type_idx] # Number of that type of particle

    # Slice out that type and portion of events
    sub_traj = all_traj[:, :, :num_events, offset : offset + n_type, :] # shape => [N_sample, steps+1, num_events, n_type, len_flow_feats]

    # Pick the 2 features to plot
    sub_traj_2d = sub_traj[..., [feat_idx_x, feat_idx_y]] # => shape [N_sample, steps+1, num_events, n_type, 2]

    # Flatten B,n_type => a single dimension
    sub_traj_2d = sub_traj_2d.reshape(N_sample, steps_plus_1, -1, 2) # => shape [N_sample, steps+1, B*n_type, 2]

    # Select the first sample for plotting
    sample_idx = 0
    traj = sub_traj_2d[sample_idx]  # shape [steps+1, B*n_type, 2]

    # sub-sample if too large to ensure readability
    if traj.shape[1] > max_points:
        traj = traj[:, :max_points, :]

    # Extract feature names for the axis labels
    chosen_features = model.flow_input_features[type_idx]
    x_label = chosen_features[feat_idx_x]
    y_label = chosen_features[feat_idx_y]

    # Dynamically select particle name
    particle_name = model.reco_particle_type_names[type_idx]

    # Make the plot
    plt.figure(figsize=(6,6))
    plt.scatter(traj[0, :, 0], traj[0, :, 1], s=10, c="black", alpha=0.8, label="Start") # Start points
    for i in range(traj.shape[1]): # Plot intermediate points as trajectory lines
        plt.plot(traj[:, i, 0], traj[:, i, 1], c="olive", alpha=0.2, linewidth=0.8)  # Connect points with a line
    plt.scatter(traj[-1, :, 0], traj[-1, :, 1], s=20, c="blue", alpha=1.0, label="End")# end

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"Trajectory for {particle_name}")
    plt.legend()
    plt.show()