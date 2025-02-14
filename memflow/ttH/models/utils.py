# utils.py

import matplotlib.pyplot as plt
from scipy.interpolate import griddata
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
            raise ValueError(f'Object index ({object_idx}) out of range {n_type})')

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

    # Loop over custom timesteps (each column)
    for col, t in enumerate(custom_timesteps):
        # Ensure the timestep index is an integer.
        t = int(round(t))
        if t < 0 or t >= steps_plus_1:
            print(f"Skipping timestep {t} (out of range)")
            continue

        # --- Top row: Density heatmap ---
        points = traj[t].cpu().numpy()  # shape: [num_points, 2]
        axes[0, col].hist2d(points[:, 0], points[:, 1],
                            bins=50,
                            density=True,
                            cmap="viridis",
                            range=[[global_x_min, global_x_max], [global_y_min, global_y_max]])
        axes[0, col].set_title(f"T = {t}")
        axes[0, col].set_xlabel(x_label)
        axes[0, col].set_ylabel(y_label)
        axes[0, col].set_xlim(global_x_min, global_x_max)
        axes[0, col].set_ylim(global_y_min, global_y_max)

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

        # Convert t_ind to a normalized time t in [0, 1], if your bridging does that:
        t_val = t / (steps_plus_1 - 1)  # e.g. steps=20 => T=21 => t goes from 0..1
        t_tensor = torch.full((points_2d.shape[0], 1), t_val, device=device)

        # -----------------------------------------------------
        # If your velocity_net expects (embed_dim + 4 + 1) inputs,
        #   i.e. you have 4 flow features [pt, eta, sinφ, cosφ].
        #   But here you want to interpret "x=pt, y=eta", then
        #   sinφ=0, cosφ=1 as a dummy. Adjust as needed if your
        #   feats differ or if you want actual φ from somewhere!
        # -----------------------------------------------------
        # points_2d is [pt, eta], so let's add sinφ=0, cosφ=1
        # shape => [N_points, 4]
        sin_phi = torch.zeros_like(points_2d[:, 0])
        cos_phi = torch.ones_like(points_2d[:, 0])
        points_4d = torch.cat([points_2d, sin_phi.unsqueeze(1), cos_phi.unsqueeze(1)], dim=1)
        # => [N_points, 4]

        # Replicate the single-object context => shape [N_points, embed_dim]
        cflat = obj_context.reshape(1, -1)  # [1, embed_dim]
        c_rep = cflat.repeat(points_4d.shape[0], 1)  # => [N_points, embed_dim]

        # Concatenate => shape [N_points, embed_dim + 4 + 1] = e.g. [N_points, 69]
        net_in = torch.cat([c_rep, points_4d, t_tensor], dim=1)

        # Call velocity_net
        with torch.no_grad():
            # Suppose the output is shape [N_points, 4] => d(pt)/dt, d(eta)/dt, d(sinφ)/dt, d(cosφ)/dt
            v_pred_4 = model.velocity_net(net_in)

        # For a 2D quiver, let's just take the first 2 components => d(pt)/dt, d(eta)/dt
        v_pred = v_pred_4[:, [0, 1]].cpu().numpy()  # shape [N_points, 2]

        # Reshape for quiver => Nx, Ny
        vx = v_pred[:, 0].reshape(Nx, Ny)
        vy = v_pred[:, 1].reshape(Nx, Ny)

        ax_mid = axes[1, col]
        Q = ax_mid.quiver(gx, gy, vx, vy, pivot='mid', cmap='coolwarm', 
                          scale=5, scale_units='xy', angles='xy', width=0.01)
        ax_mid.set_xlim(global_x_min, global_x_max)
        ax_mid.set_ylim(global_y_min, global_y_max)
        ax_mid.set_title(f"t = {t}: velocity")
        ax_mid.set_xlabel("X")
        ax_mid.set_ylabel("Y")
        fig.colorbar(Q, ax=ax_mid, label="|velocity|")

        # --- Bottom row: Trajectories ---
        num_traj = traj.shape[1]
        idx = np.random.choice(num_traj, size=min(10000, num_traj), replace=False) # How many trajectories to plot
        for i in idx:
            axes[2, col].scatter(traj[0, i, 0].cpu().numpy(), traj[0, i, 1].cpu().numpy(),
                                s=7, color="black", alpha=0.8, zorder=1)
            axes[2, col].plot(traj[:t+1, i, 0].cpu().numpy(), traj[:t+1, i, 1].cpu().numpy(),
                              color="olive", alpha=0.5, linewidth=0.8, zorder=2)
            axes[2, col].scatter(traj[t, i, 0].cpu().numpy(), traj[t, i, 1].cpu().numpy(),
                                s=7, color="blue", alpha=1.0, zorder=3)
        axes[2, col].set_xlabel(x_label)
        axes[2, col].set_ylabel(y_label)
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