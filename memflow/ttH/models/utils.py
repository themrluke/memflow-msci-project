# Script Name: utils.py
# Author: Luke Johnson


import os
import torch
import torch.nn as nn
from typing import Callable, List, Dict, Optional


def save_samples(
        samples: List[torch.Tensor],
        filename: str
    ) -> None:
    """
    Saves generated samples during inference from the CFM or cINN to .pt files.

    Args:
        - samples (List[torch.Tensor]): list of tensors, containing generated samples, one for each particle type.
        - filename (str): File name to save the generated samples to.
    """
    base_dir = "saved_samples"
    if base_dir and not os.path.exists(base_dir): # Create dir if it doesn't exist
        os.makedirs(base_dir)
    file_path = os.path.join(base_dir, filename)
    torch.save(samples, file_path)
    print(f"Samples saved to {file_path}")


def load_samples(
        filename: str
) -> Optional[List[torch.Tensor]]:
    """
    Loads in the previously saved samples from the .pt file.

    Args:
        - filename (str): File name that the generated samples were saved to.

    Returns:
        - samples (Optional[List[torch.Tensor]]): List of tensors (for each particle type) containing the generated samples.
    """
    base_dir = "saved_samples"
    file_path = os.path.join(base_dir, filename)
    if os.path.exists(file_path):
        samples = torch.load(file_path)
        print(f"Samples loaded from {file_path}")
        return samples
    else:
        print(f"File {file_path} does not exist.")
        return None


def move_batch_to_device(
        batch: Dict[str, Dict[str, List[torch.Tensor]]],
        device: torch.device
) -> Dict[str, Dict[str, List[torch.Tensor]]]:
    """
    Moves all tensors in a nested batch dict to accelerated device.

    Args:
        - batch (Dict[str, Dict[str, List[torch.Tensor]]]): Dict containing lists of tensors.
        - device (torch.device): Target device (e.g. 'cuda', 'cpu').

    Returns:
        - new_batch (Dict[str, Dict[str, List[torch.Tensor]]]): Dict where all tensors are now on device.
    """
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


def pad_t_like_x(
        t: torch.Tensor,
        x: torch.Tensor
) -> torch.Tensor:
    """
    Reshapes time vector t so it can be broadcast with with the tensor x.

    Args:
        - t (torch.Tensor): Current time value during interpolation.
        - x (torch.Tensor): Distribution of particles and features.

    Returns:
        - torch.Tensor: Reshaped time tensor.
    """
    return t.reshape(-1, *([1] * (x.dim() - 1)))



class CircularEmbedding(nn.Module):
    """
    Responsible for correcting for circular nature of φ. Replaces raw φ field with two fields: [sin(φ), cos(φ)].
    These get their own embedding before being fed to a linear layer. Expects the indices of the raw φ columns in the input.

    Parameters:
        - in_features (int): Number of input features to model.
        - out_features (int): Number of output features after embedding.
        - circular_indices (Optional[List[int]]): Indices in the input for features that are circular (φ).
        - embed_act (Optional[Callable[[], nn.Module]]): Activation function to apply following the embedding.
        - dropout (float) = Dropout.
    """
    def __init__(
            self,
            in_features: int,
            out_features: int,
            circular_indices: Optional[List[int]] = None,
            embed_act: Optional[Callable[[], nn.Module]] = None,
            dropout: float = 0.0
    ) -> None:

        super().__init__()
        self.circular_indices = circular_indices  # list of indices corresponding to phi
        # For circular indices (e.g. phi), add an extra feature
        self.linear = nn.Linear(in_features + (len(circular_indices) if circular_indices else 0), out_features)
        self.act = embed_act() if embed_act is not None else None
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for circular embedding.

        Args:
            - x (torch.Tensor): Distribution of particles and features, shape [B, N, in_features].

        Returns:
            - out (torch.Tensor): Modified tensor, shape [B, N, in_features].
        """
        if self.circular_indices is not None:
            # Extract raw phi columns
            circular_vals = x[..., self.circular_indices]

            # Compute sin and cos components
            sin_phi = torch.sin(circular_vals)
            cos_phi = torch.cos(circular_vals)

            # Remove the raw phi columns and replace with [sin(φ), cos(φ)]
            idx_all = list(range(x.shape[-1]))
            idx_remaining = [i for i in idx_all if i not in self.circular_indices]
            x_remaining = x[..., idx_remaining]
            x = torch.cat([x_remaining, sin_phi, cos_phi], dim=-1)

        # Apply linear transformation, activation, and dropout
        out = self.linear(x)
        if self.act is not None:
            out = self.act(out)
        if self.dropout is not None:
            out = self.dropout(out)

        return out
