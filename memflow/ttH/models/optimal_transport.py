# Script Name: optimal_transport.py
# Author: Luke Johnson
# Description:
#   Provides various Optimal Transport (OT) plans for use in the CFM models.
#   Includes exact and regularised Wasserstein transport, OT plans, sampling
#   trajectories, and a variety of solvers.



import math
import warnings
from functools import partial
from typing import Optional, Union

import numpy as np
import ot as pot
import torch


class OTPlanSampler:
    """
    For computing and sampling from OT plans. Support for a variety of solvers.
    Initialize OTPlanSampler with custom OT method and parameters.

    Parameters:
        - method (str): Specifies which OT solver to use, supports:
            - 'exact': Exact OT by solving the LP associated with the Wasserstein distance.
            - 'sinkhorn': Entropy regularisation to speed up transport plan calculations.
            - 'unbalanced': Relaxes mass conservation, can transport variable mass.
            - 'partial': Transports a fraction of total mass.
        - reg (float): Regularisation parameter for Sinkhorn-based solvers.
        - reg_m (float): Regularization weight for 'unbalanced' Sinkhorn-knopp solver.
        - normalize_cost (bool): Normalise the cost matrix by its min value to help stabilise Sinkhorn-based solvers.
        - num_threads (Union[int, str]): Number of threads use for the "exact" OT solver. "max", uses all threads.
        - warn (bool): Enable to raise warnings for numerical instability.
    """
    def __init__(
        self,
        method: str,
        reg: float = 0.05,
        reg_m: float = 1.0,
        normalize_cost: bool = False,
        num_threads: Union[int, str] = 1,
        warn: bool = True,
    ) -> None:

        # ot_fn takes (a, b, M) as arguments where a, b are marginals and M is cost matrix
        if method == "exact":
            self.ot_fn = partial(pot.emd, numThreads=num_threads)
        elif method == "sinkhorn":
            self.ot_fn = partial(pot.sinkhorn, reg=reg)
        elif method == "unbalanced":
            self.ot_fn = partial(pot.unbalanced.sinkhorn_knopp_unbalanced, reg=reg, reg_m=reg_m)
        elif method == "partial":
            self.ot_fn = partial(pot.partial.entropic_partial_wasserstein, reg=reg)
        else:
            raise ValueError(f"Unknown method: {method}")

        self.reg = reg                      # Regularistion parameter for Sinkhorn solvers
        self.reg_m = reg_m                  # Regularisation weight for unbalanced
        self.normalize_cost = normalize_cost
        self.warn = warn


    def get_map(
            self,
            x0: torch.Tensor,
            x1: torch.Tensor
    ) -> np.ndarray:
        """
        Computes the OT plan between initial and target states (minibatch) using squared Euclidean cost.

        Args:
            - x0 (torch.Tensor): Initial state.
            - x1 (torch.Tensor): Target state.

        Returns:
            - p (np.ndarray): OT plan.
        """
        a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        M = torch.cdist(x0, x1) ** 2
        if self.normalize_cost:
            M = M / M.max()  # should not be normalized when using minibatches
        p = self.ot_fn(a, b, M.detach().cpu().numpy())
        if not np.all(np.isfinite(p)):
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", M.mean(), M.max())
            print(x0, x1)
        if np.abs(p.sum()) < 1e-8:
            if self.warn:
                warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
            p = np.ones_like(p) / p.size

        return p


    def sample_map(
            self,
            pi: np.ndarray,
            batch_size: int,
            replace: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Samples pairs from the source and target indices.

        Args:
            - pi (numpy array): OT plan between minibatches.
            - batch_size (int): How many samples to draw.
            - replace (bool): Whether or not to sample with replacement.

        Returns:
            - i_s, i_j (tuple[np.ndarray, np.ndarray]): Indices of initial and target pairs.
        """
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(
            pi.shape[0] * pi.shape[1], p=p, size=batch_size, replace=replace
        )

        return np.divmod(choices, pi.shape[1])


    def sample_plan(
            self,
            x0: torch.Tensor,
            x1: torch.Tensor,
            replace: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes an OT plan between an initial and target minibatch and samples pairs accordingly.

        Args:
            - x0 (torch.Tensor): Initial state.
            - x1 (torch.Tensor): Target state.
            - replace(bool): Whether or not to sample with replacement.

        Returns:
            - x0[i] (torch.Tensor): Initial state sample after re-pairing.
            - x1[j] (torch.Tensor): Target state sample after re-pairing.
        """
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0], replace=replace)

        return x0[i], x1[j]


    def sample_plan_with_labels(
            self,
            x0: torch.Tensor,
            x1: torch.Tensor,
            y0: Optional[torch.Tensor] = None,
            y1: Optional[torch.Tensor] = None,
            replace: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Computes an OT plan using squared Euclidean cost and samples pairs of labelled source & target states.

        Args:
            - x0 (torch.Tensor): Initial state minibatch.
            - x1 (torch.Tensor): Target state minibatch.
            - y0 (torch.Tensor): Initial label minibatch.
            - y1 (torch.Tensor): Target label binibatch.
            - replace(bool): Whether or not to sample with replacement.

        Returns:
            - x0[i] (torch.Tensor): Initial state sample after re-pairing.
            - x1[j] (torch.Tensor): Target state sample after re-pairing.
            - y0[i] (torch.Tensor): Sampled initial labels.
            - y1[j] (torch.Tensor): Sampled target labels.
        """
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0], replace=replace)

        return (
            x0[i],
            x1[j],
            y0[i] if y0 is not None else None,
            y1[j] if y1 is not None else None,
        )


    def sample_trajectory(
            self,
            X: torch.Tensor,
    ) -> np.ndarray:
        """
        Computes OT transport trajectories over time, mapping between different sample populations.

        Args:
            - X (torch.Tensor): Sample populations across time steps.

        Returns:
            to_return (np.ndarray): Sampled OT trajectories.
        """
        times = X.shape[1]
        pis = []
        for t in range(times - 1):
            pis.append(self.get_map(X[:, t], X[:, t + 1]))

        indices = [np.arange(X.shape[0])]
        for pi in pis:
            j = []
            for i in indices[-1]:
                j.append(np.random.choice(pi.shape[1], p=pi[i] / pi[i].sum()))
            indices.append(np.array(j))

        to_return = []
        for t in range(times):
            to_return.append(X[:, t][indices[t]])
        to_return = np.stack(to_return, axis=1)

        return to_return


def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 2,
    **kwargs,
) -> float:
    """
    Computes the Wasserstein distance (order: 1 or 2) between initial and target states using Euclidean cost.

    Args:
        - x0 (torch.Tensor): Initial state minibatch.
        - x1 (torch.Tensor): Target state minibatch.
        - method (str): Solver to use 'exact' or 'sinkhorn.
        - reg (float): Regularisation parameter for 'sinkhorn'.
        - power (int): Order of Wasserstein distance (1 or 2)

    Returns:
        - ret (float): Caalculated Wasserstein distance.
    """
    assert power == 1 or power == 2

    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=int(1e7))
    if power == 2:
        ret = math.sqrt(ret)

    return ret