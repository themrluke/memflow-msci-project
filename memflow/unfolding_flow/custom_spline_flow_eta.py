r"""Spline flows."""

import torch

from math import pi
from torch.distributions import Transform

# isort: local
from zuko.flows.autoregressive import MAF
from zuko.flows.core import Unconditional
from zuko.distributions import BoxUniform
from zuko.transforms import CircularShiftTransform, ComposedTransform, MonotonicRQSTransform


def RQSTransform_custom(*phi, bound=3.5) -> Transform:
    r"""Creates a custom rational-quadratic spline (RQS) transformation."""

    return ComposedTransform(
        MonotonicRQSTransform(*phi, bound=bound),
    )


class Custom_spline_flow_eta(MAF):
    r"""Creates a neural circular spline flow (NCSF).

    Circular spline transformations are obtained by composing circular domain shifts
    with regular spline transformations. Features are assumed to lie in the half-open
    interval :math:`[-\pi, \pi[`.

    See also:
        :class:`zuko.transforms.CircularShiftTransform`

    References:
        | Normalizing Flows on Tori and Spheres (Rezende et al., 2020)
        | https://arxiv.org/abs/2002.02428

    Arguments:
        features: The number of features.
        context: The number of context features.
        bins: The number of bins :math:`K`.
        kwargs: Keyword arguments passed to :class:`zuko.flows.autoregressive.MAF`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        bins: int = 8,
        **kwargs,
    ):
        super().__init__(
            features=features,
            context=context,
            univariate=RQSTransform_custom,
            shapes=[(bins,), (bins,), (bins - 1,)],
            **kwargs,
        )
