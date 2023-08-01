# MEMFlow

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]
[![Code style: black][black-badge]][black-link]

[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]
[![Gitter][gitter-badge]][gitter-link]




[actions-badge]:            https://github.com/valsdav/MEMFlow/workflows/CI/badge.svg
[actions-link]:             https://github.com/valsdav/MEMFlow/actions
[black-badge]:              https://img.shields.io/badge/code%20style-black-000000.svg
[black-link]:               https://github.com/psf/black
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/MEMFlow
[conda-link]:               https://github.com/conda-forge/MEMFlow-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/valsdav/MEMFlow/discussions
[gitter-badge]:             https://badges.gitter.im/https://github.com/valsdav/MEMFlow/community.svg
[gitter-link]:              https://gitter.im/https://github.com/valsdav/MEMFlow/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
[pypi-link]:                https://pypi.org/project/MEMFlow/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/MEMFlow
[pypi-version]:             https://badge.fury.io/py/MEMFlow.svg
[rtd-badge]:                https://readthedocs.org/projects/MEMFlow/badge/?version=latest
[rtd-link]:                 https://MEMFlow.readthedocs.io/en/latest/?badge=latest
[sk-badge]:                 https://scikit-hep.org/assets/images/Scikit--HEP-Project-blue.svg


# MMDLoss

This loss when training the model in the sampling direction. This means that we sample from the base distribution (BoxUniform or DiagNormal until now), then go to the Rambo space and compare this sample with the target PS. This comparison is done by using MMDLoss (it's a way to compare samples from different distributions)

It has 2 types of kernels: `multiscale` and `rbf`

# Utils

## Compute_ParticlesTensor:get_HttISR

From the output of regression -> compute and return the full cartesian momenta of H, t, tbar, ISR (["t", "x", "y", "z"]). For the moment, the regression returns the ["pt", "eta", "phi"] of H, t, tbar -> need to scale these variables back to the original space, then compute the energy of the particles by using the on-shell mass. Finally, the ISR is computed as the negative sum of the momenta from the regressed particles (to have the full partonic event in the CM).

This uses awkward arrays (problem with the memory)

## Compute_ParticlesTensor:get_HttISR_numpy

Same as above, but use numpy arrays. Better memory management. 

Use an eps variable here to compute the gluon energy. Instead of sqrt(px^2 + py^2 + pz^2), use sqrt(px^2 + py^2 + pz^2) + eps. In this way, we can remove the possibility of "negative mass" for the gluon (because sometimes sqrt(px^2 + ...)^2 - px^2 - py^2 - pz^2 can return negative numbers -> this is not good for the Rambo algo)

## Compute_ParticlesTensor:get_PS

Return the rambo space given momenta,x1,x2. Same as in the Rambo Algo, except here I don't compute the det of jacobian (it's 0 by default). This is used for the moment in the unfolding flow

# Unfolding_flow

In this class, I initialize the pretrained Conditional Transformer (load the weights of the pretrained one). Then, I initialize the flow, which can use a BoxUniform or DiagNormal base distribution. This flow from the zuko library receives the following parameters:

- `features` - space dimension of the Rambo space
- `context` - space dimension of the conditioning
- `transforms` - number of transformations in the flow (aka invertible maps)
- `bins` - number of bins for each transformation, we use rational-quadratic splines
- `hidden_features` - size of the feed forward
- `randperm`
- `base` - BoxUniform or DiagNormal until now
- `base_args` - for BoxUniform = boundaries of the distrib; for DiagNormal = mean and std dev of the distrib
- `univariate_kwargs={"bound": 1. }` - keep the flow in [-1,1] box; outside this interval there is a identity transformation, which will not affect the variables; For the DiagNormal, this distrib can sample points outside [-1,1]
- `passes` - to choose between autoregressive or coupling layers

The first transformation of the flow is an affine transformation to move from [0,1] space to [-1,1] space

The forward method of the flow returns the PS_regressed from the conditional transformer