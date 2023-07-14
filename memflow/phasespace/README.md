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


# PhaseSpace

Initialize the class as in the following example:

```phasespace = PhaseSpace(E_CM, [21,21], [21,6,-6,25])```

where E_CM is the center of mass energy; [21,21] are the initial gluon particles and [21,6,-6,25] are the ISR, top,
antitop and Higgs

It contains the functions: 

- generate_random_phase_space_points
- get_momenta_from_ps
- get_ps_from_momenta

# Rambo Generator

Only the `getPSpoint_batch` function was modified. It receives the momenta of the final state particles, the energy ratios x1 and x2. In some cases, the order of the final state particles can be changed (for example: from HttISR to ISRttH) -> this changed has to be done for both momenta tensor and the tensor of final state masses.

For the MC data, because here the partons are not necessary on-shell, the rambo could return negative values if the on-shell masses are used. This is why I added an argument `target_mass`, thus for each partonic event we can compute the masses of particles and use this as an input to `getPSpoint_batch`.

Check always that the rambo transformation is invertible, e.g. momenta -> PS -> momenta. This can be done using the notebook: `notebooks/RamboPartonConversion.ipynb`

# Utils

For some tensors, I cloned the values because `backward()` function returned `Error: in-place operation`