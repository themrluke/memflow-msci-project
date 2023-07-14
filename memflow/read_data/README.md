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


# dataset_all

Read partonic and reco events. First time, run this class with the argument `build=True` set, to build all the files for the MC data. Then, use `build=False` to read the data from the files.

# Dataset_Reco_Level

Using this class, one can read the following data:

- `data_jets` and `mask_jets` (in the CM)
- `data_lepton` and `mask_lepton` (in the CM)
- `data_met` and `mask_met` (in the CM)
- `data_boost` and `mask_boost` (sum of final state particles - NOT IN CM - has px/py components)
- `data_boost_objBoosted` and `mask_boost_objBoosted` (boost of the final state particles in the CM - no px/py/pz components)
- `recoParticlesCartesian` (jets, lepton, met in the CM - ["E", "px", "py", "pz"]). I think I saved btag and prov for jets too (for lepton and MET: btag=0, prov=0)
- `recoParticles` (jets, lepton, met in the CM - ["pt", "eta", "phi"]).  I think I saved btag and prov for jets too (for lepton and MET: btag=0, prov=0)
- `scaledLogRecoParticlesCartesian` - for [E, px, py, pz] -> scale momenta as sign(var)*log(1+abs(var)); in addition, each var is scaled (E, px, py, pz), e.g. substract the energy mean and divide by the energy std. dev.
- `scaledLogRecoParticles` - for ["pt", "eta", "phi"] -> scale only pt as sign(pt)*log(1+abs(pt)); each var is svaled (pt, eta, phi)

# Dataset_Parton_Level

Using this class, one can read the following data:

- `data_partons` and `mask_partons` - in the CM - ["pt", "eta", "phi", "mass", "pdgId", "prov"] - not all the MC events are fully matched -> this means that we can't know exactly from where a b-quark is coming
- `data_lepton_partons` and `mask_partons` - in the CM
- `data_boost` and `mask_boost` - ["t", "x", "y", "z"] -> from generator info (x1 and x2)
- `data_higgs_t_tbar_ISR_cartesian` - in this order - ["E", "px", "py", "pz"] -> mass not necessary on-shell
- `data_higgs_t_tbar_ISR_cartesian_onShell` -> modify the energy of these particles to have on-shell masses (by doing this we can remove the negative values from the rambo space)
- `LogScaled_H_thad_tlep_ISR_cartesian` -> same transformation sign(var)*log(1+abs(var)); each var is scaled (I don't know why, but I also have the tensor without the scaling)
- `Log_mean_std_H_thad_tlep_ISR_cartesian` -> contains both mean and stdDev of each var (E, px, py, pz) -> I used it after the regression to go back to the real E/px/py/pz because the training was done on logScaled variables
- `data_higgs_t_tbar_ISR` - ["pt", "eta", "phi", "mass"] -> mass not necessary on-shell
- `logScaled_data_higgs_t_tbar_ISR` -> do only log for pt, but scale all the variables (except the "mass", but I need to recheck this)
- `Log_mean_std_H_thad_tlep_ISR_cartesian` -> similar as above
- `phasespace_intermediateParticles` -> use the rambo class to get the PS for intermediate particles -> problem: if I use the on-shell masses of the particles in the rambo algorithm => negative values in Rambo space, this can be removed in 2 ways: use the masses of the particles from each event (which are not necessary on-shell) or modify the MC data to have on-shell masses
- `phasespace_intermediateParticles_onShell` -> this is the PS of the particles after I modified the MC data to have on-shell masses => negative values are removed
- `phasespace_rambo_detjacobian` TODO: something like the total volume of the phasespace etc.


