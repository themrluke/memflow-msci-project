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


# EarlyStop

Earlystop for validation loss. After the validation loss increases for a number of consecutive epochs -> stop the traianing.

# Plots_for_Regression_and_PhDPresentation

The explanation is already in the name of the file

# Utils

Methods to check the mass of a particle and to compute the energy of the particles

`SavePlots` class -> this is the one used for plots in `Plots_for_Regression_and_PhdPresentation`

`FindMasks` claas -> find the matched events for higgs/thad/tlep/ISR

# Test_NoParamsForDiffArch & TestRambo_WhenMC-on-shell

First: look for different config for the flow arhitecture and count the no. of params

Second: Test neg values for the Rambo (when using MC on-shell data and when using the masses of the partons as the target_mass in Rambo algo.)

# run_generate_config

Script to generate different config for the Conditional Transformer and Unfolding Flow. I used it only at the beggining of the project.

# send_jobs

Script to send a job for different configuration file, I used to use it at the beggining of the project for sending the jobs for pretraining.

# run_pretraining

Pretraining for the Conditional Transformer. Based on `config.noProv`, the pretraining is done using or not using the prov information. The result of the Conditional Transformer is the 3-momenta of H/thad/tlep. The energy of these partons is computed using their on-shell masses and the gluon momenta is computed as the neg sum of the momenta from the regressed particles.

# run_UnscaleTensor

This is used to compute the partonic 3-momenta of H/thad/tlep using CondTransformer after the pretraining is finished. With these results we can compute the 4-momenta of full partonic event (H/thad/tlep/ISR). These 4-momenta are used in the plots between target and regressed variables (like in the Plots_for_Regression_and_PhDPresentation)

# run_model

Script for the training of the flow

By default, run it on CPU and using the training in one direction only.

The results used at this moment for the flow are in: `results_pretraining_eta_correct_21June/noProv/preTraining-MEMFlow_noprov-eta_v00_0/UnfFlowEps_RamboFixed_NewDimFeedForward_rbf` or `results_pretraining_eta_correct_21June/noProv/preTraining-MEMFlow_noprov-eta_v00_0/UnfFlowEps_RamboFixed_OnshellMC_NewDimFeedForward_rbf`. The directory `UnfFlowEps_RamboFixed_NewDimFeedForward_rbf` has neg values of Rambo for PS_target (computed from the MC data) because the partons are not necessary on-shell.

Both use the training in one direction (possibly except `UnfFlowEps_RamboFixed_OnshellMC_NewDimFeedForward_rbf/results_DiagNormal_FirstArg:0_Autoreg:True_SamplingTr:False`) which has a strange output.


