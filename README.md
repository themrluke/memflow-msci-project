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


# Installation

# Singularity image

A singularity image is unpacked on the CVMFS system: `/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest`

```
apptainer shell --bind /afs -B /cvmfs/cms.cern.ch --bind /tmp  --bind /eos/cms/  /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest
```

# Environment on lxplus-gpu

```
source /cvmfs/sft.cern.ch/lcg/views/LCG_103cuda/x86_64-centos9-gcc11-opt/setup.sh
python -m venv myenv
source myenv/bin/activate
git clone git@github.com:valsdav/MEMFlow.git
cd MEMFlow
pip install -e .
cd ..
git clone git@github.com:valsdav/zuko.git
cd zuko
pip install -e .

cd ..
pip install jupyterlab
jupyter lab build
# register the environment in jupyter lab
python -m ipykernel install --user --name=myenv

```

# Matrix Element evaluation
- Start apptainer image with madgraph and LHAPDF
- Generate the process
- Compile it for python with https://cp3.irmp.ucl.ac.be/projects/madgraph/wiki/FAQ-General-4

# How to run the scripts:

## scripts/run\_generate\_config.py
- Create multiple config files. All the possible configurations will be generated (as an example iterate over learning-rate list, hidden-layers list etc.). The config file keeps: input dataset, input shape, model and training parameters.

To run it:
```
python scripts/run_generate_config.py --input_dataset=<path-to-Dataset> --maxFiles=-1 <--preTraining>
```

The argument maxFiles represent the maximum number of config files generated. By default, it is -1 (generate all possible config files). The preTraining flag makes the script iterate only on the condTransformer hyperparameters (unfolding flow parameters are set to the default value).

## scripts/run\_model.py
- Run the model (training and validate loops) for a specific config files. 

To run it:
```
python scripts/run_model.py --model-dir=<path> <--on-GPU>
```

--model-dir: path to directory where the config file and ConditionalTransformer weights (from pretraining) are saved.

If `--on-GPU` flag is added, the script will run on GPU.

## scripts/sendJobs.py
- Send jobs:

To run it:
```
python scripts/sendJobs.py --config-directory=<configDir> <--on-GPU> <--preTraining>
```

By default, the script is running on CPU. If `--on-GPU` flag is added, the script will run on GPU.
The script will iterate over `config-Directory` and will send a job for every config file.
If `--preTraining` flag is set, the script will send jobs for `run_pretraining.py`, otherwise it will send jobs
for `run_model.py`

CAREFUL! Some paths are already set in the script, so modify them before using!!

