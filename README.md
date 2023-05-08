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

## memflow/unfolding\_flow/generate\_config.py
- Create yaml file which keeps: input dataset, input shape, model and training parameters. The yaml files are created in `scripts/configs`

To run it with a minimum number of arguments:
```
python memflow/unfolding_flow/generate_config.py --path_to_dataset=<path-to-Dataset>
```
For a full list of arguments, run:
```
python memflow/unfolding_flow/generate_config.py -h
```

## scripts/run\_generate\_config.py
- Create multiple config files. All the possible configurations will be generated (as an example iterate over learning-rate list, hidden-layers list etc.).

To run it:
```
python scripts/run_generate_config.py --path_to_dataset=<path-to-Dataset> --maxFiles=-1
```

The argument maxFiles represent the maximum number of config files generated. By default, it is -1 (generate all possible config files).

## scripts/run\_model.py
- Run the model (training and validate loops) for a specific config files. 

To run it:
```
python scripts/run_model.py --path_config=<path-configFile> --on_CPU 0
```

By default, on\_CPU argument is set to 0 (run on GPU). If it set to 1, the code will run on CPU

