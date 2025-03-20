<div align="center">
  <h1><strong> Mem-Flow Repo<br> for H -> inv </strong></h1>
  
  <p><strong>Written by: Luke Johnson (sa21722)</strong></p>
  <p><strong>Updated: 20/03/2025</strong></p>
  <p><strong>Subfolder for the 4th year research project! :)</strong></p>
</div>

<br>
<br>

# Getting started

This **README** is designed to be rendered by GitHub or VS Code with `ctrl + shift + v`.

The repo contains code for the ML networks used for the Matrix Element Method (MEM). This subfolder is specifically for the invisible Higgs boson decay produced via the ttH mechanism.

# Setting up the env

Follow this setup on a Linux node, Bristol **gpu04.dice.priv** is suitable

1. Fork this repository.
2. The environment [file](../../env.yml) contains all necessary libraries.
3. Create the Conda environment with `conda env create -f env.yml`
4. Now run: `pip install -e .` to install package in editable mode.
5. The preferred method of interaction with this repository is via [VS Code](https://code.visualstudio.com/download). Download any extensions required as prompted.

<br>

# Repo Layout

**[classifier](classifier/)**
- This folder contains the code for the acceptance and multiplicity classifier models.
- There are also notebooks to train the [acceptance](classifier/acceptance_process_DL.ipynb) and [multiplicity](classifier/multiplicity_process_DL.ipynb) networks.
- Includes [callbacks](classifier/classifier_callbacks.py) for plotting.

<br>

**[Models](models/)**
- Contains different implementations of the conditional flow matching (CFM) models.
    1. [Parallel Transfusion](models/ParallelTransfusion.py) refers to the time dependent Transformer conditioning. This is the best performing current model.
    2. [Transfer-CFM](models/TransferCFM.py) is the simplest model, encoder only setup.
    3. [Original CFM](models/TransferCFM_original.py) is the really accurate original model that was found to be incorrect as not autoregressive.
    4. [Transfusion](models/Transfusion.py) is the autoregressive CFM model similar to the Transfermer (with a lightweight CFM instead of cINN).
- [Optimal Transport](models/optimal_transport.py) includes the various OT solvers.
- [Callbacks](models/callbacks.py) contains the sampling, bias, and multi-residuals plots with the aesthetics inline with the final report.
- [Utils](models/utils.py) contains utility functions for saving model samples, loading samples, and handling the periodic nature of phi in the CFM models.



<br>

**[Trained Model Checkpoints](trained_model_checkpoints/)**
- Has the fully trained `.ckpt` files used to create the CFM results in the report.
- Can specify saving checkpoints here during training.

**[Transfer Flow](transfer_flow/)**
- Contains all the model code from Florian for his autoregressive Transfermer.

<br>

&#9642; The [data_ttH_example](data_ttH_example.ipynb) notebook is good for becoming familiar with the Parquet file layout.

&#9642; The [ttH_dataclasses](ttH_dataclasses.py) handles the preprocessing of all the hard and reco-level data before feeding into the models.

&#9642; The [train_CFM](train_CFM.ipynb) and [train_TransferFlow](train_TransferFlow.ipynb) notebooks perform the training of the CFM (all of them) and Transfermer respectively.

&#9642; [distributions_plots](distribution_plots.py) and [timings_analysis](timings_analysis.py) make all the results plots for the Transfer networks in the report. These can compare the high-level variables, inference times, 4-momenta distributions, etc. between the different models. 

&#9642; [timings](timings.ipynb) loops over different batch sizes and makes the plots for the inference times.

&#9642; [inference](inference.ipynb) and [event_level_sampling](event_level_sampling.ipynb) handles generating/loading samples for all the fancy plots comparing the models.


&#9642; This [pdf](Luke_Johnson_Thesis.pdf) has a formally written account of the results and findings of the investigation.

<br>

# Large files

- It's a good idea to store large files with git Large File System (git LFS).
    - you can download git LFS with:
    ```bash
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh 
    mkdir -p ~/bin
    export PATH=~/bin:$PATH
    apt-get download git-lfs
    dpkg-deb -x git-lfs_* ~/bin
    ```
    - and add it to your $PATH with:
    ```bash
    export PATH="$HOME/bin/usr/local/git-lfs/bin:$PATH"
    ```

- init git lfs in the repo with:
```git lfs install```

Then you can use the standard git commands.