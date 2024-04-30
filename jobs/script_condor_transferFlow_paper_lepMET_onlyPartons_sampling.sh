#!/bin/bash

#creating venv in local job dir
python -m venv myenv --system-site-packages
source myenv/bin/activate

cd $1
pip install -e .

python scripts/run_transferFlow_paperVersion_pretrained_v3_onlyExist-noMDMM_MET_OnlyPartons_sampling.py \
      --path-model $2 --on-GPU $3