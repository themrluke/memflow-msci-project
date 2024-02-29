#!/bin/bash

#creating venv in local job dir
python -m venv myenv --system-site-packages
source myenv/bin/activate

cd /afs/cern.ch/user/a/adpetre/public/memflow/MEMFlow
pip install -e .

python scripts/run_transferFlow_paperVersion.py \
      --path-config  $1 --output-dir  $2 --on-GPU $3