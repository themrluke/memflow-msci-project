#!/bin/bash

#creating venv in local job dir
python -m venv myenv --system-site-packages
source myenv/bin/activate

cd /afs/cern.ch/user/a/adpetre/public/memflow/MEMFlow
pip install -e .

python scripts/run_transferFlow_firstVersion.py \
       --config  /afs/cern.ch/user/a/adpetre/public/memflow/MEMFlow/configs/$1 \
       --output-dir  /eos/user/a/adpetre/www/ttHbbAnalysis/MEMFlow/models_archive/$2 \
       --on-GPU
