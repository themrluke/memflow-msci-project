#!/bin/bash

ANTONIOVAR="antonio"

#creating venv in local job dir
python -m venv myenv --system-site-packages
source myenv/bin/activate

if [ "${@: -1}" = "$ANTONIOVAR" ]; then
       echo "Antonio runs"
       cd /afs/cern.ch/user/a/adpetre/public/memflow/MEMFlow
       pip install -e .

       python scripts/run_transferFlow_firstVersion.py \
              --config  /afs/cern.ch/user/a/adpetre/public/memflow/MEMFlow/configs/$1 \
              --output-dir  /eos/user/a/adpetre/www/ttHbbAnalysis/MEMFlow/models_archive/$2 \
              --on-GPU
else
       echo "Davide runs"
       cd /afs/cern.ch/work/d/dvalsecc/private/MEM/MEMFlow
       pip install -e .

       python scripts/run_transferFlow_firstVersion.py \
              --path-config  /afs/cern.ch/work/d/dvalsecc/private/MEM/MEMFlow/configs/$1 \
              --output-dir  /eos/user/d/dvalsecc/www/ttHbbAnalysis/MEMFlow/models_archive/$2 \
              --on-GPU
fi