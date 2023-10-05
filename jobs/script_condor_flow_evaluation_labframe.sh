#!/bin/bash

#creating venv in local job dir
python -m venv myenv --system-site-packages
source myenv/bin/activate

cd /afs/cern.ch/work/d/dvalsecc/private/MEM/MEMFlow
pip install -e .

       
python scripts/run_flow_evaluation_labframe.py  \
       --path-config $1 \
       --output-path $2 \
       --model-weights $3 \
       --on-GPU \
       --nsamples $4 \
       --nevents $5 \
       --batch-size $6

