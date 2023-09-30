#!/bin/bash -e 
cd /work/dvalsecc/MEM/MEMFlow

python -m venv /tmp/myenv --system-site-packages

source /tmp/myenv/bin/activate

pip install -e .

python scripts/run_pretraining_distributed_huber_withmmd_labframe_gluon.py  \
       --path-config  /work/dvalsecc/MEM/MEMFlow/configs/$1 \
       --output-dir  /work/dvalsecc/MEM/models_archive/$2 \
       --on-GPU --distributed
     
