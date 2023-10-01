#!/bin/bash

cd /afs/cern.ch/work/d/dvalsecc/private/MEM/MEMFlow
pip install -e .


python scripts/run_model_nopartonpretraining.py \
       --model-dir  /afs/cern.ch/work/d/dvalsecc/private/MEM/MEMFlow/configs/pretraining_provSPANET/ \
       --output-dir /eos/user/d/dvalsecc/www/ttHbbAnalysis/MEMFlow/models_archive/flow_nopretraining_v1  --on-GPU
