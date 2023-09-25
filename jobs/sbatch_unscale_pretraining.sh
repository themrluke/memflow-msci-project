#! /bin/bash

#SBATCH -J gpu-job
#SBATCH --account=gpu_gres               # to access gpu resources
#SBATCH --partition=gpu
#SBATCH --nodes=1                        # request to run job on single node
#SBATCH --ntasks=4                      # request 10 CPU's (t3gpu01/02: balance between CPU and GPU : 5CPU/1GPU)
#SBATCH --gres=gpu:1                     # request  for two GPU's on machine, this is total  amount of GPUs for job
#SBATCH --mem=5G                        # memory (per job)
#SBATCH --time=00-01:00
#SBATCH --gres-flags=disable-binding

echo "cd /work/dvalsecc/MEM/MEMFlow && \
source myenv/bin/activate &&\
python scripts/run_UnscaleTensor.py --model-dir /work/dvalsecc/MEM/models_archive/$1/ --conf $2 --on-GPU --data-validation $3
" > script_unscale_$2.sh

apptainer exec  --nv -B /work/dvalsecc -B `pwd` /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest bash script_unscale_$2.sh
