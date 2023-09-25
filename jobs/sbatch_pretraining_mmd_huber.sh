#! /bin/bash

#SBATCH -J gpu-job
#SBATCH --account=gpu_gres               # to access gpu resources
#SBATCH --partition=gpu
#SBATCH --nodes=1                        # request to run job on single node
#SBATCH --ntasks=6                      # request 10 CPU's (t3gpu01/02: balance between CPU and GPU : 5CPU/1GPU)
#SBATCH --gres=gpu:2                     # request  for two GPU's on machine, this is total  amount of GPUs for job
#SBATCH --mem=10G                        # memory (per job)
#SBATCH --time=07-00:00
#SBATCH --gres-flags=disable-binding

apptainer exec  --nv -B /work/dvalsecc -B `pwd` /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest bash /work/dvalsecc/MEM/MEMFlow/jobs/script_mmd_huber.sh $1 $2
