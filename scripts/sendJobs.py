import argparse
import pathlib

import os
from os import listdir
from os.path import isfile, join

# get absolute paths of Files from directory
def absoluteFilePaths(directory):
    paths = []
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            paths.append(os.path.abspath(os.path.join(dirpath, f)))
            
    return paths

# make a directory (dir) if it doesn't exist
def mkdir_p(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-directory', type=str, required=True, help='path to config.yaml directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--log-dir', type=str, required=True, help='Log directory')
    parser.add_argument('--on-GPU', action="store_true",  help='run on GPU boolean')
    parser.add_argument('--preTraining', action="store_true",  help='run the preTraining script (by default: run `run_model.py`)')
    args = parser.parse_args()
    
    conf_dir = args.config_directory
    outputDir = os.path.abspath(args.output_dir)
    logDir = os.path.abspath(args.log_dir)

    on_GPU = args.on_GPU # by default run on CPU
    if on_GPU:
        on_GPU = '--on-GPU'
    else:
        on_GPU = ''
    
    # get path of the script used
    if args.preTraining:
        scriptPath = str(pathlib.Path(__file__).parent.resolve()) + "/run_pretraining.py"
        jobName = 'preTraining'
    else:
        scriptPath = str(pathlib.Path(__file__).parent.resolve()) + "/run_model.py"
        jobName = 'Training'
    
    print(scriptPath)
    # get absolute paths of all config files in conf_dir
    confFiles = absoluteFilePaths(conf_dir)
        
    job_directory = f"{os.getcwd()}/jobs"
    
    # create job directory (where job config files are kept)
    mkdir_p(job_directory)

    # create output directory
    mkdir_p(outputDir)

    # create log directory
    mkdir_p(logDir)
        
        
    for i, conf_path in enumerate(confFiles):
        
        # one job file for each config file
        job_file = os.path.join(job_directory, f"v{i}.job")

        with open(job_file, 'w') as fh:
            
            fh.writelines("#!/bin/bash\n")
            fh.writelines(f"#SBATCH --job-name={jobName}_v{i}.job\n")
            fh.writelines("#SBATCH --account=gpu_gres\n")
            fh.writelines("#SBATCH --partition=gpu\n")
            fh.writelines("#SBATCH --nodes=1        # request to run job on single node\n")
            fh.writelines("#SBATCH --ntasks=10       # request 10 CPU's\n")
            fh.writelines("#SBATCH --gres=gpu:1     # request 1 GPU's on machine\n")
            fh.writelines("#SBATCH --mem=5000M      # memory (per job)\n")
            fh.writelines("#SBATCH --time=2-00:00   # time  in format DD-HH:MM\n")
            fh.writelines(f"#SBATCH --output={logDir}/slurm-%x.%j.out # set output name  \n")
            fh.writelines(f"#SBATCH --error={logDir}/slurm-%x.%j.err # set error output name  \n")    
            
            fh.writelines("\n\n")
            fh.writelines("# Activate environment:\n")
            fh.writelines("source /work/$USER/miniconda3/etc/profile.d/conda.sh\n")
            fh.writelines("export PATH=/work/$USER/miniconda3/bin:$PATH\n")
            fh.writelines("export LD_LIBRARY_PATH=/work/adpetre/miniconda3/envs/dizertatie/lib:$LD_LIBRARY_PATH\n") 
            fh.writelines("conda activate dizertatie\n")
            
            fh.writelines("\n\n")
            fh.writelines("# each node has local /scratch space to be used during job run\n")
            fh.writelines("mkdir -p /scratch/$USER/${SLURM_JOB_ID}\n")
            fh.writelines("export TMPDIR=/scratch/$USER/${SLURM_JOB_ID}\n")
            
            fh.writelines("\n\n")
            fh.writelines("echo CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES\n")
            fh.writelines("# python program script.py should use CUDA_VISIBLE_DEVICES variable (*NOT* hardcoded GPU's numbers)\n")
            fh.writelines(f"/work/adpetre/miniconda3/envs/dizertatie/bin/python {scriptPath} --path-config={conf_path} \
                            --output-dir={outputDir} {on_GPU}\n")
            
            fh.writelines("\n\n")
            fh.writelines("# cleaning of temporal working dir when job was completed:\n")
            fh.writelines("rm  -rf /scratch/$USER/${SLURM_JOB_ID}\n")


        os.system(f"sbatch {job_file}")
                
        