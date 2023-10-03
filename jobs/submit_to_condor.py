import htcondor
import sys
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--version', type=str, required=True)
parser.add_argument('--dry', action="store_true")
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument("--good-gpus", action="store_true")
args = parser.parse_args()

model = args.model
version = args.version
dry = args.dry



col = htcondor.Collector()
credd = htcondor.Credd()
credd.add_user_cred(htcondor.CredTypes.Kerberos, None)

basedir = "/afs/cern.ch/work/d/dvalsecc/private/MEM/MEMFlow"

sub = htcondor.Submit()

if model == "huber_mmd":
    sub['Executable'] = f"{basedir}/jobs/script_condor_pretraining_huber_mmd.sh"
    sub['Error'] = f"{basedir}/jobs/error/huber-mmd-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/huber-mmd-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/huber-mmd-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"nextweek"'
    sub['arguments'] = f"pretraining_huber_mmd/pretraining_huber_mmd_v{version}.yaml flow_pretraining_huber_mmd"

    
elif model == "mmd_huber":
    sub['Executable'] = f"{basedir}/jobs/script_condor_pretraining_mmd_huber.sh"
    sub['Error'] = f"{basedir}/jobs/error/mmd-huber-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/mmd-huber-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/mmd-huber-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"nextweek"'
    sub['arguments'] = f"pretraining_mmd_huber/pretraining_mmd_huber_v{version}.yaml flow_pretraining_mmd_huber"  

elif model == "huber_mmd_labframe":
    sub['Executable'] = f"{basedir}/jobs/script_condor_pretraining_huber_mmd_labframe.sh"
    sub['Error'] = f"{basedir}/jobs/error/huber-mmd-labframe-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/huber-mmd-labframe-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/huber-mmd-labframe-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"testmatch"'
    sub['arguments'] = f"pretraining_huber_mmd_labframe/pretraining_huber_mmd_labframe_v{version}.yaml flow_pretraining_huber_mmd_labframe"

elif model == "huber_mmd_labframe_gluon":
    sub['Executable'] = f"{basedir}/jobs/script_condor_pretraining_huber_mmd_labframe_gluon.sh"
    sub['Error'] = f"{basedir}/jobs/error/huber-mmd-labframe-gluon-{version}-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/huber-mmd-labframe-gluon-{version}-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/huber-mmd-labframe-gluon-{version}-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"testmatch"'
    sub['arguments'] = f"pretraining_huber_mmd_labframe_gluon/pretraining_huber_mmd_labframe_v{version}.yaml flow_pretraining_huber_mmd_labframe_gluon"

elif model == "huber_mmd_labframe_gluon_distributed":
    sub['Executable'] = f"{basedir}/jobs/script_condor_pretraining_distributed_labframe_gluon.sh"
    sub['Error'] = f"{basedir}/jobs/error/huber-mmd-labframe-gluon-distributed-{version}-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/huber-mmd-labframe-gluon-distributed-{version}-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/huber-mmd-labframe-gluon-distributed-{version}-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"testmatch"'
    sub['arguments'] = f"pretraining_huber_mmd_labframe_gluon/pretraining_huber_mmd_labframe_v{version}.yaml flow_pretraining_huber_mmd_labframe_gluon/distrubuted_lxplus"


elif model == "flow_labframe":
    sub['Executable'] = f"{basedir}/jobs/script_condor_flow_labframe.sh"
    sub['Error'] = f"{basedir}/jobs/error/flow-pretrain-labframe-v{version}-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/flow-pretrain-labframe-v{version}-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/flow-pretrain-labframe-v{version}-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"nextweek"'
    sub['arguments'] = f"flow_pretrain_labframe_logit/flow_pretrain_labframe_logit_v{version}.yaml flow_pretrain_labframe_logit"


elif model == "flow_nopretrain":
    sub['Executable'] = f"{basedir}/jobs/script_condor_flow_nopretrain.sh"
    sub['Error'] = f"{basedir}/jobs/error/flow-nopretrain-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/flow-nopretrain-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/flow-nopretrain-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"nextweek"'
    sub['arguments'] = f"flow_nopretrain_spanet_logit/flow_nopretrain_v{version}.yaml flow_nopretrain_spanet_logit"


# General
sub['request_cpus'] = f"{args.ngpu*3}"
sub['request_gpus'] = f"{args.ngpu}"
if args.good_gpus:
    sub['requirements'] = 'regexp("A100", TARGET.CUDADeviceName) || regexp("V100", TARGET.CUDADeviceName)'

    
if not os.path.exists(f"{basedir}/configs/{sub['arguments'].split()[0]}"):
    print("Missing configuration file! The jobs has not been submitted")
    exit(1)
    

print(sub)
if not dry:
    schedd = htcondor.Schedd()
    with schedd.transaction() as txn:
        cluster_id = sub.queue(txn)

    print(f"Submitted to {cluster_id:=}")
