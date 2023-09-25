import htcondor
import sys

model = sys.argv[1]
version = sys.argv[2]

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
    sub['request_cpus'] = '3'
    sub['request_gpus'] = '1'
    sub['arguments'] = f"pretraining_huber_mmd/pretraining_huber_mmd_v{version}.yaml flow_pretraining_huber_mmd"

    
elif model == "mmd_huber":
    sub['Executable'] = f"{basedir}/jobs/script_condor_pretraining_mmd_huber.sh"
    sub['Error'] = f"{basedir}/jobs/error/mmd-huber-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/mmd-huber-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/mmd-huber-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"nextweek"'
    sub['request_cpus'] = '3'
    sub['request_gpus'] = '1'
    sub['arguments'] = f"pretraining_mmd_huber/pretraining_mmd_huber_v{version}.yaml flow_pretraining_mmd_huber"  

elif model == "huber_mmd_labframe":
    sub['Executable'] = f"{basedir}/jobs/script_condor_pretraining_huber_mmd_labframe.sh"
    sub['Error'] = f"{basedir}/jobs/error/huber-mmd-labframe-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/huber-mmd-labframe-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/huber-mmd-labframe-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"testmatch"'
    sub['request_cpus'] = '3'
    sub['request_gpus'] = '1'
    sub['arguments'] = f"pretraining_huber_mmd_labframe/pretraining_huber_mmd_labframe_v{version}.yaml flow_pretraining_huber_mmd_labframe"

    
schedd = htcondor.Schedd()
with schedd.transaction() as txn:
    cluster_id = sub.queue(txn)
     
print(cluster_id)
