import htcondor
import sys
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--version', type=str, required=True)
parser.add_argument('--dry', action="store_true")
parser.add_argument('--antonio', action="store_true")
parser.add_argument('--interactive', action="store_true")
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument("--good-gpus", action="store_true")
parser.add_argument("--args", nargs="+", type=str, help="additional args")
args = parser.parse_args()

model = args.model
version = args.version
dry = args.dry
antonio = args.antonio
interactive = args.interactive

col = htcondor.Collector()
credd = htcondor.Credd()
credd.add_user_cred(htcondor.CredTypes.Kerberos, None)

if antonio:
    basedir = "/afs/cern.ch/user/a/adpetre/public/memflow/MEMFlow"
else:
    basedir = "/afs/cern.ch/work/d/dvalsecc/private/MEM/MEMFlow"

sub = htcondor.Submit()

if interactive:
    sub['InteractiveJob'] = True

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


elif model == "flow_labframe_psloss":
    sub['Executable'] = f"{basedir}/jobs/script_condor_flow_labframe_psloss.sh"
    sub['Error'] = f"{basedir}/jobs/error/flow-pretrain-labframe-psloss-v{version}-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/flow-pretrain-labframe-psloss-v{version}-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/flow-pretrain-labframe-psloss-v{version}-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"nextweek"'
    sub['arguments'] = f"flow_pretrain_labframe_psloss_logit/flow_labframe_psloss_v{version}.yaml flow_labframe_psloss_logit"


elif model == "flow_labframe_sampling":
    sub['Executable'] = f"{basedir}/jobs/script_condor_flow_labframe_sampling.sh"
    sub['Error'] = f"{basedir}/jobs/error/flow-pretrain-labframe-sampling-v{version}-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/flow-pretrain-labframe-sampling-v{version}-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/flow-pretrain-labframe-sampling-v{version}-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"nextweek"'
    sub['arguments'] = f"flow_pretrain_labframe_sampling/flow_labframe_sampling_v{version}.yaml flow_labframe_sampling"


elif model == "flow_nopretrain":
    sub['Executable'] = f"{basedir}/jobs/script_condor_flow_nopretrain.sh"
    sub['Error'] = f"{basedir}/jobs/error/flow-nopretrain-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/flow-nopretrain-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/flow-nopretrain-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"nextweek"'
    sub['arguments'] = f"flow_nopretrain_spanet_logit/flow_nopretrain_v{version}.yaml flow_nopretrain_spanet_logit"


elif model == "flow_evaluation_labframe":
    sub['Executable'] = f"{basedir}/jobs/script_condor_flow_evaluation_labframe.sh"
    sub['Error'] = f"{basedir}/jobs/error/flow-evaluation-labframe-v{version}-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/flow-evaluation-labframe-v{version}-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/flow-evaluation-labframe-v{version}-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"workday"'
    sub['arguments'] = " ".join(args.args)

elif model == "transfer_flow_firstVersion":
    sub['Executable'] = f"{basedir}/jobs/script_condor_transfer_flow.sh"
    sub['Error'] = f"{basedir}/jobs/error/transfer_flow-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/transfer_flow-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/transfer_flow-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"nextweek"'
    sub['arguments'] = f"transferFlow_train/transferFlow_v{version}.yaml transferFlow_v{version}"

elif model == "transfer_flow_paperVersion":
    sub['Executable'] = f"{basedir}/jobs/script_condor_transfer_flow_paper.sh"
    sub['Error'] = f"{basedir}/jobs/error/transfer_flow_paper-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/transfer_flow_paper-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/transfer_flow_paper-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"nextweek"'
    sub['arguments'] = f"transferFlow_paper_train/transferFlow_v{version}.yaml transferFlow_paper_v{version}"
    
elif model == "run_transferFlow_paperVersion-fake_partons-NoBtag":
    sub['Executable'] = f"{basedir}/jobs/script_condor_transfer_flow_paper_fakepartons_nobtag.sh"
    sub['Error'] = f"{basedir}/jobs/error/transfer_flow_paper_fakepartons_nobtag-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/transfer_flow_paper_fakepartons_nobtag-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/transfer_flow_paper_fakepartons_nobtag-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"nextweek"'
    sub['arguments'] = f"transferFlow_paper_train/transferFlow_v{version}.yaml transferFlow_paper__fakepartons_nobtag_v{version}"

elif model == "run_transferFlow_paperVersion-Nofake_partons-NoBtag":
    sub['Executable'] = f"{basedir}/jobs/script_condor_transfer_flow_paper_Nofakepartons_nobtag.sh"
    sub['Error'] = f"{basedir}/jobs/error/transfer_flow_paper_Nofakepartons_nobtag-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/transfer_flow_paper_Nofakepartons_nobtag-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/transfer_flow_paper_Nofakepartons_nobtag-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"nextweek"'
    sub['arguments'] = f"transferFlow_paper_train/transferFlow_v{version}.yaml transferFlow_paper__Nofakepartons_nobtag_v{version}"

elif model == "run_transferFlow_paperVersion-MDMM-Nofake_partons-NoBtag":
    sub['Executable'] = f"{basedir}/jobs/script_condor_transfer_flow_paper_MDMM_Nofakepartons_nobtag.sh"
    sub['Error'] = f"{basedir}/jobs/error/transfer_flow_paper_MDMM_Nofakepartons_nobtag-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/transfer_flow_paper_MDMM_Nofakepartons_nobtag-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/transfer_flow_paper_Nofakepartons_MDMM_nobtag-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"nextweek"'
    sub['arguments'] = f"transferFlow_paper_train/transferFlow_v{version}.yaml transferFlow_paper_MDMM_Nofakepartons_nobtag_v{version}"

elif model == "run_transferFlow_paperVersion-MDMM-MMD-Nofake_partons-NoBtag":
    sub['Executable'] = f"{basedir}/jobs/script_condor_transfer_flow_paper_MDMM_MMD_Nofakepartons_nobtag.sh"
    sub['Error'] = f"{basedir}/jobs/error/transfer_flow_paper_MDMM_MMD_Nofakepartons_nobtag-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/transfer_flow_paper_MDMM_MMD_Nofakepartons_nobtag-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/transfer_flow_paper_Nofakepartons_MDMM_MMD_nobtag-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"nextweek"'
    sub['arguments'] = f"transferFlow_paper_train/transferFlow_v{version}.yaml transferFlow_paper_MDMM_MMD_Nofakepartons_nobtag_v{version}"

elif model == "transfer_flow_idea3":
    sub['Executable'] = f"{basedir}/jobs/script_condor_transfer_flow_idea3.sh"
    sub['Error'] = f"{basedir}/jobs/error/transfer_flow_idea3-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/transfer_flow_idea3-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/transfer_flow_idea3-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"nextweek"'
    sub['arguments'] = f"transferFlow_idea3_train/transferFlow_v{version}.yaml transferFlow_idea3_train_v{version}"

elif model == "transfer_flow_idea3_conditioned":
    sub['Executable'] = f"{basedir}/jobs/script_condor_transfer_flow_idea3_conditioned.sh"
    sub['Error'] = f"{basedir}/jobs/error/transfer_flow_idea3_conditioned-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/transfer_flow_idea3_conditioned-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/transfer_flow_idea3_conditioned-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"nextweek"'
    sub['arguments'] = f"transferFlow_idea3_conditioning_train/transferFlow_v{version}.yaml transferFlow_idea3_conditioned_train_v{version}"
    
elif model == "classifier_nojets":
    sub['Executable'] = f"{basedir}/jobs/script_classifier_nojets.sh"
    sub['Error'] = f"{basedir}/jobs/error/classifier_nojets-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/classifier_nojets-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/classifier_nojets-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"longlunch"'
    sub['arguments'] = f"classifier_no_jets/classifier_v{version}.yaml classifier_nojets_train_v{version}"
    
elif model == "classifier_nojets_v2":
    sub['Executable'] = f"{basedir}/jobs/script_classifier_nojets_v2.sh"
    sub['Error'] = f"{basedir}/jobs/error/classifier_nojets_v2-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/classifier_nojets_v2-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/classifier_nojets_v2-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"longlunch"'
    sub['arguments'] = f"classifier_no_jets_v2/classifier_v{version}.yaml classifier_nojets_2nd_train_v{version}"
    
elif model == "classifier_nojets_v3":
    sub['Executable'] = f"{basedir}/jobs/script_classifier_nojets_v3.sh"
    sub['Error'] = f"{basedir}/jobs/error/classifier_nojets_v3-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/classifier_nojets_v3-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/classifier_nojets_v3-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"longlunch"'
    sub['arguments'] = f"classifier_no_jets_v2/classifier_v{version}.yaml classifier_nojets_3rdVersion_2024_v{version}"

# General
sub['request_cpus'] = f"{args.ngpu*3}"
sub['request_gpus'] = f"{args.ngpu}"
if args.good_gpus:
    sub['requirements'] = 'regexp("A100", TARGET.CUDADeviceName) || regexp("V100", TARGET.CUDADeviceName)'

    
if model != "flow_evaluation_labframe" and not os.path.exists(f"{basedir}/configs/{sub['arguments'].split()[0]}"):
    print("Missing configuration file! The jobs has not been submitted")
    exit(1)

if antonio:
    sub['arguments'] = sub['arguments'] + " antonio"
    
print(sub)
if not dry:
    schedd = htcondor.Schedd()
    with schedd.transaction() as txn:
        cluster_id = sub.queue(txn)

    print(f"Submitted to {cluster_id:=}")
