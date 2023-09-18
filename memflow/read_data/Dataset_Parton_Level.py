from memflow.read_data import utils
from memflow.phasespace.phasespace import PhaseSpace
import os
import numpy.ma as ma
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import awkward as ak

import torch
import numpy as np
torch.set_default_dtype(torch.double)



class Dataset_PartonLevel(Dataset):
    def __init__(self, root, object_types=["partons", "lepton_partons", "boost",
                                           "H_thad_tlep_ISR", "H_thad_tlep_ISR_cartesian"], dev=None, debug=False,
                 dtype=None, build=False, parton_list=[]):

        self.fields = {
            "partons": ["pt", "eta", "phi", "mass", "pdgId", "prov"],
            "boost": ["t", "x", "y", "z"],
            "incoming_particles_boost": ["t", "x", "y", "z"],
            "lepton_partons": ["pt", "eta", "phi", "mass", "pdgId"],
            "H_thad_tlep_ISR": ["pt", "eta", "phi", "mass"],
            "H_thad_tlep_ISR_cartesian": ["E", "px", "py", "pz"]
        }
        
        print("\nPartonLevel")
        self.debug = debug

        self.root = root
        if root.endswith(".parquet"):
            self.rootDir = root.replace(".parquet","")
        else:
            self.rootDir = root            

        self.parton_list = parton_list
        os.makedirs(self.rootDir + "/processed_partons", exist_ok=True)
        self.object_types = object_types
        self.build = build

        # if build flag set or number of files in processed partons directory is 0
        if (build or len(os.listdir(self.rootDir + '/processed_partons/')) == 0):

            (self.partons_boosted, self.leptons_boosted,
            self.higgs_boosted, self.generator,
            self.gluon_ISR, self.boost) = self.get_PartonsAndLeptonsAndBoost()

            for object_type in self.object_types:
                print("Create new file for " + object_type)
                if object_type == "H_thad_tlep_ISR":
                    self.process_intermediateParticles()
                elif object_type == "H_thad_tlep_ISR_cartesian":
                    self.process_intermediateParticles_cartesian()
                else:
                    self.process(object_type)
            
            # need these files for the next operations
            self.mask_boost, self.data_boost = torch.load(
                                    self.processed_file_names("boost"))
            self.data_higgs_t_tbar_ISR_cartesian = torch.load(
                                    self.processed_file_names("H_thad_tlep_ISR_cartesian"))
            self.data_higgs_t_tbar_ISR = torch.load(
                                    self.processed_file_names("H_thad_tlep_ISR"))

            print("Create new file for data_higgs_t_tbar_ISR_cartesian_onShell")
            self.get_intermediateParticles_cartesian_onShell()

            self.data_higgs_t_tbar_ISR_cartesian_onShell = torch.load(
                                    self.processed_file_names("H_thad_tlep_ISR_cartesian_onShell"))

            print("Create new file for PhaseSpace + rambo detJacobian")
            self.get_PS_intermediateParticles()

            print("Create new file for PhaseSpace_onShell (no negative values)")
            self.get_PS_intermediateParticles_onShell()

            print("Create new file for Log_H_thad_tlep_ISR_cartesian")
            self.ProcessCartesianScaled()

            print("Create new file for Log_H_thad_tlep_ISR")
            self.ProcessScaled()

        print("Reading parton_level Files")

        self.mask_partons, self.data_partons = torch.load(
            self.processed_file_names("partons"))
        self.mask_lepton_partons, self.data_lepton_partons = torch.load(
            self.processed_file_names("lepton_partons"))
        self.mask_boost, self.data_boost = torch.load(
            self.processed_file_names("boost"))
        self.data_higgs_t_tbar_ISR = torch.load(
            self.processed_file_names("H_thad_tlep_ISR"))
        self.data_higgs_t_tbar_ISR_cartesian = torch.load(
            self.processed_file_names("H_thad_tlep_ISR_cartesian"))
        self.data_higgs_t_tbar_ISR_cartesian_onShell = torch.load(
            self.processed_file_names("H_thad_tlep_ISR_cartesian_onShell"))
        
        if 'phasespace_intermediateParticles' in self.parton_list:
            print("Load phasespace_intermediateParticles")
            self.phasespace_intermediateParticles = torch.load(
                self.processed_file_names("phasespace_intermediateParticles"))

        if 'phasespace_intermediateParticles_onShell' in self.parton_list:
            print("Load phasespace_intermediateParticles_onShell")
            self.phasespace_intermediateParticles_onShell = torch.load(
                self.processed_file_names("phasespace_intermediateParticles_onShell"))
            self.phasespace_rambo_detjacobian_onShell = torch.load(
                self.processed_file_names("phasespace_rambo_detjacobian_onShell"))

        if 'phasespace_rambo_detjacobian' in self.parton_list:
            print("Load phasespace_rambo_detjacobian")
            self.phasespace_rambo_detjacobian = torch.load(
                self.processed_file_names("phasespace_rambo_detjacobian"))
    
        if 'logScaled_data_higgs_t_tbar_ISR_cartesian' in self.parton_list or 'mean_log_data_higgs_t_tbar_ISR_cartesian' in self.parton_list:
            print("Load logScaled_data_higgs_t_tbar_ISR_cartesian")
            self.mean_log_data_higgs_t_tbar_ISR_cartesian, self.std_log_data_higgs_t_tbar_ISR_cartesian = torch.load(
                self.processed_file_names("Log_mean_std_H_thad_tlep_ISR_cartesian"))
            self.logScaled_data_higgs_t_tbar_ISR_cartesian = torch.load(
                self.processed_file_names("LogScaled_H_thad_tlep_ISR_cartesian"))

        if 'logScaled_data_higgs_t_tbar_ISR' in self.parton_list or 'mean_log_data_higgs_t_tbar_ISR' in self.parton_list:
            print("Load logScaled_data_higgs_t_tbar_ISR")
            self.mean_log_data_higgs_t_tbar_ISR, self.std_log_data_higgs_t_tbar_ISR = torch.load(
                self.processed_file_names("Log_mean_std_H_thad_tlep_ISR"))
            self.logScaled_data_higgs_t_tbar_ISR = torch.load(
                self.processed_file_names("LogScaled_H_thad_tlep_ISR"))

        if dev==torch.device('cuda') and torch.cuda.is_available():
            print("Parton: Move tensors to GPU memory")
            for field in self.parton_list:
                setattr(self, field, getattr(self, field).to(dev)) # move elements from reco_list to GPU memory
            
        if dtype != None:
            for field in self.parton_list:
                setattr(self, field, getattr(self, field).to(dtype)) # move elements from reco_list to GPU memory

    @property
    def raw_file_names(self):
        return [self.root]

    def processed_file_names(self, type):
        return (self.rootDir + '/processed_partons/' + type + '_data.pt')

    def get_PartonsAndLeptonsAndBoost(self):
        for file in self.raw_file_names:
            df = ak.from_parquet(file)

            generator = df["generator_info"]

            incoming_particles_boost = self.get_incoming_particles_boost(
                generator)

            boost = incoming_particles_boost

            partons = df["partons"]
            partons = ak.with_name(partons, name="Momentum4D")

            gluon = partons[partons.prov == 4]
            gluon = self.Reshape(gluon, utils.struct_gluon, 1)[:, 0]

            leptons = df["lepton_partons"]
            leptons = ak.with_name(leptons, name="Momentum4D")

            higgs = df["higgs"]
            higgs = ak.with_name(higgs, name="Momentum4D")[:,0] # for the new dataset format

            partons_boosted = self.boost_CM(partons, boost)
            leptons_boosted = self.boost_CM(leptons, boost)
            higgs_boosted = self.boost_CM(higgs, boost)
            gluon_ISR_boosted = self.boost_CM(gluon, boost)

        return partons_boosted, leptons_boosted, higgs_boosted, generator, gluon_ISR_boosted, boost

    def get_incoming_particles_boost(self, generator):

        for file in self.raw_file_names:

            x1_numpy = generator.x1.to_numpy()
            x2_numpy = generator.x2.to_numpy()

            pz = (x1_numpy - x2_numpy) * 6500.0
            E = (x1_numpy + x2_numpy) * 6500.0
            zeros = np.zeros(pz.shape)

            boost = ak.Array(
                {
                    "x": zeros,
                    "y": zeros,
                    "z": pz,
                    "t": E,
                }
            )

            boost = ak.with_name(boost, name="Momentum4D")

        return boost

    def boost_CM(self, objects_array, boost):

        objects_CM = objects_array.boost_p4(boost.neg3D)

        # Overwriting old pt by calling the function on the boosted object
        # overwrite objects_array because i dont like "objects_CM.type"
        objects_array["pt"] = objects_CM.pt
        objects_array["eta"] = objects_CM.eta
        objects_array["phi"] = objects_CM.phi

        return objects_array

    def Reshape(self, input, value, ax):
        max_no = ak.max(ak.num(input, axis=ax))
        input_padded = ak.pad_none(input, max_no, axis=ax)
        input_filled = ak.fill_none(input_padded, value, axis=ax)

        return input_filled

    # Get mask for object with pt = 0
    def get_mask_pt(self, objects_array):
        return objects_array[:, :, 0] > 1e-5

    def process(self, object_type):

        # Don't need to boost in CM frame because the partons/leptons are already boosted

        for file in self.raw_file_names:

            if (object_type == "boost"):
                objects = self.boost
            elif (object_type == "partons"):
                objects = self.partons_boosted
            elif (object_type == "lepton_partons"):
                objects = self.leptons_boosted

            if object_type == "partons":
                objects = self.Reshape(objects, utils.struct_partons, 1)

            d_list = utils.to_flat_numpy(
                objects, self.fields[object_type], axis=1, allow_missing=False)

            if object_type == "partons" or object_type == "lepton_partons":
                d_list = np.transpose(d_list, (0, 2, 1))
                if object_type == "partons":
                    mask = self.get_mask_pt(d_list)
                else:
                    mask = np.ones((d_list.shape[0], d_list.shape[1]))

            elif (object_type in ["boost"]):
                d_list = np.expand_dims(d_list, axis=1)
                mask = np.ones((d_list.shape[0], d_list.shape[1]))

            tensor_data = torch.tensor(d_list)
            tensor_mask = torch.tensor(mask)

            torch.save((tensor_mask, tensor_data),
                       self.processed_file_names(object_type))

    def process_intermediateParticles(self):
        higgs = self.higgs_boosted
        top_hadronic = self.get_top_hadronic()
        top_leptonic = self.get_top_leptonic()
        gluon_ISR = self.gluon_ISR

        # Don't need to boost in CM frame because the partons/leptons are already boosted

        intermediate = [higgs, top_hadronic, top_leptonic, gluon_ISR]

        for i, objects in enumerate(intermediate):
            d_list = utils.to_flat_numpy(
                objects, self.fields["H_thad_tlep_ISR"], axis=1, allow_missing=False)

            d_list = np.expand_dims(d_list, axis=1)

            if i == 0:
                intermediate_np = d_list
            else:
                intermediate_np = np.concatenate(
                    (intermediate_np, d_list), axis=1)

        tensor_data = torch.tensor(intermediate_np)
        torch.save(tensor_data, self.processed_file_names("H_thad_tlep_ISR"))

    def process_intermediateParticles_cartesian(self):
        higgs = self.higgs_boosted
        top_hadronic = self.get_top_hadronic()
        top_leptonic = self.get_top_leptonic()
        gluon_ISR = self.gluon_ISR

        # Don't need to boost in CM frame because the partons/leptons are already boosted
        intermediate = [higgs, top_hadronic, top_leptonic, gluon_ISR]

        for i, objects in enumerate(intermediate):

            objects_cartesian = self.change_to_cartesianCoordinates(objects)

            d_list = utils.to_flat_numpy(
                objects_cartesian, self.fields["H_thad_tlep_ISR_cartesian"], axis=1, allow_missing=False)

            d_list = np.expand_dims(d_list, axis=1)

            if i == 0:
                intermediate_np = d_list
            else:
                intermediate_np = np.concatenate(
                    (intermediate_np, d_list), axis=1)

        tensor_data = torch.tensor(intermediate_np)
        torch.save(tensor_data, self.processed_file_names(
            "H_thad_tlep_ISR_cartesian"))



    def get_intermediateParticles_cartesian_onShell(self):
        tensor_cartesian_onShell = self.data_higgs_t_tbar_ISR_cartesian.clone()
        mass = [125.25, 172.5, 172.5, 0.000001]
        for i in range(4):
            tensor_cartesian_onShell[:,i,0] = torch.sqrt(tensor_cartesian_onShell[:,i,1]**2 + tensor_cartesian_onShell[:,i,2]**2 + tensor_cartesian_onShell[:,i,3]**2 + mass[i]**2)

        torch.save(tensor_cartesian_onShell, self.processed_file_names(
            "H_thad_tlep_ISR_cartesian_onShell"))


    def get_PS_intermediateParticles_onShell(self):
        E_CM = 13000
        phasespace = PhaseSpace(E_CM, [21, 21], [25, 6, -6, 21], dev="cpu")

        incoming_p_boost = self.data_boost
        x1 = (incoming_p_boost[:, 0, 0] + incoming_p_boost[:, 0, 3]) / E_CM
        x2 = (incoming_p_boost[:, 0, 0] - incoming_p_boost[:, 0, 3]) / E_CM
        ps, detjinv = phasespace.get_ps_from_momenta(
            self.data_higgs_t_tbar_ISR_cartesian_onShell, x1, x2)
    
        torch.save(ps, self.processed_file_names(
            "phasespace_intermediateParticles_onShell"))
        torch.save(detjinv, self.processed_file_names("phasespace_rambo_detjacobian_onShell"))

    def get_PS_intermediateParticles(self):

        E_CM = 13000
        phasespace = PhaseSpace(E_CM, [21, 21], [25, 6, -6, 21], dev="cpu")

        incoming_p_boost = self.data_boost
        x1 = (incoming_p_boost[:, 0, 0] + incoming_p_boost[:, 0, 3]) / E_CM
        x2 = (incoming_p_boost[:, 0, 0] - incoming_p_boost[:, 0, 3]) / E_CM
        ps, detjinv = phasespace.get_ps_from_momenta(
            self.data_higgs_t_tbar_ISR_cartesian, x1, x2)
    
        torch.save(ps, self.processed_file_names(
            "phasespace_intermediateParticles"))
        torch.save(detjinv, self.processed_file_names("phasespace_rambo_detjacobian"))
        
        
    def change_to_cartesianCoordinates(self, objects):
        px_numpy = objects.px.to_numpy()
        py_numpy = objects.py.to_numpy()
        pz_numpy = objects.pz.to_numpy()
        E_numpy = objects.E.to_numpy()

        objects_cartesian = ak.Array(
            {
                "px": px_numpy,
                "py": py_numpy,
                "pz": pz_numpy,
                "E": E_numpy,
            }
        )

        objects_cartesian = ak.with_name(
            objects_cartesian, name="Momentum4D")

        return objects_cartesian
    
    def ProcessCartesianScaled(self):
        intermediateParticles = self.data_higgs_t_tbar_ISR_cartesian        
        log_intermediateParticles = torch.sign(intermediateParticles)*torch.log(1+torch.abs(intermediateParticles))
        
        mean_LogIntermediateParticles = torch.mean(log_intermediateParticles, dim=(0,1))
        std_LogIntermediateParticles = torch.std(log_intermediateParticles, dim=(0,1))
        
        scaledIntermediateParticles = \
            (log_intermediateParticles - mean_LogIntermediateParticles[None,None,:])/std_LogIntermediateParticles[None,None,:]
        
        #torch.save(log_intermediateParticles, self.processed_file_names(
        #    "Log_H_thad_tlep_ISR_cartesian"))
        torch.save((mean_LogIntermediateParticles, std_LogIntermediateParticles), self.processed_file_names(
            "Log_mean_std_H_thad_tlep_ISR_cartesian"))
        torch.save(scaledIntermediateParticles, self.processed_file_names(
            "LogScaled_H_thad_tlep_ISR_cartesian"))

    def ProcessScaled(self):
        intermediateParticles = self.data_higgs_t_tbar_ISR
        log_intermediateParticles = intermediateParticles[:,:,:3]

        pt = intermediateParticles[:,:,0]
        log_pt = torch.log(1+pt)
        log_intermediateParticles[:,:,0] = log_pt
        
        mean_LogIntermediateParticles = torch.mean(log_intermediateParticles, dim=(0,1))
        std_LogIntermediateParticles = torch.std(log_intermediateParticles, dim=(0,1))
        
        scaledIntermediateParticles = \
            (log_intermediateParticles - mean_LogIntermediateParticles[None,None,:])/std_LogIntermediateParticles[None,None,:]
        
        #torch.save(log_intermediateParticles, self.processed_file_names(
        #    "Log_H_thad_tlep_ISR"))
        torch.save((mean_LogIntermediateParticles, std_LogIntermediateParticles), self.processed_file_names(
            "Log_mean_std_H_thad_tlep_ISR"))
        torch.save(scaledIntermediateParticles, self.processed_file_names(
            "LogScaled_H_thad_tlep_ISR"))
        
        
    def get_Higgs(self):
        partons = self.partons_boosted

        # find partons with provenance 1 (b from Higgs decay)
        prov1_partons = partons[partons.prov == 1]

        higgs = prov1_partons[:, 0] + prov1_partons[:, 1]

        return higgs

    def get_W_hadronic(self):
        partons = self.partons_boosted

        # find partons with provenance 5 (quarks from W hadronic decay)
        prov5_partons = partons[partons.prov == 5]

        W = prov5_partons[:, 0] + prov5_partons[:, 1]

        return W

    def get_top_hadronic(self):
        partons = self.partons_boosted

        # find partons with provenance 2 (b from top hadronic decay)
        prov2_partons = partons[partons.prov == 2]
        W = self.get_W_hadronic()
        top_hadron = W + prov2_partons[:, 0]

        return top_hadron

    def get_W_leptonic(self):
        leptons = self.leptons_boosted

        # sum neutrino and lepton
        W = leptons[:, 0] + leptons[:, 1]

        return W

    def get_top_leptonic(self):
        partons = self.partons_boosted

        W = self.get_W_leptonic()
        # find partons with provenance 3 (b from top leptonic decay)
        prov3_partons = partons[partons.prov == 3]

        top_leptonic = W + prov3_partons[:, 0]

        return top_leptonic

    def __getitem__(self, index):
                    

        if self.debug == True:
            return (self.mask_partons[index], self.data_partons[index],
                    self.mask_lepton_partons[index], self.data_lepton_partons[index],
                    self.mask_boost[index], self.data_boost[index],
                    self.data_higgs_t_tbar_ISR[index], self.data_higgs_t_tbar_ISR_cartesian[index],
                    self.phasespace_intermediateParticles[index],
                    self.phasespace_rambo_detjacobian[index],
                    self.log_data_higgs_t_tbar_ISR_cartesian[index],
                    # no index for mean/std because size is [4]
                    self.mean_log_data_higgs_t_tbar_ISR_cartesian, self.std_log_data_higgs_t_tbar_ISR_cartesian,
                    self.logScaled_data_higgs_t_tbar_ISR_cartesian[index])
        
        
        return [getattr(self, field)[index] if field != 'mean_log_data_higgs_t_tbar_ISR_cartesian' \
                    and field != 'std_log_data_higgs_t_tbar_ISR_cartesian' \
                    else getattr(self, field) for field in self.parton_list]

    def __len__(self):
        size = len(self.mask_partons)
        return size
