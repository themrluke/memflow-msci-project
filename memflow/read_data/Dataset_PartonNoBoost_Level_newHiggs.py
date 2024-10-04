from memflow.read_data import utils
import os
import numpy.ma as ma
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import awkward as ak

import torch
import numpy as np
import hist
from numba import njit
from scipy import stats
#torch.set_default_dtype(torch.double)

from .utils import get_weight

M_HIGGS = 125.25
M_TOP = 172.5
M_GLUON = 1e-3

class Dataset_PartonLevel_NoBoost_newHiggs(Dataset):
    def __init__(self, root, object_types=["partons", "lepton_partons", "boost",
                                           "H_thad_tlep_ISR", "H_thad_tlep_ISR_cartesian"], dev=None, debug=False,
                 dtype=None, build=False, parton_list=[]):

        self.fields = {
            "partons": ["pt", "eta", "phi", "mass", "pdgId", "prov"],
            "boost": ["t", "x", "y", "z"],
            "incoming_particles_boost": ["t", "x", "y", "z"],
            "lepton_partons": ["pt", "eta", "phi", "mass", "pdgId"],
            "H_thad_tlep_ISR": ["pt", "eta", "phi", "mass"],
            "scaled_partons": ["pt", "eta", "phi", "E", "px", "py", "pz", "prov"],
            "scaled_leptons": ["pt", "eta", "phi", "E", "px", "py", "pz", "pdgId"],
            "H_thad_tlep_ISR_cartesian": ["E", "px", "py", "pz"]
        }

        tensors_bydefault = ['mask_partons', 'data_partons', 'mask_lepton_partons', 'data_lepton_partons',
                            'mask_boost', 'data_boost', 'data_higgs_t_tbar_ISR', 'data_higgs_t_tbar_ISR_cartesian',
                            'data_higgs_t_tbar_ISR_cartesian_onShell']
        
        print("PartonLevel LAB")
        self.debug = debug

        self.root = root
        if root.endswith(".parquet"):
            self.rootDir = root.replace(".parquet","")
        else:
            self.rootDir = root

        if not os.path.exists(self.rootDir):
            build=True

        self.parton_list = parton_list
        self.dir_name = '/processed_partonsNoBoost_newHiggs/'
        print(self.rootDir + self.dir_name)
        os.makedirs(self.rootDir + self.dir_name, exist_ok=True)
        self.object_types = object_types
        self.build = build

        # if build flag set or number of files in processed partons directory is 0
        if (build or len(os.listdir(self.rootDir + self.dir_name)) == 0):

            (self.partons, self.leptons,
            self.higgs, self.generator,
            self.gluon_ISR, self.boost) = self.get_PartonsAndLeptons()

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


            print("Create intermediate with b partons attached")
            self.process_intermediateParticles_And_BPartons()

            print("Create flattening weight")
            self.get_weight_flatetas()

            print("Create new file for data_higgs_t_tbar_ISR_cartesian_onShell")
            self.get_intermediateParticles_cartesian_onShell()

            self.data_higgs_t_tbar_ISR_cartesian_onShell = torch.load(
                                    self.processed_file_names("H_thad_tlep_ISR_cartesian_onShell"))

            print("Create new file for Log_H_thad_tlep_ISR_cartesian")
            self.ProcessCartesianScaled()

            print("Create new file for Log_H_thad_tlep_ISR")
            self.ProcessScaled()
            
            print("Add log_scaled_Energy in the Log_H_thad_tlep_ISR")
            self.logScaled_data_higgs_t_tbar_ISR = torch.load(
                self.processed_file_names("LogScaled_H_thad_tlep_ISR"))
            self.logScaled_data_higgs_t_tbar_ISR_phiScaled = torch.load(
                self.processed_file_names("LogScaled_H_thad_tlep_ISR_phiScaled"))
            
            self.logScaled_data_higgs_t_tbar_ISR_cartesian = torch.load(
                self.processed_file_names("LogScaled_H_thad_tlep_ISR_cartesian"))
            
            # Add Energy as the last feature: ["pt", "eta", "phi", "Energy"]
            logScaled_data_higgs_t_tbar_ISR_withEnergy = \
                torch.cat((self.logScaled_data_higgs_t_tbar_ISR, self.logScaled_data_higgs_t_tbar_ISR_cartesian[...,0].unsqueeze(dim=2)), dim=2)
                          
            print("Create new file for logScaled_data_higgs_t_tbar_ISR_withEnergy")
            torch.save(logScaled_data_higgs_t_tbar_ISR_withEnergy,
                       self.processed_file_names("logScaled_data_higgs_t_tbar_ISR_withEnergy"))

            print("Create new file for ScaledDecayProducts")
            self.ProcessScaledDecayProducts()

            self.mean_log_data_higgs_t_tbar_ISR, self.std_log_data_higgs_t_tbar_ISR = torch.load(
                self.processed_file_names("Log_mean_std_H_thad_tlep_ISR"))

            print("Create new file for ScaledDecayProducts_ptcut")
            self.ProcessScaledDecayProducts_ptcut()

            print("Create new file for angles in the CM frame")
            self.ProcessAnglesInCMFrame()
            

        print("Reading parton_level Files")

        self.mask_partons, self.data_partons = torch.load(
            self.processed_file_names("partons"))
        self.mask_lepton_partons, self.data_lepton_partons = torch.load(
            self.processed_file_names("lepton_partons"))
        self.mask_boost, self.data_boost = torch.load(
            self.processed_file_names("boost"))
        self.data_higgs_t_tbar_ISR = torch.load(
            self.processed_file_names("H_thad_tlep_ISR"))

        if "flattening_weight_HEta_tHadEta_tLepEta" in self.parton_list:
            self.flattening_weight_HEta_tHadEta_tLepEta = torch.load(
                self.processed_file_names("flattening_weight_HEta_tHadEta_tLepEta")
            )

        self.data_higgs_t_tbar_ISR_cartesian = torch.load(
            self.processed_file_names("H_thad_tlep_ISR_cartesian"))
        self.data_higgs_t_tbar_ISR_cartesian_onShell = torch.load(
            self.processed_file_names("H_thad_tlep_ISR_cartesian_onShell"))

        if 'H_thad_tlep_ISR_bPartons' in self.parton_list:
            print("Load H_thad_tlep_ISR_bPartons")
            self.H_thad_tlep_ISR_bPartons = torch.load(
                self.processed_file_names("H_thad_tlep_ISR_bPartons"))
            self.mean_log_data_higgs_t_tbar_ISR, self.std_log_data_higgs_t_tbar_ISR = torch.load(
                self.processed_file_names("Log_mean_std_H_thad_tlep_ISR"))
    
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

        if 'logScaled_data_higgs_t_tbar_ISR_phiScaled' in self.parton_list:
            print("Load logScaled_data_higgs_t_tbar_ISR_phiScaled")
            self.mean_log_data_higgs_t_tbar_ISR, self.std_log_data_higgs_t_tbar_ISR = torch.load(
                self.processed_file_names("Log_mean_std_H_thad_tlep_ISR"))
            self.logScaled_data_higgs_t_tbar_ISR_phiScaled = torch.load(
                self.processed_file_names("LogScaled_H_thad_tlep_ISR_phiScaled"))

        if 'LogScaled_bPartons' in self.parton_list or 'Log_mean_std_bPartons' in self.parton_list:
            print("Load bPartons")
            self.mean_log_bPartons, self.std_log_bPartons = torch.load(
                self.processed_file_names("Log_mean_std_bPartons"))
            self.LogScaled_bPartons = torch.load(
                self.processed_file_names("LogScaled_bPartons"))

        if 'LogScaled_partonsLeptons' in self.parton_list or 'Log_mean_std_partonsLeptons' in self.parton_list:
            print("Load LogScaled_partonsLeptons")
            self.mean_log_partonsLeptons, self.std_log_partonsLeptons = torch.load(
                self.processed_file_names("Log_mean_std_partonsLeptons"))
            self.LogScaled_partonsLeptons = torch.load(
                self.processed_file_names("LogScaled_partonsLeptons"))
            self.mask_partonsLeptons = torch.load(
                self.processed_file_names("mask_partonsLeptons"))

        if 'LogScaled_partonsLeptons_phiScaled' in self.parton_list:
            print("Load LogScaled_partonsLeptons_phiScaled")
            self.mean_log_partonsLeptons, self.std_log_partonsLeptons = torch.load(
                self.processed_file_names("Log_mean_std_partonsLeptons"))
            self.LogScaled_partonsLeptons_phiScaled = torch.load(
                self.processed_file_names("LogScaled_partonsLeptons_phiScaled"))
            self.mask_partonsLeptons = torch.load(
                self.processed_file_names("mask_partonsLeptons"))

        if 'tensor_partonsLeptons_boxcox' in self.parton_list or 'Log_mean_std_partonsLeptons_boxcox' in self.parton_list:
            print("Load tensor_partonsLeptons_boxcox")
            self.mean_log_partonsLeptons_boxcox, self.std_log_partonsLeptons_boxcox = torch.load(
                self.processed_file_names("Log_mean_std_partonsLeptons_boxcox"))
            self.tensor_partonsLeptons_boxcox = torch.load(
                self.processed_file_names("tensor_partonsLeptons_boxcox"))
            self.mask_partonsLeptons_boxcox = torch.load(
                self.processed_file_names("mask_partonsLeptons"))
            self.lambda_boxcox = torch.load(
                self.processed_file_names("lambda_boxcox"))

        if 'tensor_AllPartons' in self.parton_list:
            print("Load tensor_AllPartons")
            self.mean_log_data_higgs_t_tbar_ISR, self.std_log_data_higgs_t_tbar_ISR = torch.load(
                self.processed_file_names("Log_mean_std_H_thad_tlep_ISR"))
            self.mean_log_partonsLeptons, self.std_log_partonsLeptons = torch.load(
                self.processed_file_names("Log_mean_std_partonsLeptons"))
            self.tensor_AllPartons = torch.load(
                self.processed_file_names("tensor_AllPartons"))
            self.mask_AllPartons = torch.load(
                self.processed_file_names("mask_AllPartons"))

        if 'tensor_AllPartons_phiScaled' in self.parton_list:
            print("Load tensor_AllPartons_phiScaled")
            self.mean_log_data_higgs_t_tbar_ISR, self.std_log_data_higgs_t_tbar_ISR = torch.load(
                self.processed_file_names("Log_mean_std_H_thad_tlep_ISR"))
            self.mean_log_partonsLeptons, self.std_log_partonsLeptons = torch.load(
                self.processed_file_names("Log_mean_std_partonsLeptons"))
            self.tensor_AllPartons_phiScaled = torch.load(
                self.processed_file_names("tensor_AllPartons_phiScaled"))
            self.mask_AllPartons = torch.load(
                self.processed_file_names("mask_AllPartons"))

        if 'LogScaled_partonsLeptons_ptcut' in self.parton_list or 'Log_mean_std_partonsLeptons_ptcut' in self.parton_list:
            print("Load LogScaled_partonsLeptons")
            self.mean_log_partonsLeptons_ptcut, self.std_log_partonsLeptons_ptcut = torch.load(
                self.processed_file_names("Log_mean_std_partonsLeptons_ptcut"))
            self.LogScaled_partonsLeptons_ptcut = torch.load(
                self.processed_file_names("LogScaled_partonsLeptons_ptcut"))
            self.mask_partonsLeptons_ptcut = torch.load(
                self.processed_file_names("mask_partonsLeptons_ptcut"))

        if 'tensor_AllPartons_ptcut' in self.parton_list:
            print("Load tensor_AllPartons")
            self.mean_log_data_higgs_t_tbar_ISR_ptcut, self.std_log_data_higgs_t_tbar_ISR_ptcut = torch.load(
                self.processed_file_names("mean_std_AllPartons_ptcut"))
            self.mean_log_partonsLeptons_ptcut, self.std_log_partonsLeptons_ptcut = torch.load(
                self.processed_file_names("Log_mean_std_partonsLeptons_ptcut"))
            self.tensor_AllPartons_ptcut = torch.load(
                self.processed_file_names("tensor_AllPartons_ptcut"))
            self.mask_AllPartons_ptcut = torch.load(
                self.processed_file_names("mask_AllPartons_ptcut"))
                          
        if 'logScaled_data_higgs_t_tbar_ISR_withEnergy' in self.parton_list or 'mean_logScaled_data_higgs_t_tbar_ISR_withEnergy' in self.parton_list:
            print("Load logScaled_data_higgs_t_tbar_ISR_withEnergy")
            mean_ptEtaPhi, std_ptEtaPhi = torch.load(
                self.processed_file_names("Log_mean_std_H_thad_tlep_ISR"))
            mean_cartesian, std_cartesian = torch.load(
                self.processed_file_names("Log_mean_std_H_thad_tlep_ISR_cartesian"))
                          
            # attach mean_Energy at mean_[pt,eta,phi]
            self.mean_logScaled_data_higgs_t_tbar_ISR_withEnergy = torch.cat(
                (mean_ptEtaPhi, mean_cartesian[0].unsqueeze(dim=0)), dim=0)
            self.std_logScaled_data_higgs_t_tbar_ISR_withEnergy = torch.cat(
                (std_ptEtaPhi, std_cartesian[0].unsqueeze(dim=0)), dim=0)
                          
            self.logScaled_data_higgs_t_tbar_ISR_withEnergy = torch.load(
                self.processed_file_names("logScaled_data_higgs_t_tbar_ISR_withEnergy"))

        if 'logScaled_data_boost' in self.parton_list or 'mean_log_data_boost' in self.parton_list:
            print("Load logScaled_data_boost")
            self.mean_log_data_boost, self.std_log_data_boost = torch.load(
                self.processed_file_names("Log_mean_std_boost"))
            self.logScaled_data_boost = torch.load(
                self.processed_file_names("LogScaled_boost"))

        if 'Hb1_thad_b_q1_tlep_b_el_CM' in self.parton_list:
            print("Load Hb1_thad_b_q1_tlep_b_el_CM")
            self.mean_log_partonsLeptons, self.std_log_partonsLeptons = torch.load(
                self.processed_file_names("Log_mean_std_partonsLeptons"))
            self.Hb1_thad_b_q1_tlep_b_el_CM = torch.load(
                self.processed_file_names("Hb1_thad_b_q1_tlep_b_el_CM"))

        if dtype != None:
            for field in self.parton_list:
                setattr(self, field, getattr(self, field).to(dtype))
            for field in tensors_bydefault:
                setattr(self, field, getattr(self, field).to(dtype))

        print(f"Parton: Move tensors to device ({dev}) memory")
        for field in self.parton_list:
            setattr(self, field, getattr(self, field).to(dev)) # move elements from reco_list to GPU memory
        for field in tensors_bydefault:
            setattr(self, field, getattr(self, field).to(dev)) # move elements from reco_list to GPU memory
            
    @property
    def raw_file_names(self):
        return [self.root]

    def processed_file_names(self, type):
        return (self.rootDir + self.dir_name + type + '_data.pt')

    def get_PartonsAndLeptons(self):
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

            higgs = self.get_Higgs_bPartons(partons)
            higgs = ak.with_name(higgs, name="Momentum4D") # for the new dataset format

        return partons, leptons, higgs, generator, gluon, boost

    def get_Higgs_bPartons(self, partons):
        partons = partons

        # find partons with provenance 1 (b from Higgs decay)
        prov1_partons = partons[partons.prov == 1]

        higgs = prov1_partons[:, 0] + prov1_partons[:, 1]

        return higgs

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

    def Reshape(self, input, value, ax):
        max_no = ak.max(ak.num(input, axis=ax))
        input_padded = ak.pad_none(input, max_no, axis=ax)
        input_filled = ak.fill_none(input_padded, value, axis=ax)

        return input_filled

    # Get mask for object with pt = 0
    def get_mask_pt(self, objects_array):
        return objects_array[:, :, 0] > 1e-5

    def process(self, object_type):

        for file in self.raw_file_names:

            if (object_type == "boost"):
                objects = self.boost
            elif (object_type == "partons"):
                objects = self.partons
            elif (object_type == "lepton_partons"):
                objects = self.leptons

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
        higgs = self.higgs
        top_hadronic = self.get_top_hadronic()
        top_leptonic = self.get_top_leptonic()
        gluon_ISR = self.gluon_ISR

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


    def process_intermediateParticles_And_BPartons(self):

        data_higgs_t_tbar_ISR = self.data_higgs_t_tbar_ISR

        # add here b jets from Higgs, thad, tlep
        # order: # H, H1, H2, thad, thad1, tlep, tlep1, ISR
        partons = self.partons

        # find partons with provenance 1 (b from Higgs decay)
        prov1_partons = partons[partons.prov == 1]
        # find partons with provenance 2 (b from top hadronic decay)
        prov2_partons = partons[partons.prov == 2]
        # find partons with provenance 3 (b from top leptonic decay)
        prov3_partons = partons[partons.prov == 3]

        b_partons = [prov1_partons, prov2_partons, prov3_partons]

        for i, objects in enumerate(b_partons):
            
            d_list = utils.to_flat_numpy(
                objects, self.fields["H_thad_tlep_ISR"], axis=1, allow_missing=False)

            d_list = np.transpose(d_list, (0, 2, 1))

            if i == 0:
                intermediate_np = d_list
            else:
                intermediate_np = np.concatenate(
                    (intermediate_np, d_list), axis=1)
                
        tensor_bPartons = torch.tensor(intermediate_np) # H1 H2 thad1 thad2

        tensor_transferFlow = torch.cat((data_higgs_t_tbar_ISR[:,0:1], # H
                                        tensor_bPartons[:,0:2],        # H1,H2
                                        data_higgs_t_tbar_ISR[:,1:2],  # thad
                                        tensor_bPartons[:,2:3],        # thad1
                                        data_higgs_t_tbar_ISR[:,2:3],  # tlep
                                        tensor_bPartons[:,3:4],        # tlep1
                                        data_higgs_t_tbar_ISR[:,3:4]), dim=1) # ISR

        torch.save(tensor_transferFlow, self.processed_file_names("H_thad_tlep_ISR_bPartons"))
        

    def process_intermediateParticles_cartesian(self):
        higgs = self.higgs
        top_hadronic = self.get_top_hadronic()
        top_leptonic = self.get_top_leptonic()
        gluon_ISR = self.gluon_ISR

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

        scaledIntermediateParticles = torch.clone(log_intermediateParticles)

        scaledIntermediateParticles_phiScaled = torch.clone(log_intermediateParticles)
        scaledIntermediateParticles_phiScaled[...,:3] = \
            (log_intermediateParticles[...,:3] - mean_LogIntermediateParticles[None,None,:])/std_LogIntermediateParticles[None,None,:]
        
        scaledIntermediateParticles[...,:2] = \
            (log_intermediateParticles[...,:2] - mean_LogIntermediateParticles[None,None,:2])/std_LogIntermediateParticles[None,None,:2]
        
        #torch.save(log_intermediateParticles, self.processed_file_names(
        #    "Log_H_thad_tlep_ISR"))
        torch.save((mean_LogIntermediateParticles, std_LogIntermediateParticles), self.processed_file_names(
            "Log_mean_std_H_thad_tlep_ISR"))
        torch.save(scaledIntermediateParticles, self.processed_file_names("LogScaled_H_thad_tlep_ISR"))
        torch.save(scaledIntermediateParticles_phiScaled, self.processed_file_names("LogScaled_H_thad_tlep_ISR_phiScaled"))

        # Scaling also the boost
        boost_output = torch.cat((torch.from_numpy(self.boost.E.to_numpy()).unsqueeze(1),
                                  torch.from_numpy(self.boost.pz.to_numpy()).unsqueeze(1)), 1)
        boost_output[:,0] = torch.log(1+boost_output[:,0])
        mean_boost = torch.mean(boost_output, dim=0)
        std_boost = torch.std(boost_output, dim=0)
        scaled_boost_output = (boost_output - mean_boost)/std_boost
        torch.save((mean_boost, std_boost), self.processed_file_names("Log_mean_std_boost"))
        torch.save(scaled_boost_output, self.processed_file_names("LogScaled_boost"))

         # add here b jets from Higgs, thad, tlep
        # order: # H, H1, H2, thad, thad1, tlep, tlep1, ISR
        partons = self.partons

        # find partons with provenance 1 (b from Higgs decay)
        prov1_partons = partons[partons.prov == 1]
        # find partons with provenance 2 (b from top hadronic decay)
        prov2_partons = partons[partons.prov == 2]
        # find partons with provenance 3 (b from top leptonic decay)
        prov3_partons = partons[partons.prov == 3]

        b_partons = [prov1_partons, prov2_partons, prov3_partons]

        for i, objects in enumerate(b_partons):
            
            d_list = utils.to_flat_numpy(
                objects, self.fields["H_thad_tlep_ISR"], axis=1, allow_missing=False)

            d_list = np.transpose(d_list, (0, 2, 1))

            if i == 0:
                intermediate_np = d_list
            else:
                intermediate_np = np.concatenate(
                    (intermediate_np, d_list), axis=1)
                
        tensor_bPartons = torch.tensor(intermediate_np[:,:,:3]) # H1 H2 thad1 thad2
        tensor_bPartons[:,:,0] = torch.log(1 + tensor_bPartons[:,:,0])

        mean_tensor_bPartons = torch.mean(tensor_bPartons, dim=(0,1))
        std_tensor_bPartons = torch.std(tensor_bPartons, dim=(0,1))
        
        scaled_tensor_bPartons = \
            (tensor_bPartons - mean_tensor_bPartons[None,None,:])/std_tensor_bPartons[None,None,:]
        
        torch.save((mean_tensor_bPartons, std_tensor_bPartons), self.processed_file_names(
            "Log_mean_std_bPartons"))
        torch.save(scaled_tensor_bPartons, self.processed_file_names(
            "LogScaled_bPartons"))

    def ProcessScaledDecayProducts(self):
        
        # add here b jets from Higgs, thad, tlep
        # order: # H, H1, H2, thad, thad1, tlep, tlep1, ISR
        partons = self.partons

        # first position always massive leptons, 2nd position = neutrino
        leptons = self.leptons

        # find partons with provenance 1 (b from Higgs decay)
        prov1_partons = partons[partons.prov == 1]
        # find partons with provenance 2 (b from top hadronic decay)
        prov2_partons = partons[partons.prov == 2]
        # find partons with provenance 3 (b from top leptonic decay)
        prov3_partons = partons[partons.prov == 3]
        # find partons with provenance 4 (ISR)
        # sometimes there are events without ISR -> paddinng = partons with prov=-1 which are null
        prov4_partons = partons[partons.prov == 4]
        # find partons with provenance 5 (light q)
        prov5_partons = partons[partons.prov == 5]
        # find partons with provenance -1 (not matched) -> I don't need to add this in data
        # because I will mask the missing partons (which are the ones with ISR)

        partons = [prov1_partons, prov2_partons, prov3_partons, prov5_partons, prov4_partons, leptons]

        for i, objects in enumerate(partons):

            if i == 5:
                # lepton case -> we don't have prov so we attach the pdgID
                d_list = utils.to_flat_numpy(
                    objects, self.fields["scaled_leptons"], axis=1, allow_missing=False)
            else:
                # case prov = 4 => ISR (sometimes missing)
                if (i == 4):
                    objects = self.Reshape(objects, utils.struct_gluon, 1)

                d_list = utils.to_flat_numpy(
                    objects, self.fields["scaled_partons"], axis=1, allow_missing=False)
    
            d_list = np.transpose(d_list, (0, 2, 1))

            if i == 0:
                intermediate_np = d_list
            else:
                intermediate_np = np.concatenate(
                    (intermediate_np, d_list), axis=1)

        tensor_partonsLeptons = torch.tensor(intermediate_np) # H1 H2 thad1 thad2
        tensor_partonsLeptons_boxcox = torch.clone(tensor_partonsLeptons)
        mask_partonsLeptons = (tensor_partonsLeptons[...,3] != -1).bool() # pt > 0
        tensor_partonsLeptons[:,:,0] = torch.log(1 + tensor_partonsLeptons[:,:,0]) # take log of pt
        tensor_partonsLeptons[:,:,3:7] = torch.sign(tensor_partonsLeptons[:,:,3:7])*torch.log(1 + torch.abs(tensor_partonsLeptons[:,:,3:7])) # take log of E/px/py/pz

        mean_list = []
        std_list = []
        for i in range(7):
            feature = tensor_partonsLeptons[...,i]
            mean_feature = torch.mean(feature[mask_partonsLeptons]) # masked mean
            std_feature = torch.std(feature[mask_partonsLeptons]) # masked std
            mean_list.append(mean_feature)
            std_list.append(std_feature)

        mean_tensor_partonsLeptons = torch.Tensor(mean_list)
        std_tensor_partonsLeptons = torch.Tensor(std_list)

        parton_type = [1]* 7
        parton_type = [*parton_type, 2, 2] # 1 for jets, 2 for leptons
        parton_type = torch.Tensor(parton_type).unsqueeze(dim=1)
        parton_type = parton_type.unsqueeze(dim=0).repeat(tensor_partonsLeptons.size(0),1,1)

        tensor_partonsLeptons = torch.cat((tensor_partonsLeptons, parton_type), dim=2)

        tensor_partonsLeptons_phiScaled = torch.clone(tensor_partonsLeptons[...,[0,1,2,7,8]]) # DON'T INCLUDE E/PX/PY/PZ
        tensor_partonsLeptons_phiScaled[:,:,:3] = \
            (tensor_partonsLeptons[:,:,:3] - mean_tensor_partonsLeptons[None,None,:3])/std_tensor_partonsLeptons[None,None,:3]

        # scale everything except phi: pt/eta/E/px/py/pz
        tensor_partonsLeptons[:,:,[0,1,3,4,5,6]] = \
            (tensor_partonsLeptons[:,:,[0,1,3,4,5,6]] - mean_tensor_partonsLeptons[None,None,[0,1,3,4,5,6]])/std_tensor_partonsLeptons[None,None,[0,1,3,4,5,6]]

        # sort higgs and thad decay products (based on pt)
        tensor_partonsLeptons = self.sort_pt(tensor_partonsLeptons)
        # I don't need to modify the mask cuz ISR is in the same position
        
        torch.save((mean_tensor_partonsLeptons, std_tensor_partonsLeptons), self.processed_file_names(
            "Log_mean_std_partonsLeptons"))
        torch.save(tensor_partonsLeptons_phiScaled, self.processed_file_names("LogScaled_partonsLeptons_phiScaled"))
        torch.save(tensor_partonsLeptons, self.processed_file_names(
            "LogScaled_partonsLeptons"))
        torch.save(mask_partonsLeptons, self.processed_file_names(
            "mask_partonsLeptons"))

        # version with AllPartons
        # structure: [b_jet_index (like in decay_products), type]
        # type = 1 for jets, 2 for lepton/MET, 3 for high level parton (Higgs, t, tbar)
        parton_type = torch.Tensor([[[1,3], [2,3], [3,3], [4,3]]]).repeat(self.logScaled_data_higgs_t_tbar_ISR.size(0), 1, 1)
        
        # attach parton type to our common higgs/tops tensor + attach E/px/py/pz scaled
        self.logScaled_data_higgs_t_tbar_ISR = torch.cat((self.logScaled_data_higgs_t_tbar_ISR, self.logScaled_data_higgs_t_tbar_ISR_cartesian, parton_type), dim=2)
        self.logScaled_data_higgs_t_tbar_ISR_phiScaled = torch.cat((self.logScaled_data_higgs_t_tbar_ISR_phiScaled, parton_type), dim=2)

        # merge decay products and partons
        tensor_AllPartons = torch.cat((tensor_partonsLeptons, self.logScaled_data_higgs_t_tbar_ISR[:,:-1]), dim=1)
        tensor_AllPartons_phiScaled = torch.cat((tensor_partonsLeptons_phiScaled, self.logScaled_data_higgs_t_tbar_ISR_phiScaled[:,:-1]), dim=1)

        # mask because sometimes the ISR is missing
        mask_HttbarISR = torch.ones((self.logScaled_data_higgs_t_tbar_ISR[:,:-1].shape[:2]))
        mask_AllPartons = torch.cat((mask_partonsLeptons, mask_HttbarISR), dim=1)

        # sort higgs and thad decay products (based on pt)
        tensor_AllPartons = self.sort_pt(tensor_AllPartons)

        torch.save(tensor_AllPartons, self.processed_file_names("tensor_AllPartons"))
        torch.save(tensor_AllPartons_phiScaled, self.processed_file_names("tensor_AllPartons_phiScaled"))
        torch.save(mask_AllPartons, self.processed_file_names("mask_AllPartons"))
        
        # version with pt box cox
        pt_boxcox, lambda_boxcox = stats.boxcox(tensor_partonsLeptons_boxcox[...,0].flatten())
        tensor_partonsLeptons_boxcox[...,0] = torch.reshape(torch.Tensor(pt_boxcox), (tensor_partonsLeptons_boxcox.shape[:2]))

        mean_list = []
        std_list = []
        for i in range(3):
            feature = tensor_partonsLeptons_boxcox[...,i]
            mean_feature = torch.mean(feature[mask_partonsLeptons]) # masked mean
            std_feature = torch.std(feature[mask_partonsLeptons]) # masked std
            mean_list.append(mean_feature)
            std_list.append(std_feature)

        mean_tensor_partonsLeptons = torch.Tensor(mean_list)
        std_tensor_partonsLeptons = torch.Tensor(std_list)

        parton_type = [1]* 7
        parton_type = [*parton_type, 2, 2] # 1 for jets, 2 for leptons
        parton_type = torch.Tensor(parton_type).unsqueeze(dim=1)
        parton_type = parton_type.unsqueeze(dim=0).repeat(tensor_partonsLeptons_boxcox.size(0),1,1)

        tensor_partonsLeptons_boxcox = torch.cat((tensor_partonsLeptons_boxcox, parton_type), dim=2)
        
        tensor_partonsLeptons_boxcox[:,:,:3] = \
            (tensor_partonsLeptons_boxcox[:,:,:3] - mean_tensor_partonsLeptons[None,None,:3])/std_tensor_partonsLeptons[None,None,:3]

        torch.save((mean_tensor_partonsLeptons, std_tensor_partonsLeptons), self.processed_file_names(
            "Log_mean_std_partonsLeptons_boxcox"))
        torch.save(tensor_partonsLeptons_boxcox, self.processed_file_names(
            "tensor_partonsLeptons_boxcox"))
        torch.save(torch.Tensor([lambda_boxcox]), self.processed_file_names(
            "lambda_boxcox"))

    def ProcessScaledDecayProducts_ptcut(self):
        
        # add here b jets from Higgs, thad, tlep
        # order: # H, H1, H2, thad, thad1, tlep, tlep1, ISR
        partons = self.partons

        # first position always massive leptons, 2nd position = neutrino
        leptons = self.leptons

        # find partons with provenance 1 (b from Higgs decay)
        prov1_partons = partons[partons.prov == 1]
        # find partons with provenance 2 (b from top hadronic decay)
        prov2_partons = partons[partons.prov == 2]
        # find partons with provenance 3 (b from top leptonic decay)
        prov3_partons = partons[partons.prov == 3]
        # find partons with provenance 4 (ISR)
        # sometimes there are events without ISR -> paddinng = partons with prov=-1 which are null
        prov4_partons = partons[partons.prov == 4]
        # find partons with provenance 5 (light q)
        prov5_partons = partons[partons.prov == 5]
        # find partons with provenance -1 (not matched) -> I don't need to add this in data
        # because I will mask the missing partons (which are the ones with ISR)

        partons = [prov1_partons, prov2_partons, prov3_partons, prov5_partons, prov4_partons, leptons]

        for i, objects in enumerate(partons):

            if i == 5:
                # lepton case -> we don't have prov so we attach the pdgID
                d_list = utils.to_flat_numpy(
                    objects, self.fields["scaled_leptons"], axis=1, allow_missing=False)
            else:
                # case prov = 4 => ISR (sometimes missing)
                if (i == 4):
                    objects = self.Reshape(objects, utils.struct_gluon, 1)

                d_list = utils.to_flat_numpy(
                    objects, self.fields["scaled_partons"], axis=1, allow_missing=False)
    
            d_list = np.transpose(d_list, (0, 2, 1))

            if i == 0:
                intermediate_np = d_list
            else:
                intermediate_np = np.concatenate(
                    (intermediate_np, d_list), axis=1)

        tensor_partonsLeptons = torch.tensor(intermediate_np) # H1 H2 thad1 thad2
        mask_partonsLeptons = (tensor_partonsLeptons[...,3] != -1).bool() # pt > 0
        for i in range(tensor_partonsLeptons.shape[1]):
            tensor_partonsLeptons[:,i,0][mask_partonsLeptons[:,i]] = tensor_partonsLeptons[:,i,0][mask_partonsLeptons[:,i]] - torch.min(tensor_partonsLeptons[:,i,0][mask_partonsLeptons[:,i]])
        
        tensor_partonsLeptons[:,:,0] = torch.log(1 + tensor_partonsLeptons[:,:,0]) # take log of masked
        # I don't care because I will mask these elems when I do any computations

        mean_list = []
        std_list = []
        for i in range(3):
            feature = tensor_partonsLeptons[...,i]
            mean_feature = torch.mean(feature[mask_partonsLeptons]) # masked mean
            std_feature = torch.std(feature[mask_partonsLeptons]) # masked std
            mean_list.append(mean_feature)
            std_list.append(std_feature)

        mean_tensor_partonsLeptons = torch.Tensor(mean_list)
        std_tensor_partonsLeptons = torch.Tensor(std_list)

        parton_type = [1]* 7
        parton_type = [*parton_type, 2, 2] # 1 for jets, 2 for leptons
        parton_type = torch.Tensor(parton_type).unsqueeze(dim=1)
        parton_type = parton_type.unsqueeze(dim=0).repeat(tensor_partonsLeptons.size(0),1,1)

        tensor_partonsLeptons = torch.cat((tensor_partonsLeptons, parton_type), dim=2)
        
        tensor_partonsLeptons[:,:,:2] = \
            (tensor_partonsLeptons[:,:,:2] - mean_tensor_partonsLeptons[None,None,:2])/std_tensor_partonsLeptons[None,None,:2]

        # sort higgs and thad decay products (based on pt)
        tensor_partonsLeptons = self.sort_pt(tensor_partonsLeptons)
        
        torch.save((mean_tensor_partonsLeptons, std_tensor_partonsLeptons), self.processed_file_names(
            "Log_mean_std_partonsLeptons_ptcut"))
        torch.save(tensor_partonsLeptons, self.processed_file_names(
            "LogScaled_partonsLeptons_ptcut"))
        torch.save(mask_partonsLeptons, self.processed_file_names(
            "mask_partonsLeptons_ptcut"))

        # THIS PART BELOW IS REDUNDANT BECAUSE self.logScaled_data_higgs_t_tbar_ISR was already updated in the 
        # method 'ProcessScaledDecayProducts' -> don't need to concatenate type anymore
        
        # version with AllPartons
        # structure: [b_jet_index (like in decay_products), type]
        # type = 1 for jets, 2 for lepton/MET, 3 for high level parton (Higgs, t, tbar)
        #parton_type = torch.Tensor([[[1,3], [2,3], [3,3], [4,3]]]).repeat(self.logScaled_data_higgs_t_tbar_ISR.size(0), 1, 1)
        # attach parton type to our common higgs/tops tensor
        #self.logScaled_data_higgs_t_tbar_ISR = torch.cat((self.logScaled_data_higgs_t_tbar_ISR, parton_type), dim=2)

        # merge decay products and partons
        tensor_AllPartons = torch.cat((tensor_partonsLeptons, self.logScaled_data_higgs_t_tbar_ISR[:,:-1]), dim=1)

        # take intermediate partons -> unscale pt -> substract min pt -> scale back
        tensor_AllPartons[:, -3:, 0] =  tensor_AllPartons[:, -3:, 0]*self.std_log_data_higgs_t_tbar_ISR[0] + self.mean_log_data_higgs_t_tbar_ISR[0]
        tensor_AllPartons[:, -3:, 0] = torch.exp(tensor_AllPartons[:, -3:, 0]) - 1

        # substract min from intermediate partons pt
        for i in range(3):
            tensor_AllPartons[:, -3+i:, 0] = tensor_AllPartons[:, -3+i:, 0] - torch.min(tensor_AllPartons[:, -3+i:, 0])

        # log pt
        tensor_AllPartons[:,-3:,0] = torch.log(1 + tensor_AllPartons[:,-3:,0]) # take log of masked
        # I don't care because I will mask these elems when I do any computations

        # new mean and std of pt/eta/phi
        mean_list = []
        std_list = []
        for i in range(3):
            feature = tensor_AllPartons[...,-3:,i]
            mean_feature = torch.mean(feature) # masked mean
            std_feature = torch.std(feature) # masked std
            mean_list.append(mean_feature)
            std_list.append(std_feature)

        mean_tensor_allPartons = torch.Tensor(mean_list)
        std_tensor_AllPartons = torch.Tensor(std_list)

        # scale results
        tensor_AllPartons[:,-3:,:2] = \
            (tensor_AllPartons[:,-3:,:2] - mean_tensor_allPartons[None,None,:2])/mean_tensor_allPartons[None,None,:2]

        # mask because sometimes the ISR is missing
        mask_HttbarISR = torch.ones((self.logScaled_data_higgs_t_tbar_ISR[:,:-1].shape[:2]))
        mask_AllPartons = torch.cat((mask_partonsLeptons, mask_HttbarISR), dim=1)

        # sort higgs and thad decay products (based on pt)
        tensor_AllPartons = self.sort_pt(tensor_AllPartons)

        torch.save(tensor_AllPartons, self.processed_file_names("tensor_AllPartons_ptcut"))
        torch.save(mask_AllPartons, self.processed_file_names("mask_AllPartons_ptcut"))
        torch.save((mean_tensor_allPartons, std_tensor_AllPartons), self.processed_file_names("mean_std_AllPartons_ptcut"))

    def boost_CM(self, objects_array, boost):

        objects_CM = objects_array.boost_p4(boost.neg3D)

        # Overwriting old pt by calling the function on the boosted object
        # overwrite objects_array because i dont like "objects_CM.type"
        objects_array["rho"] = objects_CM.pt
        objects_array["pt"] = objects_CM.pt
        objects_array["eta"] = objects_CM.eta
        objects_array["phi"] = objects_CM.phi

        return ak.with_name(objects_array, "Momentum4D")

    def ProcessAnglesInCMFrame(self):

        higgs = self.get_Higgs()
        thad = self.get_top_hadronic()
        tlep = self.get_top_leptonic()
        W_had = self.get_W_hadronic()
        W_lep = self.get_W_leptonic()

        # find partons with provenance 1 (b from Higgs decay)
        prov1_partons = self.partons[self.partons.prov == 1]
        # find partons with provenance 2 (b from top hadronic decay)
        prov2_partons = self.partons[self.partons.prov == 2]
        # find partons with provenance 3 (b from top leptonic decay)
        prov3_partons = self.partons[self.partons.prov == 3]
        # find partons with provenance 5 (light q from top had decay)
        prov5_partons = self.partons[self.partons.prov == 5]

        leptons = self.leptons

        # higgs b1 and b2
        index_sort = ak.argsort(prov1_partons.pt, axis=-1, ascending=False, stable=True)
        prov1_partons_sorted = prov1_partons[index_sort]

        # check if Higgs b partons are sorted by pt
        mask_pt = prov1_partons_sorted.pt[:,0] < prov1_partons_sorted.pt[:,1]
        if ak.any(mask_pt):
            raise Exception("Higgs b1 and b2 not sorted by pt")

        # boost first parton in the CM
        prov1_partons_sorted_CM = self.boost_CM(prov1_partons_sorted, higgs)
        check_CM = prov1_partons_sorted_CM[:,0] + prov1_partons_sorted_CM[:,1]

        # check if Higgs b partons were corrected boosted in CM
        mask_px = abs(check_CM.px) > 1e-5
        mask_py = abs(check_CM.py) > 1e-5
        mask_pz = abs(check_CM.pz) > 1e-5

        if (ak.any(mask_px) or ak.any(mask_py) or ak.any(mask_pz)):
            raise Exception("Higgs b1 and b2 not in CM")

        # transform in tensor
        H_b1 = prov1_partons_sorted_CM[:,0]
        H_b1_tensor = utils.to_flat_tensor(H_b1, ['pt', 'eta', 'phi'], axis=1, allow_missing=False)

        # light quarks
        index_sort = ak.argsort(prov5_partons.pt, axis=-1, ascending=False, stable=True)
        prov5_partons_sorted = prov5_partons[index_sort]

        # check if thad q partons are sorted by pt
        mask_pt = prov5_partons_sorted.pt[:,0] < prov5_partons_sorted.pt[:,1]
        if ak.any(mask_pt):
            raise Exception("thad q1 and q2 not sorted by pt")

        # boost first parton in the CM
        prov5_partons_sorted_CM = self.boost_CM(prov5_partons_sorted, W_had)

        check_CM = prov5_partons_sorted_CM[:,0] + prov5_partons_sorted_CM[:,1]

        # check if Higgs b partons were corrected boosted in CM
        mask_px = abs(check_CM.px) > 1
        mask_py = abs(check_CM.py) > 1
        mask_pz = abs(check_CM.pz) > 1

        if (ak.any(mask_px) or ak.any(mask_py) or ak.any(mask_pz)):
            raise Exception("thad q1 and q2 not in CM")

        # transform in tensor
        thad_q1 = prov5_partons_sorted_CM[:,0]
        thad_q1_tensor = utils.to_flat_tensor(thad_q1, ['pt', 'eta', 'phi'], axis=1, allow_missing=False)

        # LEPTONS: check if first lepton = el/muon   
        mask_1 = abs(leptons[:,0].pdgId) == 11
        mask_2 = abs(leptons[:,0].pdgId) == 13

        if ak.count_nonzero(mask_1) + ak.count_nonzero(mask_2) != ak.num(mask_1, axis=0):
            raise Exception("the first object is not electron or muon")

        # boost first parton in the CM
        leptons_sorted_CM = self.boost_CM(leptons, W_lep)

        check_CM = leptons_sorted_CM[:,0] + leptons_sorted_CM[:,1]

        # check if Higgs b partons were corrected boosted in CM
        mask_px = abs(check_CM.px) > 1
        mask_py = abs(check_CM.py) > 1
        mask_pz = abs(check_CM.pz) > 1

        if (ak.any(mask_px) or ak.any(mask_py) or ak.any(mask_pz)):
            raise Exception("leptons are not in CM")

        # transform in tensor
        el_mu = leptons_sorted_CM[:,0]
        el_mu_tensor = utils.to_flat_tensor(el_mu, ['pt', 'eta', 'phi'], axis=1, allow_missing=False)

        # tlep
        tlep_b_CM = self.boost_CM(prov3_partons[:,0], tlep)
        W_lep_CM = self.boost_CM(W_lep, tlep)

        check_CM = tlep_b_CM + W_lep_CM

        mask_px = abs(check_CM.px) > 1e-1
        mask_py = abs(check_CM.py) > 1e-1
        mask_pz = abs(check_CM.pz) > 1e-1

        if (ak.any(mask_px) or ak.any(mask_py) or ak.any(mask_pz)):
            raise Exception("tlep b and W not in CM")

        tlep_b_tensor = utils.to_flat_tensor(tlep_b_CM, ['pt', 'eta', 'phi'], axis=1, allow_missing=False)

        # check if thad b and W are corrected boosted in the CM
        thad_b_CM = self.boost_CM(prov2_partons[:,0], thad)
        thad_W_CM = self.boost_CM(W_had, thad)

        check_CM = thad_b_CM + thad_W_CM

        mask_px = abs(check_CM.px) > 1e-5
        mask_py = abs(check_CM.py) > 1e-5
        mask_pz = abs(check_CM.pz) > 1e-5

        if (ak.any(mask_px) or ak.any(mask_py) or ak.any(mask_pz)):
            raise Exception("thad b and W not in CM")

        thad_b_tensor = utils.to_flat_tensor(thad_b_CM, ['pt', 'eta', 'phi'], axis=1, allow_missing=False)

        Hb1_thad_b_q1_tlep_b_el_CM = torch.cat((H_b1_tensor[:,None,:], thad_b_tensor[:,None,:], thad_q1_tensor[:,None,:],
                                                tlep_b_tensor[:,None,:], el_mu_tensor[:,None,:]), dim=1)

        torch.save(Hb1_thad_b_q1_tlep_b_el_CM,  self.processed_file_names("Hb1_thad_b_q1_tlep_b_el_CM"))

    def sort_pt(self, tensor_AllPartons):
        pt_0 = tensor_AllPartons[:,0,0]
        pt_1 = tensor_AllPartons[:,1,0]

        mask_higgs = pt_0 > pt_1

        # sort higgs based on pt
        tensor_AllPartons[:,[0,1]] = torch.where(mask_higgs[:,None,None],
                                                    tensor_AllPartons[:,[0,1]],
                                                    tensor_AllPartons[:,[1,0]])

        pt_4 = tensor_AllPartons[:,4,0]
        pt_5 = tensor_AllPartons[:,5,0]
        
        mask_thad2 = pt_4 > pt_5 

        # sort thad based on pt
        tensor_AllPartons[:,[4,5]] = torch.where(mask_thad2[:,None,None],
                                                    tensor_AllPartons[:,[4,5]],
                                                    tensor_AllPartons[:,[5,4]])

        return tensor_AllPartons

        
        
    def get_Higgs(self):
        partons = self.partons

        # find partons with provenance 1 (b from Higgs decay)
        prov1_partons = partons[partons.prov == 1]

        higgs = prov1_partons[:, 0] + prov1_partons[:, 1]

        return higgs

    def get_W_hadronic(self):
        partons = self.partons

        # find partons with provenance 5 (quarks from W hadronic decay)
        prov5_partons = partons[partons.prov == 5]

        W = prov5_partons[:, 0] + prov5_partons[:, 1]

        return W

    def get_top_hadronic(self):
        partons = self.partons

        # find partons with provenance 2 (b from top hadronic decay)
        prov2_partons = partons[partons.prov == 2]
        W = self.get_W_hadronic()
        top_hadron = W + prov2_partons[:, 0]

        return top_hadron

    def get_W_leptonic(self):
        leptons = self.leptons

        # sum neutrino and lepton
        W = leptons[:, 0] + leptons[:, 1]

        return W

    def get_top_leptonic(self):
        partons = self.partons

        W = self.get_W_leptonic()
        # find partons with provenance 3 (b from top leptonic decay)
        prov3_partons = partons[partons.prov == 3]

        top_leptonic = W + prov3_partons[:, 0]

        return top_leptonic

    def get_weight_flatetas(self):
        dataCorrect = self.data_higgs_t_tbar_ISR
        higgs = dataCorrect[:,0]
        t = dataCorrect[:,1]
        tbar = dataCorrect[:,2]
        ISR = dataCorrect[:,3]
        Nbins = 25

        bins_h = np.linspace(-4,4, Nbins)
        bins_t = np.linspace(-4,4, Nbins)
        bins_tbar = np.linspace(-4,4, Nbins)
        
        h = hist.Hist(
            hist.axis.Variable( bins_h, name="h"),
            hist.axis.Variable( bins_t, name="t"),
            hist.axis.Variable( bins_tbar, name="tbar"),
        )
        h.fill(higgs[:,1], t[:,1], tbar[:,1])

        w3d = np.where(
            h.values()>0.,
            (1/ h.values()) * len(higgs)/(Nbins**3), 
            1.)
        w3d[w3d>30] = 30
        
        xind = np.digitize(higgs[:,1],  bins_h,  right=False)-1
        yind = np.digitize(t[:,1], bins_t,       right=False)-1
        zind = np.digitize(tbar[:,1], bins_tbar, right=False)-1
        index = np.stack([xind, yind, zind], axis=1)

        w = get_weight(w3d, index, Nbins)
        torch.save(torch.from_numpy(w), self.processed_file_names(
            "flattening_weight_HEta_tHadEta_tLepEta"))

    def __getitem__(self, index):
                    

        if self.debug == True:
            return (self.mask_partons[index], self.data_partons[index],
                    self.mask_lepton_partons[index], self.data_lepton_partons[index],
                    self.mask_boost[index], self.data_boost[index],
                    self.data_higgs_t_tbar_ISR[index], self.data_higgs_t_tbar_ISR_cartesian[index],
                    self.log_data_higgs_t_tbar_ISR_cartesian[index],
                    # no index for mean/std because size is [4]
                    self.mean_log_data_higgs_t_tbar_ISR_cartesian, self.std_log_data_higgs_t_tbar_ISR_cartesian,
                    self.logScaled_data_higgs_t_tbar_ISR_cartesian[index])
        
        out = [ ]
        for field in self.parton_list:
             if field not in ['mean_log_data_higgs_t_tbar_ISR_cartesian',
                              'std_log_data_higgs_t_tbar_ISR_cartesian',
                              'mean_log_data_higgs_t_tbar_ISR',
                              'std_log_data_higgs_t_tbar_ISR',
                              'mean_log_data_boost',
                              'std_log_data_boost']:
                 out.append(getattr(self, field)[index])
        return out

    def __len__(self):
        size = len(self.mask_partons)
        return size
