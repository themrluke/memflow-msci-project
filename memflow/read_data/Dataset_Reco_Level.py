from memflow.read_data import utils
import os
import os.path
import torch
import numpy as np
import awkward as ak
from torch.utils.data import Dataset




class Dataset_RecoLevel(Dataset):
    def __init__(self, root, object_types=["jets", "lepton_reco", "met", "boost"], dev=None, debug=False,
                 dtype=None, reco_list=[]):

        self.fields = {
            "jets": ["pt", "eta", "phi", "btag", "prov"],
            "lepton_reco": ["pt", "eta", "phi"],
            "met": ["pt", "eta", "phi"],
            "boost": ["x", "y", "z", "t"],
            "recoParticles_Cartesian": ["E", "px", "py", "pz"]
        }

        self.debug = debug
        self.root = root
        self.reco_list = reco_list
        os.makedirs(self.root + "/processed_jets", exist_ok=True)
        self.object_types = object_types

        allObjects = self.object_types[:]
        allObjects.append('recoParticles_Cartesian')
        
        # if an object is missing (example: jets/lepton_reco/met/boost/recoCartesian => compute boost)
        for object_type in allObjects:
            if not os.path.isfile(self.processed_file_names(object_type)):
                print("File missing: compute boost")
                self.boost = self.get_boost()
                break

        for object_type in self.object_types:
            if not os.path.isfile(self.processed_file_names(object_type)):
                print("Create new file for " + object_type)
                self.process(object_type)
            else:
                print(object_type + " file already exists")

        if not os.path.isfile(self.processed_file_names("recoParticles_Cartesian")):
                print("Create new file for recoParticles_Cartesian")
                self.processCartesian()
        else:
                print("recoParticles_Cartesian file already exists")
                
        self.mask_jets, self.data_jets = torch.load(
            self.processed_file_names("jets"))
        self.mask_lepton, self.data_lepton = torch.load(
            self.processed_file_names("lepton_reco"))
        self.mask_met, self.data_met = torch.load(
            self.processed_file_names("met"))
        self.mask_boost, self.data_boost = torch.load(
            self.processed_file_names("boost"))
        
        self.recoParticlesCartesian = torch.load(
            self.processed_file_names("recoParticles_Cartesian"))
        
        if not os.path.isfile(self.processed_file_names("recoParticlesScaled")):
                print("Create new file for recoParticlesScaled")
                self.scaleCartesianTensor()
        else:
                print("recoParticlesScaled file already exists")
        
        self.recoParticlesScaled = torch.load(
            self.processed_file_names("recoParticlesScaled"))
        
        if dev==torch.device('cuda') and torch.cuda.is_available():
            self.mask_jets, self.data_jets = self.mask_jets.to(dev), self.data_jets.to(dev)
            self.mask_lepton, self.data_lepton = self.mask_lepton.to(dev), self.data_lepton.to(dev)
            self.mask_met, self.data_met = self.mask_met.to(dev), self.data_met.to(dev)
            self.mask_boost, self.data_boost = self.mask_boost.to(dev), self.data_boost.to(dev)
            self.recoParticlesCartesian = self.recoParticlesCartesian.to(dev)
            self.recoParticlesScaled = self.recoParticlesScaled.to(dev)
            
        if dtype != None:
            self.mask_jets, self.data_jets = self.mask_jets.to(dtype), self.data_jets.to(dtype)
            self.mask_lepton, self.data_lepton = self.mask_lepton.to(dtype), self.data_lepton.to(dtype)
            self.mask_met, self.data_met = self.mask_met.to(dtype), self.data_met.to(dtype)
            self.mask_boost, self.data_boost = self.mask_boost.to(dtype), self.data_boost.to(dtype)
            self.recoParticlesCartesian = self.recoParticlesCartesian.to(dtype)
            self.recoParticlesScaled = self.recoParticlesScaled.to(dtype)
            

    @property
    def raw_file_names(self):
        return [self.root + '/all_jets.parquet']

    def processed_file_names(self, type):

        return (self.root + '/processed_jets/' + type + '_data.pt')

    def get_boost(self):

        for file in self.raw_file_names:
            df = ak.from_parquet(file)

            jets = df["jets"]
            jets = ak.with_name(jets, name="Momentum4D")

            leptons = df["lepton_reco"]
            leptons = ak.with_name(leptons, name="Momentum4D")

            met = df["met"]
            met = ak.with_name(met, name="Momentum4D")

            boost_jets = utils.get_vector_sum(jets)
            boost = boost_jets + leptons + met

        return boost

    def boost_CM(self, objects_array, boost):
        objects_CM = objects_array.boost_p4(boost.neg3D)

        # Overwriting old pt by calling the function on the boosted object
        objects_CM["pt"] = objects_CM.pt
        objects_CM["eta"] = objects_CM.eta
        objects_CM["phi"] = objects_CM.phi

        return objects_CM

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
            df = ak.from_parquet(file)

            if (object_type == "boost"):
                objects = self.boost

            else:
                objects = ak.with_name(df[object_type], name="Momentum4D")
                objects = self.boost_CM(objects, self.boost)

            if object_type == "jets":
                objects = self.Reshape(objects, utils.struct_jets, 1)

            d_list = utils.to_flat_numpy(
                objects, self.fields[object_type], axis=1, allow_missing=False)

            if object_type == "jets":
                d_list = np.transpose(d_list, (0, 2, 1))
                mask = self.get_mask_pt(d_list)

            if (object_type == "lepton_reco" or object_type == "met" or object_type == "boost"):
                d_list = np.expand_dims(d_list, axis=1)
                mask = np.ones((d_list.shape[0], d_list.shape[1]))

            tensor_data = torch.tensor(d_list, dtype=torch.float)
            tensor_mask = torch.tensor(mask, dtype=torch.float)

            torch.save((tensor_mask, tensor_data),
                       self.processed_file_names(object_type))
            
    def processCartesian(self):
        
        objects = ["jets", "lepton_reco", "met"] # don't need boost
        
        # Don't need mask -> concat jet/lepton/met masks
        for file in self.raw_file_names:
            df = ak.from_parquet(file)

            for object_type in objects:
                
                objects = ak.with_name(df[object_type], name="Momentum4D")
                objects = self.boost_CM(objects, self.boost)

                if object_type == "jets":
                    objects = self.Reshape(objects, utils.struct_jets, 1)

                d_list = utils.to_flat_numpy(
                        objects, self.fields['recoParticles_Cartesian'], axis=1, allow_missing=False)

                if object_type == "jets":
                    d_list = np.transpose(d_list, (0, 2, 1))

                if (object_type == "lepton_reco" or object_type == "met"):
                    d_list = np.expand_dims(d_list, axis=1)

                tensor_data = torch.tensor(d_list, dtype=torch.float)

                if object_type == "jets":
                    recoParticlesCartesian = tensor_data
                else:
                    recoParticlesCartesian = torch.cat((recoParticlesCartesian, tensor_data), 1)
            
            
            torch.save((recoParticlesCartesian),
                           self.processed_file_names('recoParticles_Cartesian'))
            
    def scaleCartesianTensor(self):
        
        # get masks of 'recoParticlesCartesian'
        recoMask = torch.cat((self.mask_jets, self.mask_lepton, self.mask_met), dim=1)
        
        recoMaskNumpy = recoMask.numpy()
        recoNumpy = self.recoParticlesCartesian.numpy()
        
        for i in range(4):
            feature = recoNumpy[:,:,i]
            feature_masked = np.ma.masked_array(feature, mask=np.logical_not(recoMaskNumpy))

            feature_scaled = (feature_masked - feature_masked.mean())/feature_masked.std()
            feature_scaled[np.logical_not(recoMaskNumpy)] = 0
            
            # use only the ndarray from masked array
            feature_scaled = np.ma.getdata(feature_scaled)
            feature_tensor = torch.tensor(feature_scaled, dtype=torch.float).unsqueeze(dim=2)
            if i == 0:
                recoParticlesScaled = feature_tensor
            else:
                recoParticlesScaled = torch.cat((recoParticlesScaled, feature_tensor), dim=2)
        
        torch.save((recoParticlesScaled),
                           self.processed_file_names('recoParticlesScaled'))

    def __getitem__(self, index):
        
        if self.debug == True:
            return (self.mask_lepton[index], self.data_lepton[index], self.mask_jets[index],
                    self.data_jets[index], self.mask_met[index], self.data_met[index],
                    self.mask_boost[index], self.data_boost[index],
                    self.recoParticlesCartesian[index], self.recoParticlesScaled[index])
        
        return (getattr(self, field)[index] for field in self.reco_list)

    def __len__(self):
        size = len(self.mask_lepton)
        return size
