from memflow.read_data import utils
import os
import os.path
import torch
import numpy as np
import awkward as ak
from torch.utils.data import Dataset


class Dataset_RecoLevel(Dataset):
    def __init__(self, root, object_types=["jets", "lepton_reco", "met", "boost"], dev=None, debug=False,
                 dtype=None, build=False, reco_list=[]):

        self.fields = {
            "jets": ["pt", "eta", "phi", "btag","prov_Thad", "prov_Tlep", "prov_H", "prov"],
            "lepton_reco": ["pt", "eta", "phi"],
            "met": ["pt", "eta", "phi"],
            "boost": ["E", "px", "py", "pz"],
            "boost_objBoosted": ["E", "px", "py", "pz"],
            "recoParticles_Cartesian": ["E", "px", "py", "pz"]
        }

        print("\nRecoLevel")
        self.debug = debug
        
        self.root = root
        if root.endswith(".parquet"):
            self.rootDir = root.replace(".parquet","")
        else:
            self.rootDir = root

        if not os.path.exists(self.rootDir):
            build=True
            
        self.reco_list = reco_list

        os.makedirs(self.rootDir + "/processed_jets", exist_ok=True)
        self.object_types = object_types

        allObjects = self.object_types[:]
        allObjects.append('recoParticles_Cartesian')
        # if build flag set or number of files in processed jets directory is 0
        if (build or  len(os.listdir(self.rootDir + '/processed_jets/')) == 0):
            (self.boost, self.boost_objBoosted) = self.get_boost()

            for object_type in self.object_types:
                print("Create new file for " + object_type)
                self.process(object_type)

            print("Create new file for recoParticles_Cartesian")
            self.processCartesian()

            # Need these tensors for scaleObjects
            self.mask_jets, self.data_jets = torch.load(self.processed_file_names("jets"))
            self.mask_lepton, self.data_lepton = torch.load(self.processed_file_names("lepton_reco"))
            self.mask_met, self.data_met = torch.load(self.processed_file_names("met"))
            self.mask_boost, self.data_boost = torch.load(self.processed_file_names("boost"))
            self.recoParticlesCartesian = torch.load(self.processed_file_names("recoParticles_Cartesian"))

            print("Create new file for LogData")
            self.scaleObjects()

        self.mask_jets, self.data_jets = torch.load(self.processed_file_names("jets"))
        self.mask_lepton, self.data_lepton = torch.load(self.processed_file_names("lepton_reco"))
        self.mask_met, self.data_met = torch.load(self.processed_file_names("met"))
        self.mask_boost, self.data_boost = torch.load(self.processed_file_names("boost"))
        #self.mask_boost_objBoosted, self.data_boost_objBoosted = torch.load(self.processed_file_names("boost_objBoosted"))
        tensors_bydefault = ['mask_jets', 'data_jets', 'mask_lepton', 'data_lepton',
                            'mask_met', 'data_met', 'mask_boost', 'data_boost']
        
        print("Reading reco_level Files")
        if 'recoParticlesCartesian' in self.reco_list:
            print("Load recoParticles_Cartesian")
            self.recoParticlesCartesian = torch.load(self.processed_file_names("recoParticles_Cartesian"))
        
        if 'recoParticles' in self.reco_list:
            print("Load recoParticles")
            self.recoParticles = torch.load(self.processed_file_names('recoParticles'))
        
        if 'scaledLogJets' in self.reco_list:
            print("Load scaledLogJets")
            self.scaledLogJets, self.LogJets, self.meanJets, self.stdJets = torch.load(
                                                                self.processed_file_names('scaledLogJets'))

        if 'scaledLogLepton' in self.reco_list:
            print("Load scaledLogLepton")
            self.scaledLogLepton, self.LogLepton, self.meanLepton, self.stdLepton = \
                                            torch.load(self.processed_file_names('scaledLogLepton'))
        
        if 'scaledLogMet' in self.reco_list:
            print("Load scaledLogMet")
            self.scaledLogMet, self.LogMet, self.meanMet, self.stdMet = \
                                            torch.load(self.processed_file_names('scaledLogMet'))
        
        if 'scaledLogBoost' in self.reco_list:
            print("Load scaledLogBoost")
            self.scaledLogBoost, self.LogBoost, self.meanBoost, self.stdBoost = \
                                            torch.load(self.processed_file_names('scaledLogBoost'))
        
        if 'scaledLogRecoParticlesCartesian' in self.reco_list:
            print("Load scaledLogRecoParticlesCartesian")
            self.scaledLogRecoParticlesCartesian, self.LogRecoParticlesCartesian, self.meanRecoCartesian, self.stdRecoCartesian = \
                                            torch.load(self.processed_file_names('scaledLogRecoParticlesCartesian'))

        if 'scaledLogRecoParticles' in self.reco_list:
            print("Load scaledLogRecoParticles")
            self.scaledLogRecoParticles, self.LogRecoParticles, self.meanRecoParticles, self.stdRecoParticles = \
                                            torch.load(self.processed_file_names('scaledLogRecoParticles'))

        if torch.cuda.is_available():
            print("Parton: Move tensors to GPU memory")
            for field in self.reco_list:
                setattr(self, field, getattr(self, field).to(dev)) # move elements from reco_list to GPU memory
            for field in tensors_bydefault:
                setattr(self, field, getattr(self, field).to(dev)) # move elements from reco_list to GPU memory
            
        if dtype != None:
            for field in self.reco_list:
                setattr(self, field, getattr(self, field).to(dtype))
            for field in tensors_bydefault:
                setattr(self, field, getattr(self, field).to(dtype))
            

    @property
    def raw_file_names(self):
        return [self.root]

    def processed_file_names(self, type):

        return (self.rootDir + '/processed_jets/' + type + '_data.pt')

    def get_boost(self):

        for file in self.raw_file_names:
            df = ak.from_parquet(file)

            jets = df["jets"]
            jets = ak.with_name(jets, name="Momentum4D")

            leptons = df["lepton_reco"]
            leptons = ak.with_name(leptons, name="Momentum4D")[:,0] # takingthe leading lepton

            met = df["met"]
            met = ak.with_name(met, name="Momentum4D")

            boost_jets = utils.get_vector_sum(jets)
            boost = boost_jets + leptons + met

            jets_boosted = self.boost_CM(jets, boost)
            leptons_boosted = self.boost_CM(leptons, boost)
            met_boosted = self.boost_CM(met, boost)

            jets_sum = utils.get_vector_sum(jets_boosted)
            boost_objBoosted = jets_sum + leptons_boosted + met_boosted

        return boost, boost_objBoosted

    def boost_CM(self, objects_array, boost):
        objects_CM = objects_array.boost_p4(boost.neg3D)

        # Overwriting old pt by calling the function on the boosted object
        objects_CM["pt"] = objects_CM.pt
        objects_CM["eta"] = objects_CM.eta
        objects_CM["phi"] = objects_CM.phi

        return objects_CM

    def Reshape(self, input, value, ax, max_no=None):
        if max_no is None:
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

            elif (object_type == "boost_objBoosted"):
                objects = self.boost_objBoosted

            else:
                objects = ak.with_name(df[object_type], name="Momentum4D")
                objects = self.boost_CM(objects, self.boost)

            if object_type == "jets":
                objects = self.Reshape(objects, utils.struct_jets, 1, max_no=16)

            if object_type == "lepton_reco":
                # taking the leading lepton
                objects = objects[:,0]
            
            d_list = utils.to_flat_numpy(
                objects, self.fields[object_type], axis=1, allow_missing=False)

            if object_type == "jets":
                d_list = np.transpose(d_list, (0, 2, 1))
                mask = self.get_mask_pt(d_list)

            if (object_type == "lepton_reco" or object_type == "met" or object_type == "boost" or object_type == "boost_objBoosted"):
                d_list = np.expand_dims(d_list, axis=1)
                mask = np.ones((d_list.shape[0], d_list.shape[1]))
            
            if (object_type == "lepton_reco" or object_type == "met"):
                #missing fields
                missing_fields = np.zeros((d_list.shape[0], d_list.shape[1],
                                           len(self.fields["jets"])-len(self.fields[object_type])
                                           ))
                d_list = np.concatenate((d_list, missing_fields), axis=2)
                
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
                    objects = self.Reshape(objects, utils.struct_jets, 1, 16)

                if object_type == "lepton_reco":
                    # only the leading lepton
                    objects = objects[:,0]
                    
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
    
    
    def scaleTensor(self, object_tensor, mask, isCartesian=False):
        
        # get masks of 'recoParticlesCartesian'
        
        maskNumpy = mask.numpy()
        objectNumpy = object_tensor.numpy()
        
        if isCartesian:
            # case [E,px,py,pz] or [x,y,z,t]
            no_elements = 4
            log_objectNumpy = np.sign(objectNumpy)*np.log(1+np.abs(objectNumpy)) 
            log_objectTensor = torch.tensor(log_objectNumpy, dtype=torch.float)
        else:
            # case ["pt", "eta", "phi", "btag", + prov
            no_elements = 3
            pt = objectNumpy[:,:,0]
            log_pt = np.log(1+pt) #sign and abs not necessary
            objectNumpy[:,:,0] = log_pt
            
            log_objectTensor = torch.tensor(objectNumpy, dtype=torch.float)
            
        for i in range(no_elements):
            feature = log_objectTensor[:,:,i].numpy()
            feature_masked = np.ma.masked_array(feature, mask=np.logical_not(maskNumpy))

            feature_scaled = (feature_masked - feature_masked.mean())/feature_masked.std()
            feature_scaled[np.logical_not(maskNumpy)] = 0
            
            # use only the ndarray from masked array
            feature_scaled = np.ma.getdata(feature_scaled)
            feature_tensor = torch.tensor(feature_scaled, dtype=torch.float).unsqueeze(dim=2)
            mean_tensor = torch.tensor(feature_masked.mean(), dtype=torch.float).reshape(1)
            std_tensor = torch.tensor(feature_masked.std(), dtype=torch.float).reshape(1)
            if i == 0:
                logScaled_objectTensor = feature_tensor
                meanLogTensor = mean_tensor
                stdLogTensor = std_tensor
            else:
                logScaled_objectTensor = torch.cat((logScaled_objectTensor, feature_tensor), dim=2)
                meanLogTensor = torch.cat((meanLogTensor, mean_tensor), dim=0)
                stdLogTensor = torch.cat((stdLogTensor, std_tensor), dim=0)
        
        # overwrite in object_tensor the new logScaled_objectTensor (to keep btag and prov)
        # then reassign logScaled_objectTensor (because this is the object returned)
        if isCartesian == False:
            logScaled_objectTensor = torch.cat((logScaled_objectTensor, object_tensor[:,:,3:]), dim=2)
            
        return meanLogTensor, stdLogTensor, logScaled_objectTensor, log_objectTensor
                
        
    def scaleObjects(self):
        
        meanJets, stdJets, scaledLogJets, LogJets = self.scaleTensor(self.data_jets,
                                                                     self.mask_jets, isCartesian=False)
        meanLepton, stdLepton, scaledLogLepton, LogLepton = self.scaleTensor(self.data_lepton, self.mask_lepton,
                                                                isCartesian=False)
        meanMet, stdMet, scaledLogMet, LogMet = self.scaleTensor(self.data_met, self.mask_met, isCartesian=False)
        
        meanBoost, stdBoost, scaledLogBoost, LogBoost = self.scaleTensor(self.data_boost,
                                                                         self.mask_boost, isCartesian=True)
        
        recoMask = torch.cat((self.mask_jets, self.mask_lepton, self.mask_met), dim=1)
        meanRecoCartesian, stdRecoCartesian, scaledLogRecoParticlesCartesian, LogRecoParticlesCartesian = \
                                           self.scaleTensor(self.recoParticlesCartesian,
                                                            recoMask, isCartesian=True)
        
        recoParticles = torch.cat((self.data_jets, self.data_lepton, self.data_met), dim=1)
        meanRecoParticles, stdRecoParticles, scaledLogRecoParticles, LogRecoParticles = self.scaleTensor(recoParticles,
                                                                                  recoMask, isCartesian=False)
        
        # attach btag and prov to cartesian full tensor
        scaledLogRecoParticlesCartesian = torch.cat((scaledLogRecoParticlesCartesian, recoParticles[:,:,3:]), dim=2)
        LogRecoParticlesCartesian = torch.cat((LogRecoParticlesCartesian, recoParticles[:,:,3:]), dim=2)
        
        torch.save((recoParticles), self.processed_file_names('recoParticles'))
        
        #torch.save((scaledLogJets, LogJets, meanJets, stdJets), self.processed_file_names('scaledLogJets'))
        #torch.save((scaledLogLepton, LogLepton, meanLepton, stdLepton), self.processed_file_names('scaledLogLepton'))
        #torch.save((scaledLogMet, LogMet, meanMet, stdMet), self.processed_file_names('scaledLogMet'))
        #torch.save((scaledLogBoost, LogBoost, meanBoost, stdBoost), self.processed_file_names('scaledLogBoost'))
        torch.save((scaledLogRecoParticlesCartesian, LogRecoParticlesCartesian, meanRecoCartesian, stdRecoCartesian),
                       self.processed_file_names('scaledLogRecoParticlesCartesian'))
        torch.save((scaledLogRecoParticles, LogRecoParticles, meanRecoParticles, stdRecoParticles),
                       self.processed_file_names('scaledLogRecoParticles'))

    def __getitem__(self, index):
        
        if self.debug == True:
            return (self.mask_lepton[index], self.data_lepton[index], self.mask_jets[index],
                    self.data_jets[index], self.mask_met[index], self.data_met[index],
                    self.mask_boost[index], self.data_boost[index],
                    self.scaledLogRecoParticlesCartesian[index], self.LogRecoParticlesCartesian[index])
        
        return [getattr(self, field)[index] if 'mean' not in field and 'std' not in field \
                    else getattr(self, field) for field in self.reco_list]

    def __len__(self):
        size = len(self.mask_lepton)
        return size

