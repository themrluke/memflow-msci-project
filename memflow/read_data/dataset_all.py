from .Dataset_Parton_Level import Dataset_PartonLevel
from .Dataset_Reco_Level import Dataset_RecoLevel
from .Dataset_PartonNoBoost_Level import Dataset_PartonLevel_NoBoost
from .Dataset_RecoNoBoost_Level import Dataset_RecoLevel_NoBoost
import torch
from torch.utils.data import Dataset, DataLoader
from itertools import chain
from operator import itemgetter

class DatasetCombined(Dataset):
    def __init__(self, root, datasets=["partons_CM", "reco_CM"],
                 dev=None, debug=False, dtype=None, build=False,
                 reco_list=['mask_lepton', 'data_lepton', 'mask_jets', 'data_jets', 'mask_met', 'data_met',
                            'mask_boost', 'data_boost', 'recoParticlesCartesian'],
                 parton_list=['phasespace_intermediateParticles', 'phasespace_rambo_detjacobian',
                               'mean_log_data_higgs_t_tbar_ISR_cartesian',
                              'std_log_data_higgs_t_tbar_ISR_cartesian', 'logScaled_data_higgs_t_tbar_ISR_cartesian']):

        self.datasets = [ ] 
        if "partons_CM" in datasets:
            print("Loading partons in CM")
            self.parton_data_CM = Dataset_PartonLevel(root, dev=dev, debug=debug, dtype=dtype, build=build, parton_list=parton_list)
            self.datasets.append(self.parton_data_CM)
                        
        if "reco_CM" in datasets:
            print("Loading reco in CM")
            self.reco_CM = Dataset_RecoLevel(root, dev=dev, debug=debug, dtype=dtype, build=build, reco_list=reco_list)
            self.datasets.append(self.reco_CM)
            
        if "partons_lab" in datasets:
            print("Loading parton in LAB")
            self.parton_lab = Dataset_PartonLevel_NoBoost(root, dev=dev, debug=debug, dtype=dtype, build=build, parton_list=parton_list)
            self.datasets.append(self.parton_lab)
            
        if "reco_lab" in datasets:
            print("Loading reco in LAB")
            self.reco_lab = Dataset_RecoLevel_NoBoost(root, dev=dev, debug=debug, dtype=dtype, build=build, reco_list=reco_list)
            self.datasets.append(self.reco_lab)
            
        print('')
        
    def __getitem__(self, index):
        return list(chain.from_iterable(map(itemgetter(index),  self.datasets)))
    
    def __len__(self):
        return len(self.datasets[0])
