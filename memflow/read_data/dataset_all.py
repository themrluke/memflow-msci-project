from .Dataset_Parton_Level import Dataset_PartonLevel
from .Dataset_Reco_Level import Dataset_RecoLevel
from .Dataset_PartonNoBoost_Level import Dataset_PartonLevel_NoBoost
from .Dataset_RecoNoBoost_Level import Dataset_RecoLevel_NoBoost
from .Dataset_PartonNoBoost_Level_newHiggs import Dataset_PartonLevel_NoBoost_newHiggs
import torch
from torch.utils.data import Dataset, DataLoader
from itertools import chain
from operator import itemgetter

class DatasetCombined(Dataset):
    def __init__(self, root, datasets=["partons_CM", "reco_CM"],
                 dev=None, debug=False, dtype=None, build=False, new_higgs=True,
                 reco_list_lab=['mask_lepton', 'data_lepton', 'mask_jets', 'data_jets', 'mask_met', 'data_met',
                            'mask_boost', 'data_boost', 'recoParticlesCartesian'],
                 parton_list_lab=['mean_log_data_higgs_t_tbar_ISR_cartesian',
                                  'std_log_data_higgs_t_tbar_ISR_cartesian', 'logScaled_data_higgs_t_tbar_ISR_cartesian'],
                 reco_list_cm=['mask_lepton', 'data_lepton', 'mask_jets', 'data_jets', 'mask_met', 'data_met',
                            'mask_boost', 'data_boost', 'recoParticlesCartesian'],
                 parton_list_cm=['phasespace_intermediateParticles', 'phasespace_rambo_detjacobian',
                               'mean_log_data_higgs_t_tbar_ISR_cartesian',
                              'std_log_data_higgs_t_tbar_ISR_cartesian', 'logScaled_data_higgs_t_tbar_ISR_cartesian']):

        self.datasets = [ ]
        print(">>Loading datasets")
        for dataset in datasets:
            if dataset == "partons_CM":
                print("Loading partons in CM")
                self.partons_CM= Dataset_PartonLevel(root, dev=dev, debug=debug, dtype=dtype, build=build, parton_list=parton_list_cm)
                self.datasets.append(self.partons_CM)

            elif dataset == "reco_CM":
                print("Loading reco in CM")
                self.reco_CM = Dataset_RecoLevel(root, dev=dev, debug=debug, dtype=dtype, build=build, reco_list=reco_list_cm)
                self.datasets.append(self.reco_CM)

            elif dataset == "partons_lab":
                print("Loading partons in LAB")
                if new_higgs:
                    self.partons_lab = Dataset_PartonLevel_NoBoost_newHiggs(root, dev=dev, debug=debug, dtype=dtype, build=build, parton_list=parton_list_lab)
                else:
                    self.partons_lab = Dataset_PartonLevel_NoBoost(root, dev=dev, debug=debug, dtype=dtype, build=build, parton_list=parton_list_lab)
                self.datasets.append(self.partons_lab)

            elif dataset == "reco_lab":
                print("Loading reco in LAB")
                self.reco_lab = Dataset_RecoLevel_NoBoost(root, dev=dev, debug=debug, dtype=dtype, build=build, reco_list=reco_list_lab)
                self.datasets.append(self.reco_lab)

            else:
                raise ValueError("Invalid dataset requested")

        print("Loaded datasets: ", datasets)
        
    def __getitem__(self, index):
        return list(chain.from_iterable(map(itemgetter(index),  self.datasets)))
    
    def __len__(self):
        return len(self.datasets[0])
