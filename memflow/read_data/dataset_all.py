from .Dataset_Parton_Level import Dataset_PartonLevel
from .Dataset_Reco_Level import Dataset_RecoLevel
from .Dataset_PartonNoBoost_Level import Dataset_PartonLevel_NoBoost
from .Dataset_RecoNoBoost_Level import Dataset_RecoLevel_NoBoost
import torch
from torch.utils.data import Dataset, DataLoader

class DatasetCombined(Dataset):
    def __init__(self, root, boost_CM=True, dev=None, debug=False, dtype=None, build=False,
                 reco_list=['mask_lepton', 'data_lepton', 'mask_jets', 'data_jets', 'mask_met', 'data_met',
                            'mask_boost', 'data_boost', 'recoParticlesCartesian'],
                 parton_list=['phasespace_intermediateParticles', 'phasespace_rambo_detjacobian',
                               'mean_log_data_higgs_t_tbar_ISR_cartesian',
                              'std_log_data_higgs_t_tbar_ISR_cartesian', 'logScaled_data_higgs_t_tbar_ISR_cartesian']):

        if boost_CM:
            print('Get data boosted in CM')
            self.reco_data = Dataset_RecoLevel(root, dev=dev, debug=debug, dtype=dtype, build=build, reco_list=reco_list)
            self.parton_data = Dataset_PartonLevel(root, dev=dev, debug=debug, dtype=dtype, build=build, parton_list=parton_list)
        else:
            print('Get data NOT boosted in CM (lab frame)')
            self.reco_data = Dataset_RecoLevel_NoBoost(root, dev=dev, debug=debug, dtype=dtype, build=build, reco_list=reco_list)
            self.parton_data = Dataset_PartonLevel_NoBoost(root, dev=dev, debug=debug, dtype=dtype, build=build, parton_list=parton_list)
            
        print('')
        
    def __getitem__(self, index):
        return *self.parton_data[index], *self.reco_data[index]

    def __len__(self):
        return len(self.parton_data)
