import os
import torch
import awkward as ak
import numpy as np
from hepunits.units import MeV, GeV
from sklearn import preprocessing

from memflow.HH.base import HardBase, RecoDoubleLepton

class TTDoubleLeptonHardDataset(HardBase):
    @property
    def initial_states_pdgid(self):
        return [21,21]

    @property
    def final_states_pdgid(self):
        return [6,-6,11,-12,-11,12]

    @property
    def final_states_object_name(self):
        return None
        #return ['b','bbar','l-','vbar','l+','v']

    @property
    def processed_path(self):
        return os.path.join(
            self.build_dir,
            'tt_hard'
        )

    def process(self):
        # Safety checks #
        # Need to chechk both, and before applying the selection
        self.check_quantities(
            {
                'top' : 1,
                'antitop' : 1,
                'bottom' : 1,
                'antibottom' : 1,
                'W_plus_from_top' : 1,
                'W_minus_from_antitop' : 1,
                (
                    'lep_plus',
                    'lep_minus',
                    'neutrino',
                    'antineutrino',
                ) : 4,
                (
                    'quark_up',
                    'quark_down',
                    'antiquark_up',
                    'antiquark_down',
                ) : 0,
            },
            verbose=True,
        )

        # Apply the cuts to select ttbar fully leptonic #
        self.select_present_particles(
            [
                'top',
                'antitop',
                'W_plus_from_top',
                'W_minus_from_antitop',
                'bottom',
                'antibottom',
                'lep_plus',
                'lep_minus',
                'neutrino',
                'antineutrino',
            ]
        )

        # For now exclude the tau decays of the Ws #
        #mask_tau_veto = np.logical_and(
        #    self.data['lep_plus_pdgId'] != -15,
        #    self.data['lep_minus_pdgId'] != +15,
        #)
        #print (f'From {len(mask_tau_veto)}, removing {sum(~mask_tau_veto)} tau decay events')
        #self.data.cut(mask_tau_veto)


        # Make generator info #
        boost = self.make_boost(
            self.data['Generator_x1'],
            self.data['Generator_x2'],
        )

        # Register ISR #
        # Need to be done before final states if a cut on N(ISR) is done
        self.register_ISR()

        # Make particles #
        self.data.make_particles(
            'final_states',
            {
                'px'  : [
                    'bottom_Px',
                    'antibottom_Px',
                    'lep_plus_Px',
                    'neutrino_Px',
                    'lep_minus_Px',
                    'antineutrino_Px',
                ],
                'py'  : [
                    'bottom_Py',
                    'antibottom_Py',
                    'lep_plus_Py',
                    'neutrino_Py',
                    'lep_minus_Py',
                    'antineutrino_Py',
                ],
                'pz'  : [
                    'bottom_Pz',
                    'antibottom_Pz',
                    'lep_plus_Pz',
                    'neutrino_Pz',
                    'lep_minus_Pz',
                    'antineutrino_Pz',
                ],
                'E'  : [
                    'bottom_E',
                    'antibottom_E',
                    'lep_plus_E',
                    'neutrino_E',
                    'lep_minus_E',
                    'antineutrino_E',
                ],
                'pdgId'  : [
                    'bottom_pdgId',
                    'antibottom_pdgId',
                    'lep_plus_pdgId',
                    'neutrino_pdgId',
                    'lep_minus_pdgId',
                    'antineutrino_pdgId',
                ],
            },
            lambda vec: vec.E > 0.,
        )
        self.register_particles(['final_states'])

        self.match_coordinates(boost,self.data['final_states']) # need to be done after the boost

        ## Register gen level particles #
        #self.register_object(
        #    name = 'boost',
        #    obj = boost,
        #)
        #ME_fields = ['E','px','py','pz'] # in Rambo format
        #self.register_object(
        #    name = 'b',
        #    obj = self.data['bquarks'][:,0],
        #    fields = ME_fields,
        #)
        #self.register_object(
        #    name = 'bbar',
        #    obj = self.data['bquarks'][:,1],
        #    fields = ME_fields,
        #)
        #self.register_object(
        #    name = 'l+',
        #    obj = self.data['leptons'][:,0],
        #    fields = ME_fields,
        #)
        #self.register_object(
        #    name = 'v',
        #    obj = self.data['leptons'][:,1],
        #    fields = ME_fields,
        #)
        #self.register_object(
        #    name = 'l-',
        #    obj = self.data['leptons'][:,2],
        #    fields = ME_fields,
        #)
        #self.register_object(
        #    name = 'vbar',
        #    obj = self.data['leptons'][:,3],
        #    fields = ME_fields,
        #)



class TTDoubleLeptonRecoDataset(RecoDoubleLepton):
    @property
    def processed_path(self):
        return os.path.join(
            self.build_dir,
            'tt_reco',
        )

