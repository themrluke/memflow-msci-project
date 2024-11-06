import os
import torch
import awkward as ak
import numpy as np
from hepunits.units import MeV, GeV
from sklearn import preprocessing

from memflow.HH.base import HardBase, RecoDoubleLepton


class HHbbWWDoubleLeptonHardDataset(HardBase):
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
            os.getcwd(),
            'hh_hard'
        )

    @property
    def attention_idx(self):
        return {
            'ISR' : [0]
        }

    def process(self):
        # Safety checks #
        # In HH DL samples, we have
        # - bbW(->lnu)W(->lnu)
        # - bbZ(->ll)Z(->nunu)
        # Need to chech both, and before applying the selection
        self.check_quantities(
            {
                'H1' : 2,
                'H2' : 2,
                'bottom' : 1,
                'antibottom' : 1,
                'W_plus' : [0,1],
                'W_minus' : [0,1],
                'Z1' : [0,2],
                'Z2' : [0,2],
                ('W_plus','W_minus','Z1'): 2,
                (
                    'lep_plus_from_W',
                    'lep_minus_from_W',
                    'neutrino_from_W',
                    'antineutrino_from_W',
                ) : [0,4],
                (
                    'quark_up_from_W',
                    'quark_down_from_W',
                    'antiquark_up_from_W',
                    'antiquark_down_from_W',
                ) : 0,
                (
                    'lep_plus_from_Z',
                    'lep_minus_from_Z',
                    'neutrino_from_Z',
                    'antineutrino_from_Z',
                ) : [0,4],
                (
                    'quark_up_from_Z',
                    'quark_down_from_Z',
                    'antiquark_up_from_Z',
                    'antiquark_down_from_Z',
                ) : 0,
            },
            verbose=True,
        )
        # Apply the cuts to select ttbar fully leptonic #
        self.select_present_particles(
            [
                'H1',
                'H2',
                'W_plus',
                'W_minus',
                'bottom',
                'antibottom',
                'lep_plus_from_W',
                'lep_minus_from_W',
                'neutrino_from_W',
                'antineutrino_from_W',
            ]
        )

        # For now exclude the tau decays of the Ws #
        #mask_tau_veto = np.logical_and(
        #    self.data['lep_plus_from_W_pdgId'] != -15,
        #    self.data['lep_minus_from_W_pdgId'] != +15,
        #)
        #print (f'From {len(mask_tau_veto)}, removing {sum(~mask_tau_veto)} tau decay events')
        #self.data.cut(mask_tau_veto)

        # Make generator info #
        #boost = self.make_boost(generator.x1,generator.x2)
        x1 = np.random.random((self.data.events,1))
        x2 = np.random.random((self.data.events,1))
        boost = self.make_boost(x1,x2)

        # Make particles #
        self.data.make_particles(
            'final_states',
            {
                'px'  : [
                    'bottom_Px',
                    'antibottom_Px',
                    'lep_plus_from_W_Px',
                    'neutrino_from_W_Px',
                    'lep_minus_from_W_Px',
                    'antineutrino_from_W_Px',
                ],
                'py'  : [
                    'bottom_Py',
                    'antibottom_Py',
                    'lep_plus_from_W_Py',
                    'neutrino_from_W_Py',
                    'lep_minus_from_W_Py',
                    'antineutrino_from_W_Py',
                ],
                'pz'  : [
                    'bottom_Pz',
                    'antibottom_Pz',
                    'lep_plus_from_W_Pz',
                    'neutrino_from_W_Pz',
                    'lep_minus_from_W_Pz',
                    'antineutrino_from_W_Pz',
                ],
                'E'  : [
                    'bottom_E',
                    'antibottom_E',
                    'lep_plus_from_W_E',
                    'neutrino_from_W_E',
                    'lep_minus_from_W_E',
                    'antineutrino_from_W_E',
                ],
                'pdgId'  : [
                    'bottom_pdgId',
                    'antibottom_pdgId',
                    'lep_plus_from_W_pdgId',
                    'neutrino_from_W_pdgId',
                    'lep_minus_from_W_pdgId',
                    'antineutrino_from_W_pdgId',
                ],
            },
            lambda vec: vec.E > 0.,
        )
        self.make_radiation_particles()
        self.match_coordinates(boost,self.data['final_states']) # need to be done after the boost

        self.register_particles(['final_states','ISR','FSR'])
        self.preprocess_particles(['final_states','ISR','FSR'])

#        # Register gen level particles #
#        self.register_object(
#            name = 'boost',
#            obj = boost,
#        )
#        ME_fields = ['E','px','py','pz'] # in Rambo format
#        self.register_object(
#            name = 'b',
#            obj = self.data['bquarks'][:,0],
#            fields = ME_fields,
#        )
#        self.register_object(
#            name = 'bbar',
#            obj = self.data['bquarks'][:,1],
#            fields = ME_fields,
#        )
#        self.register_object(
#            name = 'l+',
#            obj = self.data['leptons'][:,0],
#            fields = ME_fields,
#        )
#        self.register_object(
#            name = 'v',
#            obj = self.data['leptons'][:,1],
#            fields = ME_fields,
#        )
#        self.register_object(
#            name = 'l-',
#            obj = self.data['leptons'][:,2],
#            fields = ME_fields,
#        )
#        self.register_object(
#            name = 'vbar',
#            obj = self.data['leptons'][:,3],
#            fields = ME_fields,
#        )



class HHbbWWDoubleLeptonRecoDataset(RecoDoubleLepton):
    @property
    def processed_path(self):
        return os.path.join(
            os.getcwd(),
            'hh_reco',
        )

