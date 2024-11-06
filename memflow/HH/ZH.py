import os
import torch
import awkward as ak
import numpy as np
from hepunits.units import MeV, GeV
from sklearn import preprocessing

from memflow.HH.base import HardBase, RecoDoubleLepton


class ZHDoubleLeptonHardDataset(HardBase):
    @property
    def initial_states_pdgid(self):
        return [21,21]

    @property
    def final_states_pdgid(self):
        return [6,-6,11,-11]

    @property
    def final_states_object_name(self):
        return None
        #return ['b','bbar','l-','vbar','l+','v']

    @property
    def processed_path(self):
        return os.path.join(
            os.getcwd(),
            'zz_hard'
        )

    def process(self):
        # Safety checks #
        # Z(->ll)H(->bb)
        self.check_quantities(
            {
                'H' : 1,
                'Z' : 1,
                'bottom': 1,
                'antibottom': 1,
                'lep_plus': 1,
                'lep_minus': 1,
            },
            verbose=True,
        )

        # Now going to use select_present_particles
        # because we either have res or nonres leptons
        # For now exclude the taus completely #
        #mask_tau_veto = np.logical_and.reduce(
        #    (
        #        self.data['lep_plus_pdgId'] != +15,
        #        self.data['lep_minus_pdgId'] != +15,
        #    )
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
                    'lep_plus_Px',
                    'lep_minus_Px',
                ],
                'py'  : [
                    'bottom_Py',
                    'antibottom_Py',
                    'lep_plus_Py',
                    'lep_minus_Py',
                ],
                'pz'  : [
                    'bottom_Pz',
                    'antibottom_Pz',
                    'lep_plus_Pz',
                    'lep_minus_Pz',
                ],
                'E'  : [
                    'bottom_E',
                    'antibottom_E',
                    'lep_plus_E',
                    'lep_minus_E',
                ],
                'pdgId'  : [
                    'bottom_pdgId',
                    'antibottom_pdgId',
                    'lep_plus_pdgId',
                    'lep_minus_pdgId',
                ],

            },
            lambda vec: vec.E > 0.,
        )
        self.make_radiation_particles()
        self.match_coordinates(boost,self.data['final_states']) # need to be done after the boost

        self.register_particles(['final_states','ISR','FSR'])
        self.preprocess_particles(['final_states','ISR','FSR'])

        # Register gen level particles #
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

class ZHDoubleLeptonRecoDataset(RecoDoubleLepton):
    @property
    def processed_path(self):
        return os.path.join(
            os.getcwd(),
            'zz_reco',
        )

