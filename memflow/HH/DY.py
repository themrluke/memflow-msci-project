import os
import torch
import awkward as ak
import numpy as np
from hepunits.units import MeV, GeV
from sklearn import preprocessing

from memflow.HH.base import HardBase, RecoDoubleLepton

class DYDoubleLeptonHardDataset(HardBase):
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
            'dy_hard'
        )

    def process(self):
        # Safety checks #
        # The leptons are created either from resonant or non-resonant
        # Jets should be from non-resonant quarks -> in ISR
        self.check_quantities(
            {
                'Z' : [0,1],
                (
                    'lep_from_Z',
                    'antilep_from_Z',
                ) : [0,2],
                (
                    'lep_from_nonres',
                    'antilep_from_nonres',
                ) : [0,2],
                (
                    'lep_from_Z',
                    'antilep_from_Z',
                    'lep_from_nonres',
                    'antilep_from_nonres',
                ) : 2,
            },
            verbose=True,
        )

        # Now going to use select_present_particles
        # because we either have res or nonres leptons
        # For now exclude the taus completely #
        #mask_tau_veto = np.logical_and.reduce(
        #    (
        #        self.data['lep_from_Z_pdgId'] != +15,
        #        self.data['antilep_from_Z_pdgId'] != -15,
        #        self.data['lep_from_nonres_pdgId'] != +15,
        #        self.data['antilep_from_nonres_pdgId'] != -15,
        #    )
        #)
        #print (f'From {len(mask_tau_veto)}, removing {sum(~mask_tau_veto)} tau decay events')
        #self.data.cut(mask_tau_veto)


        # Make generator info #
        #boost = self.make_boost(generator.x1,generator.x2)
        x1 = np.random.random((self.data.events,1))
        x2 = np.random.random((self.data.events,1))
        boost = self.make_boost(x1,x2)

        # Register ISR #
        # Need to be done before final states if a cut on N(ISR) is done
        self.register_ISR()

        # Make particles #
        self.data.make_particles(
            'final_states',
            {
                'px'  : [
                    'lep_from_Z_Px',
                    'antilep_from_Z_Px',
                    'lep_from_nonres_Px',
                    'antilep_from_nonres_Px',
                ],
                'py'  : [
                    'lep_from_Z_Py',
                    'antilep_from_Z_Py',
                    'lep_from_nonres_Py',
                    'antilep_from_nonres_Py',
                ],
                'pz'  : [
                    'lep_from_Z_Pz',
                    'antilep_from_Z_Pz',
                    'lep_from_nonres_Pz',
                    'antilep_from_nonres_Pz',
                ],
                'E'  : [
                    'lep_from_Z_E',
                    'antilep_from_Z_E',
                    'lep_from_nonres_E',
                    'antilep_from_nonres_E',
                ],
                'pdgId'  : [
                    'lep_from_Z_pdgId',
                    'antilep_from_Z_pdgId',
                    'lep_from_nonres_pdgId',
                    'antilep_from_nonres_pdgId',
                ],
            },
            lambda vec: vec.E > 0.,
        )


        self.register_particles(['final_states'])

        self.match_coordinates(boost,self.data['final_states']) # need to be done after the boost

        self.preprocess_particles(['final_states','ISR'])

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

class DYDoubleLeptonRecoDataset(RecoDoubleLepton):
    @property
    def processed_path(self):
        return os.path.join(
            os.getcwd(),
            'dy_reco',
        )

