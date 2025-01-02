import os
import torch
import awkward as ak
import numpy as np
from hepunits.units import MeV, GeV
from sklearn import preprocessing

from memflow.HH.base import HardBase, RecoDoubleLepton


class ZZDoubleLeptonHardDataset(HardBase):
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
            self.build_dir,
            'zz_hard'
        )

    def process(self):
        # Safety checks #
        # There is one or two Z
        # They can produce leptons or quarks, + nonres
        self.check_quantities(
            {
                'Z1' : [1,2],
                'Z2' : [1,2],
                (
                    'lep_from_Z1',
                    'antilep_from_Z1',
                ) : [0,2],
                (
                    'lep_from_Z2',
                    'antilep_from_Z2',
                ) : [0],
                (
                    'lep_from_nonres',
                    'antilep_from_nonres',
                ) : [0,2],
                (
                    'quark_from_Z1',
                    'antiquark_from_Z1',
                ) : [0,2],
                (
                    'quark_from_Z2',
                    'antiquark_from_Z2',
                ) : [0,2],
            },
            verbose=True,
        )
        # Make sure n(leptons)+n(quarks) == 4
        mask_4_decay = sum(
            [
                self.data['n_lep_from_Z1'],
                self.data['n_lep_from_Z2'],
                self.data['n_antilep_from_Z1'],
                self.data['n_antilep_from_Z2'],
                self.data['n_quark_from_Z1'],
                self.data['n_quark_from_Z2'],
                self.data['n_antiquark_from_Z1'],
                self.data['n_antiquark_from_Z2'],
                self.data['n_lep_from_nonres'],
                self.data['n_antilep_from_nonres'],
            ]
        ) == 4
        print (f'Selecting n(leptons)+n(quarks) == 4 : {ak.sum(mask_4_decay)} out of {ak.num(mask_4_decay,axis=0)} events')
        self.data.cut(mask_4_decay)

        # For now exclude the taus completely #
        #mask_tau_veto = np.logical_and.reduce(
        #    (
        #        self.data['lep_from_Z1_pdgId'] != +15,
        #        self.data['antilep_from_Z1_pdgId'] != -15,
        #        self.data['lep_from_Z2_pdgId'] != +15,
        #        self.data['antilep_from_Z2_pdgId'] != -15,
        #        self.data['lep_from_nonres_pdgId'] != +15,
        #        self.data['antilep_from_nonres_pdgId'] != -15,
        #    )
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
                    'lep_from_Z1_Px',
                    'antilep_from_Z1_Px',
                    'quark_from_Z1_Px',
                    'antiquark_from_Z1_Px',
                    'lep_from_Z2_Px',
                    'antilep_from_Z2_Px',
                    'quark_from_Z2_Px',
                    'antiquark_from_Z2_Px',
                    'lep_from_nonres_Px',
                    'antilep_from_nonres_Px',
                ],
                'py'  : [
                    'lep_from_Z1_Py',
                    'antilep_from_Z1_Py',
                    'quark_from_Z1_Py',
                    'antiquark_from_Z1_Py',
                    'lep_from_Z2_Py',
                    'antilep_from_Z2_Py',
                    'quark_from_Z2_Py',
                    'antiquark_from_Z2_Py',
                    'lep_from_nonres_Py',
                    'antilep_from_nonres_Py',
                ],
                'pz'  : [
                    'lep_from_Z1_Pz',
                    'antilep_from_Z1_Pz',
                    'quark_from_Z1_Pz',
                    'antiquark_from_Z1_Pz',
                    'lep_from_Z2_Pz',
                    'antilep_from_Z2_Pz',
                    'quark_from_Z2_Pz',
                    'antiquark_from_Z2_Pz',
                    'lep_from_nonres_Pz',
                    'antilep_from_nonres_Pz',
                ],
                'E'  : [
                    'lep_from_Z1_E',
                    'antilep_from_Z1_E',
                    'quark_from_Z1_E',
                    'antiquark_from_Z1_E',
                    'lep_from_Z2_E',
                    'antilep_from_Z2_E',
                    'quark_from_Z2_E',
                    'antiquark_from_Z2_E',
                    'lep_from_nonres_E',
                    'antilep_from_nonres_E',
                ],
                'pdgId'  : [
                    'lep_from_Z1_pdgId',
                    'antilep_from_Z1_pdgId',
                    'quark_from_Z1_pdgId',
                    'antiquark_from_Z1_pdgId',
                    'lep_from_Z2_pdgId',
                    'antilep_from_Z2_pdgId',
                    'quark_from_Z2_pdgId',
                    'antiquark_from_Z2_pdgId',
                    'lep_from_nonres_pdgId',
                    'antilep_from_nonres_pdgId',
                ],

            },
            lambda vec: vec.E > 0.,
        )
        self.register_particles(['final_states'])

        self.match_coordinates(boost,self.data['final_states']) # need to be done after the boost


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

class ZZDoubleLeptonRecoDataset(RecoDoubleLepton):
    @property
    def processed_path(self):
        return os.path.join(
            self.build_dir,
            'zz_reco',
        )

