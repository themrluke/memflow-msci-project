import os
import torch
import awkward as ak
import numpy as np
from hepunits.units import MeV, GeV
from sklearn import preprocessing

from memflow.dataset.dataset import AbsDataset, HardDataset, RecoDataset
from memflow.HH.base import Base, RecoDoubleLepton
from memflow.dataset.preprocessing import (
    lowercutshift,
    logmodulus,
    SklearnScaler,
    PreprocessingPipeline,
    PreprocessingStep,
)

class TTDoubleLeptonHardDataset(Base,HardDataset):
    def __init__(self,**kwargs):
        Base.__init__(self,**kwargs)
        HardDataset.__init__(self,**kwargs)

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
            'tt_hard'
        )

    def process(self):
        # Apply the cuts to select ttbar fully leptonic #
        self.select_objects(
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
        # Make generator info #
        #boost = self.make_boost(generator.x1,generator.x2)
        x1 = np.random.random((self.data.events,1))
        x2 = np.random.random((self.data.events,1))
        boost = self.make_boost(x1,x2)

        # Make particles #
        self.data.make_particles(
            'leptons',
            {
                'px'  : [
                    'lep_plus_Px',
                    'neutrino_Px',
                    'lep_minus_Px',
                    'antineutrino_Px',
                ],
                'py'  : [
                    'lep_plus_Py',
                    'neutrino_Py',
                    'lep_minus_Py',
                    'antineutrino_Py',
                ],
                'pz'  : [
                    'lep_plus_Pz',
                    'neutrino_Pz',
                    'lep_minus_Pz',
                    'antineutrino_Pz',
                ],
                'E'  : [
                    'lep_plus_E',
                    'neutrino_E',
                    'lep_minus_E',
                    'antineutrino_E',
                ],
                'pdgId'  : [
                    'lep_plus_pdgId',
                    'neutrino_pdgId',
                    'lep_minus_pdgId',
                    'antineutrino_pdgId',
                ],
            },
            lambda vec: vec.E > 0.,
        )
        self.data.make_particles(
            'bquarks',
            {
                'px'  : [
                    'bottom_Px',
                    'antibottom_Px',
                ],
                'py'  : [
                    'bottom_Py',
                    'antibottom_Py',
                ],
                'pz'  : [
                    'bottom_Pz',
                    'antibottom_Pz',
                ],
                'E'  : [
                    'bottom_E',
                    'antibottom_E',
                ],
                'pdgId'  : [
                    'bottom_pdgId',
                    'antibottom_pdgId',
                ],
            },
            lambda vec: vec.E > 0.,
        )
        if self.coordinates == 'cylindrical':
            fields = ['pt','eta','phi','mass']
        if self.coordinates == 'cartesian':
            fields = ['px','py','pz','E']
        for name in ['leptons','bquarks']:
            obj = self.data[name]
            obj = self.change_coordinates(obj)
            if self.apply_boost:
                obj = self.boost(obj,boost)
            obj_padded, obj_mask = self.reshape(
                input = obj,
                value = 0.,
                ax = 1,
            )
            self.register_object(
                name = name,
                obj = obj_padded,
                mask = obj_mask,
                fields = fields
            )
        self.match_coordinates(boost,obj) # need to be done after the boost

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

        # Preprocessing #
        if self.apply_preprocessing:
            if self.coordinates == 'cylindrical':
                self.register_preprocessing_step(
                    PreprocessingStep(
                        names = ['leptons','bquarks'],
                        scaler_dict = {
                            'pt'   : logmodulus(),
                        },
                    )
                )
                self.register_preprocessing_step(
                    PreprocessingStep(
                        names = ['leptons','bquarks'],
                        scaler_dict = {
                            'pt'   : SklearnScaler(preprocessing.StandardScaler()),
                            'eta'  : SklearnScaler(preprocessing.StandardScaler()),
                            'phi'  : SklearnScaler(preprocessing.StandardScaler()),
                            'm'    : SklearnScaler(preprocessing.StandardScaler()),
                        },
                    )
                )
            if self.coordinates == 'cartesian':
                raise NotImplementedError


class TTDoubleLeptonRecoDataset(RecoDoubleLepton):
    @property
    def processed_path(self):
        return os.path.join(
            os.getcwd(),
            'tt_reco',
        )

