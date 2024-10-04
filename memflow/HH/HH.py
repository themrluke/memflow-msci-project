import os
import torch
import awkward as ak
import numpy as np
from hepunits.units import MeV, GeV
from sklearn import preprocessing

from memflow.dataset.dataset import AbsDataset, GenDataset, RecoDataset
from memflow.HH.base import Base, RecoDoubleLepton
from memflow.dataset.preprocessing import (
    lowercutshift,
    logmodulus,
    SklearnScaler,
    PreprocessingPipeline,
    PreprocessingStep,
)


class HHbbWWDoubleLeptonGenDataset(Base,GenDataset):
    def __init__(self,**kwargs):
        Base.__init__(self,**kwargs)
        GenDataset.__init__(self,**kwargs)

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
            'hh_gen'
        )

    def process(self):
        # Apply the cuts to select ttbar fully leptonic #
        self.select_objects(
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

        # Make generator info #
#        boost = self.make_boost(generator.x1,generator.x2)
        x1 = np.random.random((self.data.events,1))
        x2 = np.random.random((self.data.events,1))
        boost = self.make_boost(x1,x2)

        # Make particles #
        self.data.make_particles(
            'leptons',
            {
                'px'  : [
                    'lep_plus_from_W_Px',
                    'neutrino_from_W_Px',
                    'lep_minus_from_W_Px',
                    'antineutrino_from_W_Px',
                ],
                'py'  : [
                    'lep_plus_from_W_Py',
                    'neutrino_from_W_Py',
                    'lep_minus_from_W_Py',
                    'antineutrino_from_W_Py',
                ],
                'pz'  : [
                    'lep_plus_from_W_Pz',
                    'neutrino_from_W_Pz',
                    'lep_minus_from_W_Pz',
                    'antineutrino_from_W_Pz',
                ],
                'E'  : [
                    'lep_plus_from_W_E',
                    'neutrino_from_W_E',
                    'lep_minus_from_W_E',
                    'antineutrino_from_W_E',
                ],
                'pdgId'  : [
                    'lep_plus_from_W_pdgId',
                    'neutrino_from_W_pdgId',
                    'lep_minus_from_W_pdgId',
                    'antineutrino_from_W_pdgId',
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
        self.data.make_particles(
            'higgs',
            {
                'px'  : [
                    'H1_Px',
                    'H2_Px',
                ],
                'py'  : [
                    'H1_Py',
                    'H2_Py',
                ],
                'pz'  : [
                    'H1_Pz',
                    'H2_Pz',
                ],
                'E'  : [
                    'H1_E',
                    'H2_E',
                ],
                'pdgId'  : [
                    'H1_pdgId',
                    'H2_pdgId',
                ],
            },
            lambda vec: vec.E > 0.,
        )
        if self.coordinates == 'cylindrical':
            fields = ['pt','eta','phi','mass','pdgId']
        if self.coordinates == 'cartesian':
            fields = ['px','py','pz','E','pdgId']
        for name in ['leptons','higgs','bquarks']:
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

        # Register gen level particles #
        self.register_object(
            name = 'boost',
            obj = boost,
        )
        ME_fields = ['E','px','py','pz'] # in Rambo format
        self.register_object(
            name = 'b',
            obj = self.data['bquarks'][:,0],
            fields = ME_fields,
        )
        self.register_object(
            name = 'bbar',
            obj = self.data['bquarks'][:,1],
            fields = ME_fields,
        )
        self.register_object(
            name = 'l+',
            obj = self.data['leptons'][:,0],
            fields = ME_fields,
        )
        self.register_object(
            name = 'v',
            obj = self.data['leptons'][:,1],
            fields = ME_fields,
        )
        self.register_object(
            name = 'l-',
            obj = self.data['leptons'][:,2],
            fields = ME_fields,
        )
        self.register_object(
            name = 'vbar',
            obj = self.data['leptons'][:,3],
            fields = ME_fields,
        )

        # Preprocessing #
        if self.apply_preprocessing:
            if self.coordinates == 'cylindrical':
                self.register_preprocessing_step(
                    PreprocessingStep(
                        names = ['leptons','higgs','bquarks'],
                        scaler_dict = {
                            'pt'   : logmodulus(),
                        },
                    )
                )
                self.register_preprocessing_step(
                    PreprocessingStep(
                        names = ['leptons','higgs','bquarks'],
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



class HHbbWWDoubleLeptonRecoDataset(RecoDoubleLepton):
    @property
    def processed_path(self):
        return os.path.join(
            os.getcwd(),
            'hh_reco',
        )

