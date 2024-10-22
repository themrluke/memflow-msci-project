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


class DYDoubleLeptonHardDataset(Base,HardDataset):
    def __init__(self,**kwargs):
        Base.__init__(self,**kwargs)
        HardDataset.__init__(self,**kwargs)

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
        # Apply the cuts to select ttbar fully leptonic #
        self.select_objects(
            [
                'lep_from_Z',
                'antilep_from_Z',
                'quark_from_nonres',
                'antiquark_from_nonres',
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
                    'lep_from_Z_Px',
                    'antilep_from_Z_Px',
                ],
                'py'  : [
                    'lep_from_Z_Py',
                    'antilep_from_Z_Py',
                ],
                'pz'  : [
                    'lep_from_Z_Pz',
                    'antilep_from_Z_Pz',
                ],
                'E'  : [
                    'lep_from_Z_E',
                    'antilep_from_Z_E',
                ],
                'pdgId'  : [
                    'lep_from_Z_pdgId',
                    'antilep_from_Z_pdgId',
                ],
            },
            lambda vec: vec.E > 0.,
        )
        self.data.make_particles(
            'bquarks',
            {
                'px'  : [
                    'quark_from_nonres_Px',
                    'antiquark_from_nonres_Px',
                ],
                'py'  : [
                    'quark_from_nonres_Py',
                    'antiquark_from_nonres_Py',
                ],
                'pz'  : [
                    'quark_from_nonres_Pz',
                    'antiquark_from_nonres_Pz',
                ],
                'E'  : [
                    'quark_from_nonres_E',
                    'antiquark_from_nonres_E',
                ],
                'pdgId'  : [
                    'quark_from_nonres_pdgId',
                    'antiquark_from_nonres_pdgId',
                ],
            },
            lambda vec: vec.E > 0.,
        )
        if self.coordinates == 'cylindrical':
            fields = ['pt','eta','phi','mass','pdgId']
        if self.coordinates == 'cartesian':
            fields = ['px','py','pz','E','pdgId']
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



class DYDoubleLeptonRecoDataset(RecoDoubleLepton):
    @property
    def processed_path(self):
        return os.path.join(
            os.getcwd(),
            'dy_reco',
        )

