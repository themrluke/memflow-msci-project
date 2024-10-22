import os
import torch
import awkward as ak
import numpy as np
from hepunits.units import MeV, GeV
from sklearn import preprocessing

from memflow.dataset.dataset import AbsDataset, HardDataset, RecoDataset
from memflow.dataset.preprocessing import (
    lowercutshift,
    logmodulus,
    SklearnScaler,
    PreprocessingPipeline,
    PreprocessingStep,
)

from IPython import embed

class Base:
    def __init__(self,coordinates='cartesian',apply_preprocessing=False,apply_boost=False,**kwargs):
        self.coordinates = coordinates
        self.apply_preprocessing = apply_preprocessing
        self.apply_boost = apply_boost
        assert self.coordinates in ['cartesian','cylindrical']

    @staticmethod
    def get_coordinates(obj):
        if all([f in obj.fields for f in ['pt','eta','phi','mass']]):
            return 'cylindrical'
        elif all([f in obj.fields for f in ['px','py','pz','E']]):
            return 'cartesian'
        else:
            raise RuntimeError(f'Could not find coordinates from {obj.fields}')

    def change_coordinates(self,obj):
        current_coord = self.get_coordinates(obj)
        if current_coord == 'cartesian' and self.coordinates == 'cylindrical':
            obj = self.cartesian_to_cylindrical(obj)
        if current_coord == 'cylindrical' and self.coordinates == 'cartesian':
            obj = self.cylindrical_to_cartesian(obj)
        return obj

    def select_objects(self,obj_names):
        mask = np.logical_and.reduce(
            (
                [self.data[f'{obj_name}_E']>=0 for obj_name in obj_names]
            )
        )
        print (f'Selecting {mask.sum()} events out of {len(mask)}')
        self.data.cut(mask)

    @property
    def energy(self):
        return 13000 * GeV

class RecoDoubleLepton(Base,RecoDataset):
    def __init__(self,**kwargs):
        Base.__init__(self,**kwargs)
        RecoDataset.__init__(self,**kwargs)

    def process(self):
        # Make particles #
        n_jets = 15
        jets = self.data.make_particles(
            'jets',
            {
                'px'      : [f'j{i}_Px' for i in range(1,n_jets+1)],
                'py'      : [f'j{i}_Py' for i in range(1,n_jets+1)],
                'pz'      : [f'j{i}_Pz' for i in range(1,n_jets+1)],
                'E'       : [f'j{i}_E' for i in range(1,n_jets+1)],
                'btag'    : [f'j{i}_btag' for i in range(1,n_jets+1)],
                'btagged' : [f'j{i}_btagged' for i in range(1,n_jets+1)],
            },
            lambda vec: vec.E > 0.,
        )
        n_e = 4
        electrons = self.data.make_particles(
            'electrons',
            {
                'px'      : [f'e{i}_Px' for i in range(1,n_e+1)],
                'py'      : [f'e{i}_Py' for i in range(1,n_e+1)],
                'pz'      : [f'e{i}_Pz' for i in range(1,n_e+1)],
                'E'       : [f'e{i}_E' for i in range(1,n_e+1)],
                'pdgId'   : [f'e{i}_pdgId' for i in range(1,n_e+1)],
                'charge'  : [f'e{i}_charge' for i in range(1,n_e+1)],
            },
            lambda vec: vec.E > 0.,
        )
        n_m = 4
        muons = self.data.make_particles(
            'muons',
            {
                'px'      : [f'm{i}_Px' for i in range(1,n_m+1)],
                'py'      : [f'm{i}_Py' for i in range(1,n_m+1)],
                'pz'      : [f'm{i}_Pz' for i in range(1,n_m+1)],
                'E'       : [f'm{i}_E' for i in range(1,n_m+1)],
                'pdgId'   : [f'm{i}_pdgId' for i in range(1,n_m+1)],
                'charge'  : [f'm{i}_charge' for i in range(1,n_m+1)],
            },
            lambda vec: vec.E > 0.,
        )
        met = self.data.make_particles(
            'met',
            {
                'px'      : ['met_Px'],
                'py'      : ['met_Py'],
                'pz'      : ['met_Pz'],
                'E'       : ['met_E'],
            },
        )

        # Cartesian to cylindrical #
        jets = self.change_coordinates(jets)
        electrons = self.change_coordinates(electrons)
        muons = self.change_coordinates(muons)
        met = self.change_coordinates(met)

        # Make boost #
        boost = self.make_boost(jets,electrons,muons,met)

        # Boost objects #
        if self.apply_boost:
            jets = self.boost(jets,boost)
            electrons = self.boost(electrons,boost)
            muons = self.boost(muons,boost)
            met = self.boost(met,boost)

        self.match_coordinates(boost,jets) # need to be done after the boost

        jets_padded, jets_mask = self.reshape(
            input = jets,
            value = 0.,
            ax = 1,
        )
        electrons_padded, electrons_mask = self.reshape(
            input = electrons,
            value = 0.,
            ax = 1,
        )
        muons_padded, muons_mask = self.reshape(
            input = muons,
            value = 0.,
            ax = 1,
        )

        self.match_coordinates(boost,jets) # need to be done after the boost


        # Get jet weights #
        N_events = ak.count(jets_padded,axis=0)[0]
        N_jets = ak.max(ak.count(jets,axis=1))
        weight_jets = torch.ones((N_events,N_jets))
        for i in range(N_jets):
            weight_jets[:,i] *= N_events / ak.sum(jets_mask,axis=0)[i]

        # Register objects #
        self.register_object(
            name = 'boost',
            obj = boost,
        )
        self.register_object(
            name = 'jets',
            obj = jets_padded,
            mask = jets_mask,
            #weights = weight_jets,
        )
        self.register_object(
            name = 'electrons',
            obj = electrons_padded,
            mask = electrons_mask,
        )
        self.register_object(
            name = 'muons',
            obj = muons_padded,
            mask = muons_mask,
        )
        self.register_object(
            name = 'met',
            obj = met,
        )

        # Preprocessing #
        if self.apply_preprocessing:
            if self.coordinates != 'cylindrical':
                raise NotImplementedError

            self.register_preprocessing_step(
                PreprocessingStep(
                    names = ['muons','electrons'],
                    scaler_dict = {
                        'pt' : lowercutshift(10),
                    },
                )
            )
            self.register_preprocessing_step(
                PreprocessingStep(
                    names = ['jets'],
                    scaler_dict = {
                        'pt' : lowercutshift(25),
                    },
                )
            )
            self.register_preprocessing_step(
                PreprocessingStep(
                    names = ['jets','electrons','muons','met'],
                    scaler_dict = {
                        'pt' : logmodulus(),
                        'mass' : logmodulus(),
                    },
                )
            )
            self.register_preprocessing_step(
                PreprocessingStep(
                    names = ['jets','electrons','muons','met'],
                    scaler_dict = {
                        'pt'   : SklearnScaler(preprocessing.StandardScaler()),
                        'eta'  : SklearnScaler(preprocessing.MinMaxScaler(feature_range=(-1, 1), clip=True)),
                        'phi'  : SklearnScaler(preprocessing.MinMaxScaler(feature_range=(-1, 1), clip=True)),
                        'mass' : SklearnScaler(preprocessing.StandardScaler()),
                    },
                    fields_select = [
                        ('pt','eta','phi','mass'),
                        ('pt','eta','phi','mass'),
                        ('pt','eta','phi','mass'),
                        ('pt','phi'),
                    ]
                )
            )

    @property
    def attention_idx(self):
        # leptons and met always there
        # jets only the first two (by SR selection always there)
        return {
            'jets' : [0,1]
        }
