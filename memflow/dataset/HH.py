import os
import torch
import awkward as ak
import numpy as np
from hepunits.units import MeV, GeV
from sklearn.preprocessing import scale, power_transform, quantile_transform

from memflow.dataset.dataset import AbsDataset, GenDataset, RecoDataset
from memflow.dataset.preprocessing import logmodulus, PreprocessingPipeline, PreprocessingStep

from IPython import embed

class HHBase:
    def __init__(self,coordinates='cartesian',preprocessing=False,apply_boost=False,**kwargs):
        self.coordinates = coordinates
        self.preprocessing = preprocessing
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

class HHGenDataset(HHBase,GenDataset):
    def __init__(self,**kwargs):
        HHBase.__init__(self,**kwargs)
        GenDataset.__init__(self,**kwargs)

    @property
    def energy(self):
        return 13000 * GeV

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
        # Preprocessor #
        if self.preprocessing:
            gen_preprocess = PreprocessingPipeline(
                [
                    PreprocessingStep(
                        {
                            'pt' : logmodulus,
                        }
                    ),
                    PreprocessingStep(
                        {
                            'pt'   : scale,
                            'eta'  : scale,
                            'phi'  : scale,
                            'mass' : scale,
                        }
                    )

                ]
            )
        else:
            gen_preprocess = None
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
                preprocessing = gen_preprocess,
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



class HHRecoDataset(HHBase,RecoDataset):
    def __init__(self,**kwargs):
        HHBase.__init__(self,**kwargs)
        RecoDataset.__init__(self,**kwargs)

    @property
    def energy(self):
        return 13000 * GeV

    @property
    def processed_path(self):
        return os.path.join(
            os.getcwd(),
            'hh_reco',
        )

    def process(self):
        # Preprocessing #
        if self.preprocessing:
            reco_preprocess = PreprocessingPipeline(
                [
                    PreprocessingStep(
                        {
                            'pt' : logmodulus,
                        }
                    ),
                    PreprocessingStep(
                        {
                            'pt'   : scale,
                            'eta'  : scale,
                            'phi'  : scale,
                            'mass' : scale,
                        }
                    )

                ]
            )
        else:
            reco_preprocess = None

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
        )[:,0]

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

        # Register objects #
        self.register_object(
            name = 'boost',
            obj = boost,
            preprocessing = reco_preprocess,
        )
        self.register_object(
            name = 'jets',
            obj = jets_padded,
            mask = jets_mask,
            preprocessing = reco_preprocess,
        )
        self.register_object(
            name = 'electrons',
            obj = electrons_padded,
            mask = electrons_mask,
            preprocessing = reco_preprocess,
        )
        self.register_object(
            name = 'muons',
            obj = muons_padded,
            mask = muons_mask,
            preprocessing = reco_preprocess,
        )
        self.register_object(
            name = 'met',
            obj = met,
            preprocessing = reco_preprocess,
        )

