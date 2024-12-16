import os
import numpy as np
import torch
import awkward as ak
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
    def __init__(self,build_dir,coordinates='cartesian',apply_preprocessing=False,apply_boost=False,**kwargs):
        self.build_dir = build_dir
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

    def check_quantities(self,quantities,verbose=False):
        assert isinstance(quantities,dict), f'Quantities must be a dict, got {type(quants)}'
        print ('Checking particle quantities')
        for part,n_req in quantities.items():
            # Required number of particles #
            if isinstance(n_req,int):
                n_req = set([n_req])
            elif isinstance(n_req,(tuple,list)):
                n_req = set(n_req)
            elif isinstance(n_req,set):
                pass
            else:
                raise NotImplementedError(f'Type {type(n_req)} not understood')
            # Particle names #
            if isinstance(part,str):
                part = tuple([part])
            elif isinstance(part,tuple):
                pass
            else:
                raise NotImplementedError(f'Type {type(part)} not understood')
            # Get number of particles from data #
            n_data = set(
                np.unique(
                    sum(
                        [
                            self.data[f'n_{p}']
                            for p in part
                        ]
                    )
                )
            )
            if verbose:
                print (f'\tParticle(s) {part} : required {n_req}, found {n_data}')
            if n_data != n_req:
                raise RuntimeError(f'Particle {part} should have {n_req} multiplicity, but found {n_data}')
        print ('... done, no problem found')

    def select_present_particles(self,obj_names):
        mask = np.logical_and.reduce(
            (
                [self.data[f'{obj_name}_E']>=0 for obj_name in obj_names]
            )
        )
        print (f'Selecting decay : {mask.sum()} events out of {len(mask)}')
        self.data.cut(mask)

    @property
    def energy(self):
        return 13000 * GeV

    @property
    def intersection_branch(self):
        return 'event'



class HardBase(Base,HardDataset):
    def __init__(self,n_ISR=None,**kwargs):
        self.n_ISR = n_ISR

        Base.__init__(self,**kwargs)
        HardDataset.__init__(self,**kwargs)

    def register_ISR(self):
        # Make particles #
        ISR = self.data.make_particles(
            'ISR',
            {
                'px'  : [
                    f'ISR_{i+1}_Px'
                    for i in range(15)
                ],
                'py'  : [
                    f'ISR_{i+1}_Py'
                    for i in range(15)
                ],
                'pz'  : [
                    f'ISR_{i+1}_Pz'
                    for i in range(15)
                ],
                'E'  : [
                    f'ISR_{i+1}_E'
                    for i in range(15)
                ],
                'pdgId'  : [
                    f'ISR_{i+1}_pdgId'
                    for i in range(15)
                ],
                'parent'  : [
                    f'ISR_{i+1}_parent'
                    for i in range(15)
                ],
            },
            lambda vec: np.logical_and.reduce(
                (
                    vec.parent >= 0,
                    vec.E > 0.,
                    abs(vec.eta) <= 8,
                )
            )
        )
        # order ISR by pt
        idx = ak.argsort(ISR.pt,ascending=False)
        ISR = ISR[idx]

        # Printout #
        max_ISR = ak.max(ak.num(ISR,axis=1))
        n_events = ak.num(ISR,axis=0)
        print (f'Out of {n_events} events :')
        for i in range(max_ISR+1):
            n_sel_ISR = ak.sum((ak.num(ISR,axis=1) == i))
            print (f'  - {n_sel_ISR:8d} [{n_sel_ISR/n_events*100:3.2f}%] events with n(ISR) = {i}')

        # Reshape based on request n_ISR #
        if self.n_ISR is not None:
            assert isinstance(self.n_ISR,int)
            assert self.n_ISR >= 0
            mask_ISR = ak.num(ISR,axis=1) == self.n_ISR
            print (f'Required {self.n_ISR} ISR : selecting {ak.sum(mask_ISR)} events')
            self.data.cut(mask_ISR)
            if self.n_ISR == 0:
                print ('Required ISR is 0, will not register it')
                return
        # Reshape #
        ISR_padded, ISR_mask = self.reshape(
            input = self.data['ISR'], # recalling from data to take the cut into consideration
            value = 0.,
            max_no = self.n_ISR,
        )

        # Register #
        if self.coordinates == 'cylindrical':
            fields = ['pt','eta','phi','mass','pdgId']
        if self.coordinates == 'cartesian':
            fields = ['px','py','pz','E','pdgId']
        self.register_object(
            name = 'ISR',
            obj = ISR_padded,
            mask = ISR_mask,
            fields = fields,
        )


    def register_particles(self,particles=[]):
        for particle in particles:
            if self.coordinates == 'cylindrical':
                fields = ['pt','eta','phi','mass','pdgId']
            if self.coordinates == 'cartesian':
                fields = ['px','py','pz','E','pdgId']
            obj = self.data[particle]
            obj = self.change_coordinates(obj)
            if self.apply_boost:
                obj = self.boost(obj,boost)
            obj_padded, obj_mask = self.reshape(
                input = obj,
                value = 0.,
            )
            self.register_object(
                name = particle,
                obj = obj_padded,
                mask = obj_mask,
                fields = fields
            )

    def finalize(self):
        particles = ['final_states']
        if 'ISR' in self._objects.keys():
            particles.append('ISR')

        if self.apply_preprocessing:
            #self.register_preprocessing_step(
            #    PreprocessingStep(
            #        names = particles,
            #        scaler_dict = {
            #            'pdgId'   : SklearnScaler(
            #                preprocessing.OneHotEncoder(
            #                    categories = [
            #                        # category indices must be sorted
            #                        np.array(
            #                            [
            #                                -16,-15,-14,-13,-12,-11,    # antileptons
            #                                -5,-4,-3,-2,-1,             # antiquarks
            #                                1,2,3,4,5,                  # quarks
            #                                11,12,13,14,15,16,          # leptons
            #                                21,                         # gluons
            #                            ]
            #                        )
            #                    ],
            #                    sparse_output = False,
            #                    handle_unknown = 'ignore', # ignores zero-padded missing particles
            #                ),
            #            ),
            #        },
            #    )
            #)
            self.register_preprocessing_step(
                PreprocessingStep(
                    names = particles,
                    scaler_dict = {
                        'pdgId'  : SklearnScaler(preprocessing.MinMaxScaler(feature_range=(-1, 1), clip=True)),
                    },
                )
            )


            if self.coordinates == 'cylindrical':
                self.register_preprocessing_step(
                    PreprocessingStep(
                        names = particles,
                        scaler_dict = {
                            'pt'   : logmodulus(),
                        },
                    )
                )
                self.register_preprocessing_step(
                    PreprocessingStep(
                        names = particles,
                        scaler_dict = {
                            'pt'   : SklearnScaler(preprocessing.StandardScaler()),
                            'eta'  : SklearnScaler(preprocessing.StandardScaler()),
                            'phi'  : SklearnScaler(preprocessing.StandardScaler()),
                            'm'    : SklearnScaler(preprocessing.StandardScaler()),
                        },
                    )
                )
            elif self.coordinates == 'cartesian':
                self.register_preprocessing_step(
                    PreprocessingStep(
                        names = particles,
                        scaler_dict = {
                            'px' : logmodulus(),
                            'py' : logmodulus(),
                            'pz' : logmodulus(),
                            'E'  : logmodulus(),
                        },
                    )
                )
                self.register_preprocessing_step(
                    PreprocessingStep(
                        names = particles,
                        scaler_dict = {
                            'px' : SklearnScaler(preprocessing.StandardScaler()),
                            'px' : SklearnScaler(preprocessing.StandardScaler()),
                            'px' : SklearnScaler(preprocessing.StandardScaler()),
                            'E'  : SklearnScaler(preprocessing.StandardScaler()),
                        },
                    )
                )
            else:
                raise RuntimeError



class RecoDoubleLepton(Base,RecoDataset):
    def __init__(self,topology,**kwargs):
        self.topology = topology
        assert self.topology in ['resolved','boosted']

        # Base classes #
        Base.__init__(self,**kwargs)
        RecoDataset.__init__(self,**kwargs)


    def process(self):
        # Make particles #
        n_jets = 6
        self.data.make_particles(
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
        n_e = 2
        self.data.make_particles(
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
        n_m = 2
        self.data.make_particles(
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
        self.data.make_particles(
            'met',
            {
                'px'      : ['met_Px'],
                'py'      : ['met_Py'],
                'pz'      : ['met_Pz'],
                'E'       : ['met_E'],
            },
        )
        # Make selection #
        print ('Initial reco events :',self.data.events)
        mask_leptons = (ak.num(self.data['electrons'].pt,axis=1) + ak.num(self.data['muons'].pt,axis=1)) == 2
        self.data.cut(mask_leptons)
        print ('Dilepton cut :',self.data.events)
        if self.topology == 'resolved':
            #mask_resolved = np.logical_and.reduce(
            #    (
            #        self.data['flag_SR']==1,
            #        self.data['n_AK4']>=2,
            #        self.data['n_AK4B']>=1,
            #    )
            #)
            mask_resolved = np.logical_and.reduce(
                (
                    ak.num(self.data['jets'],axis=1) >= 2,           # >= 2 jets
                    ak.sum(self.data['jets'].btagged,axis=1) >= 1,   # >= 1 btagged jets
                )
            )
            self.data.cut(mask_resolved)
            print ('Resolved events :',self.data.events)
        if self.topology == 'boosted':
            raise NotImplementedError


        # Get objects (for easier handling) #
        jets      = self.data['jets']
        electrons = self.data['electrons']
        muons     = self.data['muons']
        met       = self.data['met']


        # Change jet order #
        # Firs make sure they are btag-ordered #
        idx_btag = ak.argsort(jets.btag,ascending=False)
        jets = jets[idx_btag]
        # Keep the two first jet btag-ordered
        # Rest should be pt-ordered
        idx_pt = ak.argsort(jets.pt[:,2:],ascending=False)
        idx = ak.concatenate((idx_btag[:,:2],idx_pt+2),axis=1)
        jets = jets[idx]

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
            max_no = 4,
        )
        electrons_padded, electrons_mask = self.reshape(
            input = electrons,
            value = 0.,
        )
        muons_padded, muons_mask = self.reshape(
            input = muons,
            value = 0.,
        )
        # Order leptons (- then +)
        # Doing after the padding so in cases where 1 muon + 1 electron,
        # we still put the negative lepton first (because padded has charge 0)
        # Also need to reorder the mask
        idx_e = ak.argsort(electrons_padded.charge,ascending=True)
        electrons_padded = electrons_padded[idx_e]
        electrons_mask   = electrons_mask[idx_e]
        idx_m = ak.argsort(muons_padded.charge,ascending=True)
        muons_padded = muons_padded[idx_m]
        muons_mask   = muons_mask[idx_m]

        # Matc coordinates #
        self.match_coordinates(boost,jets) # need to be done after the boost


        # Get jet weights #
        N_events = ak.num(jets_padded,axis=0)
        N_jets = ak.max(ak.num(jets_padded,axis=1))
        weight_jets = torch.ones((N_events,N_jets))
        #for i in range(N_jets):
        #    weight_jets[:,i] *= N_events / ak.sum(jets_mask,axis=0)[i]


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

    def finalize(self):
        # Preprocessing #
        if self.apply_preprocessing:
            if self.coordinates != 'cylindrical':
                raise NotImplementedError

            #self.register_preprocessing_step(
            #    PreprocessingStep(
            #        names = ['jets','electrons','muons','met'],
            #        scaler_dict = {
            #            'pdgId'   : SklearnScaler(
            #                preprocessing.OneHotEncoder(
            #                    categories = [np.array([-15,-13,-11,0,+11,+13,+15])],
            #                    sparse_output = False,
            #                    handle_unknown = 'ignore', # ignores zero-padded missing particles
            #                ),
            #            ),
            #            'charge'   : SklearnScaler(
            #                preprocessing.OneHotEncoder(
            #                    categories = [np.array([-1,0,+1])],
            #                    sparse_output = False,
            #                    handle_unknown = 'ignore', # ignores zero-padded missing particles
            #                ),
            #            ),
            #        },
            #    )
            #)
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
                    fields_select = [
                        ('pt','mass'),
                        ('pt','mass'),
                        ('pt','mass'),
                        ('pt',),
                    ]

                )
            )
            self.register_preprocessing_step(
                PreprocessingStep(
                    names = ['electrons','muons','jets','met'],
                    scaler_dict = {
                        'pt'     : SklearnScaler(preprocessing.StandardScaler()),
                        'eta'    : SklearnScaler(preprocessing.MinMaxScaler(feature_range=(-1, 1), clip=True)),
                        'phi'    : SklearnScaler(preprocessing.MinMaxScaler(feature_range=(-1, 1), clip=True)),
                        'mass'   : SklearnScaler(preprocessing.StandardScaler()),
                        'pdgId'  : SklearnScaler(preprocessing.MinMaxScaler(feature_range=(-1, 1), clip=True)),
                        'charge' : SklearnScaler(preprocessing.MinMaxScaler(feature_range=(-1, 1), clip=True)),
                    },
                    fields_select = [
                        ('pt','eta','phi','mass','pdgId','charge'),
                        ('pt','eta','phi','mass','pdgId','charge'),
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
