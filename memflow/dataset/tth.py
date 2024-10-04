import os
import torch
import awkward as ak
import numpy as np
from hepunits.units import MeV, GeV
from sklearn import preprocessing

from memflow.dataset.dataset import GenDataset, RecoDataset
from memflow.dataset.preprocessing import (
    lowercutshift,
    logmodulus,
    SklearnScaler,
    PreprocessingPipeline,
    PreprocessingStep,
)

class ttHBase:
    def __init__(self,coordinates='cylindrical',apply_preprocessing=False,apply_boost=False,**kwargs):
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



M_GLUON = 1e-3
class ttHGenDataset(ttHBase,GenDataset):
    struct_gluon = ak.zip(
        {
            "pt": np.float32(M_GLUON),
            "eta": np.float32(0.),
            "phi": np.float32(0.),
            "mass": np.float64(M_GLUON),
            "pdgId": bool(0),
            "prov": -1
        },
        with_name='Momentum4D',
    )
    struct_partons = ak.zip(
		{
			"pt": np.float32(0),
            "eta": np.float32(0),
            "phi": np.float32(0),
            "mass": np.float64(0),
            "pdgId": bool(0),
            "prov": -1,
		},
		with_name='Momentum4D',
	)

    def __init__(self,**kwargs):
        ttHBase.__init__(self,**kwargs)
        GenDataset.__init__(self,**kwargs)

    @property
    def energy(self):
        return 13000 * GeV

    @property
    def initial_states_pdgid(self):
        return [21,21]

    @property
    def final_states_pdgid(self):
        return [25,6,-6,21]

#    @property
#    def final_states_object_name(self):
#        return ['higgs','top_leptonic','top_hadronic','ISR']

    @property
    def processed_path(self):
        return os.path.join(
            os.getcwd(),
            'tth_gen'
        )

    def process(self):
        # Make generator info #
        generator = self.data['generator_info']
        boost = self.make_boost(generator.x1,generator.x2)
        self.register_object(
            name = 'boost',
            obj = boost,
        )

        # Make partons #
        partons = self.data.make_particles('partons_vec','partons')

        partons_padded, partons_mask = self.reshape(
            input = partons,
            value = self.struct_partons,
            ax = 1,
        )
        self.register_object(
            name = 'partons',
            obj = partons_padded,
            mask = partons_mask,
        )


        # Make lepton_partons #
        leptons = self.data.make_particles('lepton_partons_vec','lepton_partons')
        self.register_object(
            name = 'leptons',
            obj = leptons,
        )

        # Make ME final states #
        higgs = self.data.make_particles('higgs_vec','higgs')
        if 'm' in higgs.fields:
            higgs['mass'] = higgs['m'] # renaming for convention
            del higgs['m']

        W_leptonic = leptons[:,0] + leptons[:,1]
        partons_from_W = partons[partons.prov==5]
        W_hadronic = partons_from_W[:,0] + partons_from_W[:,1]

        top_leptonic = W_leptonic + partons[partons.prov==3][:,0]
        top_hadronic = W_hadronic + partons[partons.prov==2][:,0]

        gluons,mask = self.reshape(
            input = partons[partons.prov==4],
            value = self.struct_gluon,
            ax = 1,
        )

        ## Boost objects #
        if self.apply_boost:
            higgs = self.boost(higgs,boost)
            W_leptonic = self.boost(W_leptonic,boost)
            W_hadronic = self.boost(W_hadronic,boost)
            top_leptonic = self.boost(top_leptonic,boost)
            top_hadronic = self.boost(top_hadronic,boost)
            gluons = self.boost(gluons,boost)

        # For PS #
        ME_fields = ['E','px','py','pz'] # in Rambo format
        self.register_object(
            name = 'higgs_ME',
            obj = self.data['higgs_vec'],
            fields = ME_fields,
        )
        self.register_object(
            name = 'W_leptonic_ME',
            obj = W_leptonic,
            fields = ME_fields,
        )
        self.register_object(
            name = 'W_hadronic_ME',
            obj = W_hadronic,
            fields = ME_fields,
        )
        self.register_object(
            name = 'top_leptonic_ME',
            obj = top_leptonic,
            fields = ME_fields,
        )
        self.register_object(
            name = 'top_hadronic_ME',
            obj = top_hadronic,
            fields = ME_fields,
        )
        self.register_object(
            name = 'ISR_ME',
            obj = gluons[:,0],
            mask = mask[:,0],
            fields = ME_fields,
        )

        # For transfer #
        if self.coordinates == 'cylindrical':
            gen_fields = ['pt','eta','phi','mass']
        if self.coordinates == 'cartesian':
            gen_fields = ['px','py','pz','E']
        self.register_object(
            name = 'higgs',
            obj = self.data['higgs_vec'],
            fields = gen_fields,
        )
        self.register_object(
            name = 'top_leptonic',
            obj = top_leptonic,
            fields = gen_fields,
        )
        self.register_object(
            name = 'top_hadronic',
            obj = top_hadronic,
            fields = gen_fields,
        )
        self.register_object(
            name = 'ISR',
            obj = gluons[:,0],
            mask = mask[:,0],
            fields = gen_fields,
        )

        # Preprocessing #
        if self.apply_preprocessing:
            if self.coordinates == 'cylindrical':
                self.register_preprocessing_step(
                    PreprocessingStep(
                        names = ['higgs','top_leptonic','top_hadronic','ISR'],
                        scaler_dict = {
                            'pt' : logmodulus(),
                            'mass' : logmodulus(),
                        },
                    )
                )
                self.register_preprocessing_step(
                    PreprocessingStep(
                        names = ['higgs','top_leptonic','top_hadronic','ISR'],
                        scaler_dict = {
                            'pt'   : SklearnScaler(preprocessing.StandardScaler()),
                            'eta'  : SklearnScaler(preprocessing.StandardScaler()),
                            'phi'  : SklearnScaler(preprocessing.StandardScaler()),
                            'mass' : SklearnScaler(preprocessing.StandardScaler()),
                        },
                    )
                )
            if self.coordinates == 'cartesian':
                self.register_preprocessing_step(
                    PreprocessingStep(
                        names = ['higgs','top_leptonic','top_hadronic','ISR'],
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
                        names = ['higgs','top_leptonic','top_hadronic','ISR'],
                        scaler_dict = {
                            'px' : SklearnScaler(preprocessing.StandardScaler()),
                            'py' : SklearnScaler(preprocessing.StandardScaler()),
                            'pz' : SklearnScaler(preprocessing.StandardScaler()),
                            'E'  : SklearnScaler(preprocessing.StandardScaler()),
                        },
                    )
                )


class ttHRecoDataset(ttHBase,RecoDataset):
    struct_jets = ak.zip(
        {
            "pt": np.float32(0),
            "eta": np.float32(0),
            "phi": np.float32(0),
            "btag": np.float32(0),
            "m": np.float64(0),
            "matched": bool(0),
            "prov_Thad": np.float32(0),
            "prov_Tlep": np.float32(0),
            "prov_H": np.float32(0),
            "prov": -1,
        },
        with_name='Momentum4D',
    )

    def __init__(self,**kwargs):
        ttHBase.__init__(self,**kwargs)
        RecoDataset.__init__(self,**kwargs)

    @property
    def energy(self):
        return 13000 * GeV

    @property
    def processed_path(self):
        return os.path.join(
            os.getcwd(),
            'tth_reco',
        )

    def process(self):
        # Get jets leptons and met #
        jets = self.data.make_particles('jets_vec','jets')
        leptons = self.data.make_particles('lepton_reco_vec','lepton_reco')
        lepton = leptons[:,0] # take leading one
        met = self.data.make_particles('met_vec','met')

        # Make boost #
        boost = self.make_boost(jets,lepton,met)

        ## Boost objects #
        if self.apply_boost:
            jets = self.boost(jets,boost)
            lepton = self.boost(lepton,boost)
            met = self.boost(met,boost)

        # Re-order jets #
        # Order : [b(H),b(H),b(thad),q(thad),q(thad),b(tlep),q(ISR),additional jets]
        # prov flag : [1,1,2,5,5,3,4,-1,...,-1]
        prov_flags = [1,2,5,3,4,-1]
        jets_padded, jets_mask = [],[]
        for prov_flag in prov_flags:
            j,m = self.reshape(
                input = jets[jets.prov == prov_flag],
                value = self.struct_jets,
                ax = 1,
            )
            jets_padded.append(j)
            jets_mask.append(m)
        jets_padded = ak.concatenate(jets_padded,axis=1)
        jets_mask = ak.concatenate(jets_mask,axis=1)

        self.match_coordinates(boost,jets) # need to be done after the boost

        # Register objects #
        self.register_object(
            name = 'boost',
            obj = boost,
        )
        self.register_object(
            name = 'jets',
            obj = jets_padded,
            mask = jets_mask,
        )
        self.register_object(
            name = 'lepton',
            obj = lepton,
        )
        self.register_object(
            name = 'met',
            obj = met,
        )
        # Preprocessing #
        if self.apply_preprocessing:
            self.register_preprocessing_step(
                PreprocessingStep(
                    names = ['jets'],
                    scaler_dict = {
                        'pt' : lowercutshift(30),
                    },
                )
            )
            self.register_preprocessing_step(
                PreprocessingStep(
                    names = ['lepton'],
                    scaler_dict = {
                        'pt' : lowercutshift(25),
                    },
                )
            )
            self.register_preprocessing_step(
                PreprocessingStep(
                    names = ['met'],
                    scaler_dict = {
                        'pt' : lowercutshift(20),
                    },
                )
            )
            self.register_preprocessing_step(
                PreprocessingStep(
                    names = ['jets','lepton','met'],
                    scaler_dict = {
                        'pt' : logmodulus(),
                        'm'  : logmodulus(),
                    },
                    fields_select = [
                        ('pt','m'),
                        ('pt','m'),
                        ('pt',),
                    ]
                )
            )
            self.register_preprocessing_step(
                PreprocessingStep(
                    names = ['jets','lepton','met'],
                    scaler_dict = {
                        'pt'   : SklearnScaler(preprocessing.StandardScaler()),
                        'eta'  : SklearnScaler(preprocessing.StandardScaler()),
                        'phi'  : SklearnScaler(preprocessing.StandardScaler()),
                        'm'    : SklearnScaler(preprocessing.StandardScaler()),
                    },
                    fields_select = [
                        ('pt','eta','phi','m'),
                        ('pt','eta','phi','m'),
                        ('pt','phi'),
                    ]
                )
            )

    @property
    def correlation_idx(self):
        # jets to be always included
        # [b(H),b(H),b(thad),q(thad),q(thad),b(tlep),q(ISR)]
        return {
            'jets' : [0,1,2,3,4,5,6]
        }
