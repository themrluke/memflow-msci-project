import os
import torch
import awkward as ak
import numpy as np
from hepunits.units import MeV, GeV
from sklearn.preprocessing import scale, power_transform, quantile_transform

from memflow.dataset.dataset import GenDataset, RecoDataset
from memflow.dataset.preprocessing import logmodulus, PreprocessingPipeline, PreprocessingStep

M_GLUON = 1e-3
class ttHGenDataset(GenDataset):
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
        super().__init__(**kwargs)


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
        # Preprocessor #
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
            preprocessing = gen_preprocess,
        )


        # Make lepton_partons #
        leptons = self.data.make_particles('lepton_partons_vec','lepton_partons')
        self.register_object(
            name = 'leptons',
            obj = leptons,
            preprocessing = gen_preprocess,
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

        ME_fields = ['E','px','py','pz'] # in Rambo format
        self.register_object(
            name = 'higgs',
            obj = self.cylindrical_to_cartesian(self.data['higgs_vec']),
            fields = ME_fields,
        )
        self.register_object(
            name = 'W_leptonic',
            obj = W_leptonic,
            fields = ME_fields,
        )
        self.register_object(
            name = 'W_hadronic',
            obj = W_hadronic,
            fields = ME_fields,
        )
        self.register_object(
            name = 'top_leptonic',
            obj = top_leptonic,
            fields = ME_fields,
        )
        self.register_object(
            name = 'top_hadronic',
            obj = top_hadronic,
            fields = ME_fields,
        )
        self.register_object(
            name = 'ISR',
            obj = gluons[:,0],
            mask = mask[:,0],
            fields = ME_fields,
        )


class ttHRecoDataset(RecoDataset):
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
        super().__init__(**kwargs)

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
        # Preprocessing #
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
                        'm'    : scale,
                    }
                )
            ]
        )

        # Get jets leptons and met #
        jets = self.data.make_particles('jets_vec','jets')
        leptons = self.data.make_particles('lepton_reco_vec','lepton_reco')
        lepton = leptons[:,0] # take leading one
        met = self.data.make_particles('met_vec','met')

        # Make boost #
        boost = self.make_boost(jets,lepton,met)

        # Boost objects #
        jets = self.boost(jets,boost)
        lepton = self.boost(lepton,boost)
        met = self.boost(met,boost)

        jets_padded, jets_mask = self.reshape(
            input = jets,
            value = self.struct_jets,
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
            name = 'lepton',
            obj = lepton,
            preprocessing = reco_preprocess,
        )
        self.register_object(
            name = 'met',
            obj = met,
            preprocessing = reco_preprocess,
        )

    @property
    def correlation_idx(self):
        return {
            'jets' : [0,1,2,3]
        }
