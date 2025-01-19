import os
import torch
import awkward as ak
import numpy as np
from hepunits.units import MeV, GeV
from sklearn import preprocessing

from memflow.dataset.dataset import HardDataset, RecoDataset
from memflow.dataset.preprocessing import *


class ttHBase:
    def __init__(self,coordinates='cylindrical',apply_preprocessing=False,apply_boost=False,**kwargs):
        self.coordinates = coordinates
        self.apply_preprocessing = apply_preprocessing
        self.apply_boost = apply_boost
        assert self.coordinates in ['cartesian','cylindrical']

    @staticmethod
    def get_coordinates(obj):
        if all([f in obj.fields for f in ['pt','eta','phi','mass']]):
            print('get coordinates function returns CYLINDRICAL')
            return 'cylindrical'

        elif all([f in obj.fields for f in ['px','py','pz','E']]):
            print('get coordinates function returns CARTESIAN')
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
class ttHHardDataset(ttHBase, HardDataset):
    # This is a class for hard-scattering dataset
    # Will highlight below the all the required methods and properties
    # Some clues will be given on helper methods
    def __init__(self,**kwargs):
        # The kwargs are the args required by the base HardDataset class
        # (data,selection,default_features,build,device,dtype)
        # Any user requested argument can also be defined in the __init__ args

        # Call abstract class
        ttHBase.__init__(self,**kwargs)
        HardDataset.__init__(self,**kwargs)

        # Here save in self any argument you need to use further

    @property
    def energy(self):
        # Return the center-of-mass energy in GeV for the ttH process
        return 13000 * GeV  # Replace 13000 GeV with the actual value if different

    # Some properties are required for the ME integration
    # This is to be linked with Rambo
    # If you are not using it right now, you can just return None

    @property
    def initial_states_pdgid(self):
        # return the pdgids of the two initial state gluons
        return [21, 21]  # Initial particles (gluons)

    @property
    def final_states_pdgid(self):
        # return the pdgids of the process final states as they are in the ME
        # Assuming the final states of the ttH process are t, tbar, H
        return [25, 6, -6]  # PDG IDs for Higgs, top, anti-top

    @property
    def final_states_object_name(self):
        # names associated to the final states
        # these names will be fetched in the object registered in the process method
        # Define the final objects (Higgs, tops)
        return ["higgs", "top", "antitop"]

    @property
    def processed_path(self):
        # return path of where to store the tensors in case build=True
        # Directory to save processed ttH data
        return os.path.join(os.getcwd(), 'ttH_hard')

    @property
    def attention_idx(self):
        # Need to return a dict (or None if not needed)
        # - key : name of a registered object
        # - value : indices to consider in the attention mask (even if not present)

        # having no return {} will by default attend to all the particles
        # To deactivate attention, need to specify empty index:
        # e.g.: 'higgs': [],
        return {
            'higgs': [0], # Only 1 Higgs so use 0 index
            'tops': [0, 1], # There are 2 top quarks so indexes 0, 1
            'bottoms': [0, 1],
            'Ws': [0, 1],
            'quarks': [0, 1, 2, 3],
            'Zs': [0],
            'neutrinos': [0, 1, 2, 3],
        }

    # The process method is the one where the data object is treated
    # Objects from the awkward arrays are registered
    def process(self):

        # 1) : select the decay you are interested in for your process
        # -> make mask (data type dependent) and apply it

        mask = np.logical_and.reduce(
            # The numbers correspond to the expeted counts of particles in each step
            # Names are the keys in awkward array dataset self.data: print(data_hard)
            # These keys correspond to branches in the Parquet file
            [
                # Higgs decay : H->ZZ->4nu #
                ak.num(self.data['higgs_idx']) == 1,
                ak.num(self.data['Z_from_higgs_idx']) == 2,
                ak.num(self.data['neutrinos_from_Z_idx']) == 4,

                # top decay : t->b q qbar #
                ak.num(self.data['top_idx']) == 1,
                ak.num(self.data['W_plus_from_top_idx']) == 1,
                ak.num(self.data['quark_from_W_plus_idx']) == 1,
                ak.num(self.data['antiquark_from_W_plus_idx']) == 1,

                # antitop decay : tbar->bbar q qbar #
                ak.num(self.data['antitop_idx']) == 1,
                ak.num(self.data['W_minus_from_antitop_idx']) == 1,
                ak.num(self.data['quark_from_W_minus_idx']) == 1,
                ak.num(self.data['antiquark_from_W_minus_idx']) == 1,
            ]
        )
        print (f'Selecting {mask.sum()} events out of {len(mask)}')

        print(f'Before cut: {len(self.data)} events')
        self.data.cut(mask)
        self.events = self.data.events
        print(f'After cut: {self.events} events')


        # 2) : obtain boost from the x1 and x2 parton fractions
        # Note : this is only needed in the case of the ME integration, not the transfer-flow
        # x1 = self.data[<branch_x1>]
        # x2 = self.data[<branch_x2>]
        # boost = self.make_boost(x1,x2)
        boost = self.make_boost(self.data['Generator_x1'],self.data['Generator_x2'])
        # Note : for the ME case, you need to register the boost as it is used in rambo
        # (see below to register)
        self.register_object(name='boost', obj=boost)

        # 3) : Make the particles
        # Obtain the awkward arrays with Momentum4D to make your particles

        # self.data.make_particles(
        #     <name>,
        #     <dict>,
        #     ...
        # )

        higgs = self.data.make_particles(
            'higgs', # This name to be later used in the registration step
            # The below names match the branch keys in self.data
            {
                'pt'  : [
                    'higgs_pt',
                ],
                'eta'  : [
                    'higgs_eta',
                ],
                'phi'  : [
                    'higgs_phi',
                ],
                'mass'  : [
                    'higgs_mass',
                ],
                'pdgId'  : [
                    'higgs_pdgId',
                ],
            },
        )

        tops = self.data.make_particles(
            'tops',
            {
                'pt'  : [
                    'top_pt',
                    'antitop_pt',
                ],
                'eta'  : [
                    'top_eta',
                    'antitop_eta',
                ],
                'phi'  : [
                    'top_phi',
                    'antitop_phi',
                ],
                'mass'  : [
                    'top_mass',
                    'antitop_mass',
                ],
                'pdgId'  : [
                    'top_pdgId',
                    'antitop_pdgId',
                ],
            },
        )

        bottoms = self.data.make_particles(
            'bottoms',
            {
                'pt'  : [
                    'bottom_pt',
                    'antibottom_pt',
                ],
                'eta'  : [
                    'bottom_eta',
                    'antibottom_eta',
                ],
                'phi'  : [
                    'bottom_phi',
                    'antibottom_phi',
                ],
                'mass'  : [
                    'bottom_mass',
                    'antibottom_mass',
                ],
                'pdgId'  : [
                    'bottom_pdgId',
                    'antibottom_pdgId',
                ],
            },
        )

        Ws = self.data.make_particles(
            'Ws',
            {
                'pt'  : [
                    'W_plus_from_top_pt',
                    'W_minus_from_antitop_pt',
                ],
                'eta'  : [
                    'W_plus_from_top_eta',
                    'W_minus_from_antitop_eta',
                ],
                'phi'  : [
                    'W_plus_from_top_phi',
                    'W_minus_from_antitop_phi',
                ],
                'mass'  : [
                    'W_plus_from_top_mass',
                    'W_minus_from_antitop_mass',
                ],
                'pdgId'  : [
                    'W_plus_from_top_pdgId',
                    'W_minus_from_antitop_pdgId',
                ],
            },
        )

        quarks = self.data.make_particles(
            'quarks',
            {
                'pt'  : [
                    'quark_from_W_plus_pt',
                    'antiquark_from_W_plus_pt',
                    'quark_from_W_minus_pt',
                    'antiquark_from_W_minus_pt',
                ],
                'eta'  : [
                    'quark_from_W_plus_eta',
                    'antiquark_from_W_plus_eta',
                    'quark_from_W_minus_eta',
                    'antiquark_from_W_minus_eta',
                ],
                'phi'  : [
                    'quark_from_W_plus_phi',
                    'antiquark_from_W_plus_phi',
                    'quark_from_W_minus_phi',
                    'antiquark_from_W_minus_phi',
                ],
                'mass'  : [
                    'quark_from_W_plus_mass',
                    'antiquark_from_W_plus_mass',
                    'quark_from_W_minus_mass',
                    'antiquark_from_W_minus_mass',
                ],
                'pdgId'  : [
                    'quark_from_W_plus_pdgId',
                    'antiquark_from_W_plus_pdgId',
                    'quark_from_W_minus_pdgId',
                    'antiquark_from_W_minus_pdgId',
                ],
            },
            pad_value = 0.,
        )

        Zs = self.data.make_particles(
            'Zs',
            {
                'pt'  : [
                    'Z_from_higgs_pt',
                ],
                'eta'  : [
                    'Z_from_higgs_eta',
                ],
                'phi'  : [
                    'Z_from_higgs_phi',
                ],
                'mass'  : [
                    'Z_from_higgs_mass',
                ],
                'pdgId'  : [
                    'Z_from_higgs_pdgId',
                ],
            },
        )

        neutrinos = self.data.make_particles(
            'neutrinos',
            {
                'pt'  : [
                    'neutrinos_from_Z_pt',
                ],
                'eta'  : [
                    'neutrinos_from_Z_eta',
                ],
                'phi'  : [
                    'neutrinos_from_Z_phi',
                ],
                'mass'  : [
                    'neutrinos_from_Z_mass',
                ],
                'pdgId'  : [
                    'neutrinos_from_Z_pdgId',
                ],
            },
        )


        # In case you want to boost your objects to the CM, see below
        # <obj_boosted> = self.boost(<obj>,boost)

        if self.apply_boost:
            higgs = self.boost(higgs, boost)
            tops = self.boost(tops, boost)
            bottoms = self.boost(bottoms, boost)
            Ws = self.boost(Ws, boost)
            quarks = self.boost(quarks, boost)
            Zs = self.boost(Zs, boost)
            neutrinos = self.boost(neutrinos, boost)

        # 4) : Register the objecs to do awkward->rectangular array->tensor

        # self.register_object(
        #     name = <name>,
        #     obj = <awkward_array>,
        #     fields = <list_of_fields>,
        # )

        # The <name> is the object name that will be used to identify the objects:
        #   - in the `selection` when you instantiate the class
        #   - in the preprocessing, to apply the correct functions to the objects
        # The <awkward_array> must be a Momentum4D type of array
        # The <list_of_fieds> is the list of fields from the awkward array to save in the tensor
        # this is necessary to keep track of the features in the transformer
        # typically this will be ['pt','eta','phi','mass',('pdgId')]
        # note: the fields can be from the array directly, or attributes of vector

        ME_fields = ['pt', 'eta', 'phi', 'mass', 'pdgId'] # ['E','px','py','pz'] in Rambo format

        self.register_object(
            name = 'higgs',
            obj = higgs,
            fields = ME_fields,
            )

        self.register_object(
            name = 'tops',
            obj = tops,
            fields = ME_fields,
            )

        self.register_object(
            name = 'bottoms',
            obj = bottoms,
            fields = ME_fields,
            )

        self.register_object(
            name = 'Ws',
            obj = Ws,
            fields = ME_fields,
            )

        self.register_object(
            name = 'quarks',
            obj = quarks,
            fields = ME_fields,
            )

        self.register_object(
            name = 'Zs',
            obj = Zs,
            fields = ME_fields,
            )

        self.register_object(
            name = 'neutrinos',
            obj = neutrinos,
            fields = ME_fields,
            )

    # Here you can modify the object tensors and any final change you want
    # Most importantly here you can register the preprocessing steps
    # We do it here rather than in the process, because that way we can load the raw
    # tensors and modify the preprocessing any time
    def finalize(self):
        # Register the preprocessing steps
        # You might want to apply multiple scaling in steps
        # Needs to be registered to be able to inverse all steps at the end

        # self.register_preprocessing_step(
        #     PreprocessingStep(
        #         names = <list_of_names>,
        #         scaler_dict = <scaler_dict>,
        #         fields_select = <list_fields>,
        #     )
        # )

        # # <list_of_names> is the list of registered names in step 4
        # you can either
        #   - include multiple objects,
        #     -> preprocessing will be fit (eg using a sklearn) scaler on all the objects
        #   - register preprocessing for particles individually
        #     -> each will be fit for each particle independently
        # <scaler_dict> : dict of
        #   - key : field as saved in the register objects
        #     -> apply different preprocessing for each feature
        #   - value : a class defined in memflow.dataset.preprocessing
        #       (see below)
        # <fields_select> : list (same length as <list_of_names>) of fields to consider
        # this can be useful when for example you include leptons and met,
        # and want to preprocess pt for both, but only eta for leptons

        # Available preprocessing :
        # - logmodulus : apply sign(x) * log(1+|x|)
        # - lowercutshift : rescale by the cut value
        # - SklearnScaler : anything from the sklearn.preprocessing
        #      Note : dimension change preprocessing allowed (eg onehot encoding)
        # - anything custom, see the logic in memflow.dataset.preprocessing

        if self.apply_preprocessing:

            if self.coordinates == 'cylindrical':
                self.register_preprocessing_step(
                    PreprocessingStep(
                        names=['higgs', 'tops', 'bottoms', 'Ws', 'quarks', 'Zs', 'neutrinos'],
                        scaler_dict={
                            'pt': logmodulus(),
                            'mass': logmodulus(),
                        }
                    )
                )
                self.register_preprocessing_step(
                    PreprocessingStep(
                        names=['higgs', 'tops', 'bottoms', 'Ws', 'quarks', 'Zs', 'neutrinos'],
                        scaler_dict={
                            'pt': SklearnScaler(preprocessing.StandardScaler()),
                            'eta': SklearnScaler(preprocessing.StandardScaler()),
                            'phi': SklearnScaler(preprocessing.StandardScaler()),
                            'mass': SklearnScaler(preprocessing.StandardScaler()),
                        }
                    )
                )

            if self.coordinates == 'cartesian':
                self.register_preprocessing_step(
                    PreprocessingStep(
                        names=['higgs', 'tops', 'bottoms', 'Ws', 'quarks', 'Zs', 'neutrinos'],
                        scaler_dict={
                            'px': logmodulus(),
                            'py': logmodulus(),
                            'pz': logmodulus(),
                            'E': logmodulus(),
                        }
                    )
                )
                self.register_preprocessing_step(
                    PreprocessingStep(
                        names=['higgs', 'tops', 'bottoms', 'Ws', 'quarks', 'Zs', 'neutrinos'],
                        scaler_dict={
                            'px': SklearnScaler(preprocessing.StandardScaler()),
                            'py': SklearnScaler(preprocessing.StandardScaler()),
                            'pz': SklearnScaler(preprocessing.StandardScaler()),
                            'E': SklearnScaler(preprocessing.StandardScaler()),
                        }
                    )
                )


class ttHRecoDataset(ttHBase, RecoDataset):

    # This is a class for reco-level dataset
    # Will highlight below the all the required methods and properties

    def __init__(self,**kwargs):
        # Call abstract class
        ttHBase.__init__(self,**kwargs)
        RecoDataset.__init__(self,**kwargs)

    # Most of the parts are similar to the HardDataset
    # Everything related to ME can be dismissed:
    # - initial_states_pdgid
    # - final_states_pdgid
    # - final_states_object_name

    @property
    def energy(self):
        # Return the center-of-mass energy in GeV for the ttH process
        return 13000 * GeV  # Replace 13000 GeV with the actual value if different

    @property
    def processed_path(self):
        # return path of where to store the tensors in case build=True
        # Directory to save processed ttH data
        return os.path.join(os.getcwd(), 'ttH_reco')

    @property
    def attention_idx(self):
        return None
        # {
                # 'met': [0],
                # 'jets': [0, 1],

        #     }

    # The process method is the one where the data object is treated
    # Objects from the awkward arrays are registered
    def process(self):

        # For now only consider events in SR
        mask = self.data['region'] == 0
        print('Before cut', self.data.events)
        self.data.cut(mask)
        print('After cut', self.data.events)

        # boost = self.make_boost(jets,electrons,muons,met)

        # 3) : Make the particles
        jets = self.data.make_particles(
            'jets',
            {
                'pt'      : 'cleanedJet_pt',
                'eta'     : 'cleanedJet_eta',
                'phi'     : 'cleanedJet_phi',
                'mass'    : 'cleanedJet_mass',
                'btag'    : 'cleanedJet_btagDeepFlavB',
            },
            pad_value = 0.,
        )
        met = self.data.make_particles(
            'met',
            {
                'pt'      : 'InputMet_pt',
                'eta'     : 0.,
                'phi'     : 'InputMet_phi',
                'mass'    : 0.,
            },
            pad_value = 0.,
        )

        # In case you expect a variable number of particles (not always the case)
        # <mask> can be obtained when reshaping the awkward array as below
        # This is not always needed, for example when you know you have the same number of that particle per event

        # <array_reshaped>, <array_mask> = self.reshape(
        #     input = <awkward_array>,
        #     value = <padding_value>, # number used to pad the missing entries (0 typically)
        #     max_no = <value/None> # maximum number of particles in the padded array
        #     # if max_no is None, will use the maximum number of particles in all events
        # )
        # self.register_object(
        #     name = <name>,
        #     obj = <array_reshaped>,
        #     fields = <list_of_fields>,
        #     mask = <array_mask>,
        # )

        jets_padded, jets_mask = self.reshape(
            input = jets,
            value = 0.
        )

        if self.apply_boost:
            raise ValueError('Do not use boost for now')


        # 4) : Register the objecs to do awkward->rectangular array->tensor

        self.register_object(
            name = 'jets',
            obj = jets_padded,
            mask = jets_mask,
            fields = ['pt', 'eta', 'phi', 'mass', 'btag'],
            )


        self.register_object(
            name = 'met',
            obj = met,
            fields = ['pt', 'eta', 'phi', 'mass'],
            )

    def finalize(self):
        # Register the preprocessing steps
        # You might want to apply multiple scaling in steps
        # Needs to be registered to be able to inverse all steps at the end

        # self.register_preprocessing_step(
        #     PreprocessingStep(
        #         names = <list_of_names>,
        #         scaler_dict = <scaler_dict>,
        #         fields_select = <list_fields>,
        #     )
        # )


        # # <list_of_names> is the list of registered names in step 4
        # you can either
        #   - include multiple objects,
        #     -> preprocessing will be fit (eg using a sklearn) scaler on all the objects
        #   - register preprocessing for particles individually
        #     -> each will be fit for each particle independently
        # <scaler_dict> : dict of
        #   - key : field as saved in the register objects
        #     -> apply different preprocessing for each feature
        #   - value : a class defined in memflow.dataset.preprocessing
        #       (see below)
        # <fields_select> : list (same length as <list_of_names>) of fields to consider
        # this can be useful when for example you include leptons and met,
        # and want to preprocess pt for both, but only eta for leptons

        # Available preprocessing :
        # - logmodulus : apply sign(x) * log(1+|x|)
        # - lowercutshift : rescale by the cut value
        # - SklearnScaler : anything from the sklearn.preprocessing
        #      Note : dimension change preprocessing allowed (eg onehot encoding)
        # - anything custom, see the logic in memflow.dataset.preprocessing

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
                    names = ['met'],
                    scaler_dict = {
                        'pt' : lowercutshift(20),
                    },
                )
            )
            self.register_preprocessing_step(
                PreprocessingStep(
                    names = ['jets','met'],
                    scaler_dict = {
                        'pt' : logmodulus(),
                        'mass'  : logmodulus(),
                    },
                    fields_select = [
                        ('pt','mass'),
                        ('pt',),
                    ]
                )
            )
            self.register_preprocessing_step(
                PreprocessingStep(
                    names = ['jets','met'],
                    scaler_dict = {
                        'pt'   : SklearnScaler(preprocessing.StandardScaler()),
                        'eta'  : SklearnScaler(preprocessing.MinMaxScaler(feature_range=(-1, 1), clip=True)),
                        'phi'  : SklearnScaler(preprocessing.MinMaxScaler(feature_range=(-1, 1), clip=True)),
                        'mass'    : SklearnScaler(preprocessing.StandardScaler()),
                    },
                    fields_select = [
                        ('pt','eta','phi','mass'),
                        ('pt','phi'),
                    ]
                )
            )